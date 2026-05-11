import numpy as np

def safe_f(value):
    if isinstance(value, list):
        return float(value[0])
    return float(value)

class EnergyModel:
    def __init__(self, config):
        # Temel Konveksiyon Katsayısı
        self.base_h_gs = safe_f(config['gas']['h_gs'])
        
        # Fiziksel Sabitler
        self.sigma = 5.67037e-8  # W/(m^2.K^4)

        # Emisivite Değerleri (Gaz, Katı ve Refrakter)
        self.eps_g = safe_f(config['gas'].get('emissivity_g', 0.6))
        self.eps_s = safe_f(config['material'].get('emissivity_s', 0.85))
        # Refrakter emisivitesi (Tuğla genellikle 0.8-0.9 arasıdır)
        self.eps_w = safe_f(config['kiln'].get('emissivity_w', 0.9))

        # KRİTİK GÜNCELLEME: Gaz-Katı radyasyon etkileşimi için efektif emisivite
        # Basit ortalama yerine, gri gövde etkileşimini temsil eden efektif değer.
        self.eps_eff = (self.eps_g * self.eps_s)**0.5 # Geometrik ortalama daha baskındır

    def calculate_convection_coeff(self, current_fan_rate, nominal_fan=800.0):
        """Fan hızına bağlı olarak konveksiyon katsayısını hesaplar."""
        fan_ratio = current_fan_rate / nominal_fan
        # Döner fırınlarda Reynolds sayısı bağımlılığı genellikle 0.6-0.8 üssündedir.
        # 0.4 değeri transferi çok kısıtlıyor, 0.67 (2/3) daha gerçekçi bir türbülans temsilidir.
        return self.base_h_gs * (max(0.1, fan_ratio) ** 0.67)

    def calculate_radiation_flux(self, Tg, Ts, Tw, area):
        """
        Radyasyon akısını 3'lü denge modeline göre hesaplar.
        Enerjiyi Gaz-Katı ve Refrakter-Katı arasında paylaştırır.
        """
        # Sıcaklık değerlerini fiziksel sınırlar içinde tutuyoruz.
        Tg_c = np.clip(Tg, 100.0, 3200.0)
        Ts_c = np.clip(Ts, 100.0, 3200.0)
        Tw_c = np.clip(Tw, 100.0, 3200.0)

        # 1. Gazdan Katıya Doğrudan Radyasyon
        # View factor (f_gs): Gaz fırın hacmini doldurduğu için view factor 1.0'a yakındır.
        # 0.50 değeri enerjinin yarısını "yok sayıyordu", 0.85'e çekildi.
        f_gs = 0.85 
        q_gas_to_solid = f_gs * self.eps_eff * self.sigma * area * (Tg_c**4 - Ts_c**4 + 1e-6)

        # 2. Refrakterden Katıya Radyasyon (Re-radiation)
        # Tuğlaların (Tw) malzemeye (Ts) olan radyasyonu.
        # Fırın döndükçe tuğla yüzeyinin büyük kısmı malzemeye ısı pompalar.
        f_ws = 0.70 # View factor artırıldı
        
        # Bileşik emisivite formülü: 1 / (1/eps_s + 1/eps_w - 1) yaklaşımı
        # İki paralel/eğri yüzey arasındaki transfer için daha doğrudur.
        eps_combined = 1.0 / (1.0/self.eps_s + 1.0/self.eps_w - 1.0)
        
        q_wall_to_solid = f_ws * eps_combined * self.sigma * area * (Tw_c**4 - Ts_c**4 + 1e-6)

        # Toplam radyasyon yükü
        total_rad = q_gas_to_solid + q_wall_to_solid
        
        # Sayısal stabilite kontrolü
        return np.nan_to_num(total_rad, nan=0.0, posinf=1e15, neginf=-1e15)

    def get_reaction_heat(self, rates, m_dot_s, dH_vec):
        """Kimyasal reaksiyonların toplam ısıl yükünü (Watt) hesaplar."""
        # rates: (n_reactions, n_nodes), dH_vec: (n_reactions,)
        # np.sum(rates * dH_vec[:, None], axis=0) her node için toplam dH*rate verir
        q_rxn = np.sum(rates * dH_vec[:, None], axis=0) * m_dot_s
        return np.nan_to_num(q_rxn, nan=0.0)