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

        # Ortalama emisivite (Gaz-Katı etkileşimi için baz değer)
        self.eps = 0.5 * (self.eps_g + self.eps_s)

    def calculate_convection_coeff(self, current_fan_rate, nominal_fan=800.0):
        """Fan hızına bağlı olarak konveksiyon katsayısını hesaplar."""
        fan_ratio = current_fan_rate / nominal_fan
        # Sayısal kararlılık için fan_ratio'ya alt sınır koyuldu
        return self.base_h_gs * (max(0.1, fan_ratio) ** 0.4)

    def calculate_radiation_flux(self, Tg, Ts, Tw, area):
        """
        Radyasyon akısını 3'lü denge modeline göre hesaplar.
        Enerjiyi Gaz-Katı ve Refrakter-Katı arasında paylaştırır.
        """
        # KRİTİK GÜNCELLEME: Sayısal patlamayı (overflow/nan) engellemek için 
        # sıcaklık değerlerini fiziksel sınırlar içinde tutuyoruz (Clipping).
        # Bu işlem terminaldeki "overflow in power" hatasını kalıcı olarak çözer.
        Tg_c = np.clip(Tg, 100.0, 3000.0)
        Ts_c = np.clip(Ts, 100.0, 3000.0)
        Tw_c = np.clip(Tw, 100.0, 3000.0)

        # 1. Gazdan Katıya Doğrudan Radyasyon
        # View factor: Gazın katı üzerindeki doğrudan izdüşümü
        f_gs = 0.50 
        # T^4 hesaplamasından sonra oluşabilecek çok küçük gürültüler için 1e-6 eklendi
        q_gas_to_solid = f_gs * self.eps * self.sigma * area * (Tg_c**4 - Ts_c**4 + 1e-6)

        # 2. Refrakterden Katıya Radyasyon (Re-radiation)
        # Tuğlaların depoladığı ısının malzemeye transferi
        f_ws = 0.50 
        eps_ws = self.eps_w * self.eps_s # Bileşik emisivite
        q_wall_to_solid = f_ws * eps_ws * self.sigma * area * (Tw_c**4 - Ts_c**4 + 1e-6)

        # Toplam radyasyon yükü
        # Eğer hala nan oluşursa (ki bu clipping ile engellenmiştir), 0.0 döndürülür.
        total_rad = q_gas_to_solid + q_wall_to_solid
        return np.nan_to_num(total_rad, nan=0.0, posinf=1e12, neginf=-1e12)

    def get_reaction_heat(self, rates, m_dot_s, dH_vec):
        """Kimyasal reaksiyonların toplam ısıl yükünü (Watt) hesaplar."""
        # rates: (n_reactions, n_nodes), dH_vec: (n_reactions,)
        # np.sum(rates * dH_vec[:, None], axis=0) her node için toplam dH*rate verir
        # Reaksiyon ısılarını da nan/inf kontrolünden geçiriyoruz
        q_rxn = np.sum(rates * dH_vec[:, None], axis=0) * m_dot_s
        return np.nan_to_num(q_rxn, nan=0.0)