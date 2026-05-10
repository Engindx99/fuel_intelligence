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
        return self.base_h_gs * (fan_ratio ** 0.4)

    def calculate_radiation_flux(self, Tg, Ts, Tw, area):
        """
        Radyasyon akısını 3'lü denge modeline göre hesaplar.
        Enerjiyi Gaz-Katı ve Refrakter-Katı arasında paylaştırır.
        """
        # 1. Gazdan Katıya Doğrudan Radyasyon (Daha az agresif pay)
        # Gazın kütlesinden katı yüzeyine doğrudan çarpma
        f_gs = 0.50 # View factor: Gazın katı üzerindeki doğrudan izdüşümü
        q_gas_to_solid = f_gs * self.eps * self.sigma * area * (Tg**4 - Ts**4)

        # 2. Refrakterden Katıya Radyasyon (Re-radiation)
        # Tuğlaların depoladığı ısının malzemeye transferi
        # Bu kısım ısıl ataleti (inertia) temsil eder ve ısınmayı yavaşlatır
        f_ws = 0.50 # View factor: Refrakterden katıya yansıma payı
        eps_ws = self.eps_w * self.eps_s # Bileşik emisivite
        q_wall_to_solid = f_ws * eps_ws * self.sigma * area * (Tw**4 - Ts**4)

        # Toplam radyasyon yükü
        return q_gas_to_solid + q_wall_to_solid

    def get_reaction_heat(self, rates, m_dot_s, dH_vec):
        """Kimyasal reaksiyonların toplam ısıl yükünü (Watt) hesaplar."""
        # rates: (n_reactions, n_nodes), dH_vec: (n_reactions,)
        # np.sum(rates * dH_vec[:, None], axis=0) her node için toplam dH*rate verir
        return np.sum(rates * dH_vec[:, None], axis=0) * m_dot_s