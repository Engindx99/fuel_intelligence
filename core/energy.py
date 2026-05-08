import numpy as np

def safe_f(value):
    """Liste veya tekil değer olarak gelen konfigürasyon verisini float'a çevirir."""
    if isinstance(value, list): return float(value[0])
    return float(value)

class EnergyModel:
    def __init__(self, config):
        # Solver ve fiziksel sabitlerle tam uyum
        self.base_h_gs = safe_f(config['gas']['h_gs']) 
        self.sigma = 5.67037e-8  # W/(m^2.K^4)
        self.eps_g = safe_f(config['gas'].get('emissivity_g', 0.3))
        self.eps_s = safe_f(config['material'].get('emissivity_s', 0.85))
        
        # Etkin emisivite - Doğrudan config['energy']['eps_eff'] üzerinden çekiliyor
        if 'energy' in config and 'eps_eff' in config['energy']:
            self.eps_eff = safe_f(config['energy']['eps_eff'])
        else:
            # Yedek mekanizma (Fallback)
            self.eps_eff = (self.eps_g + self.eps_s) / 2.0

    def calculate_convection_coeff(self, current_fan_rate, nominal_fan=850.0):
        """
        Gaz debisine (fan) bağlı olarak h_gs katsayısını dinamik günceller.
        """
        fan_ratio = current_fan_rate / nominal_fan
        return self.base_h_gs * (fan_ratio**0.4)

    def calculate_radiation_flux(self, Tg, Ts, area):
        """
        Stefan-Boltzmann yasasına göre net radyasyon ısı akısını hesaplar.
        """
        return self.eps_eff * self.sigma * area * (Tg**4 - Ts**4)

    def get_reaction_heat(self, rates, m_dot_s, dH_vec):
        """
        Fırın boyunca gerçekleşen tüm reaksiyonların toplam ısı etkisini Watt cinsinden döndürür.
        """
        return np.sum(rates * dH_vec) * m_dot_s