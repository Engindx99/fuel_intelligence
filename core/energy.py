import numpy as np

def safe_f(value):
    if isinstance(value, list): return float(value[0])
    return float(value)

class EnergyModel:
    def __init__(self, config):
        # Temel katsayıyı config'den al
        self.base_h_gs = safe_f(config['gas']['h_gs']) 
        self.sigma = 5.67037e-8  # Stefan-Boltzmann
        self.eps_g = safe_f(config['gas'].get('emissivity_g', 0.3))
        self.eps_s = safe_f(config['material'].get('emissivity_s', 0.85))

    def heat_exchange(self, Tg, Ts, area, rpm=1.2):
        area = float(area)
        Tg = float(Tg)
        Ts = float(Ts)
        
        # 1. RPM DUYARLILIĞI İÇİN KONVEKSİYON REVİZYONU
        # RPM arttıkça yüzey tazelenmesi artar ancak bu durum 
        # kalış süresindeki kaybı telafi etmemeli. 
        # RPM etkisini karekök seviyesinde tutarak sönümleme etkisini azaltıyoruz.
        h_eff = self.base_h_gs * np.sqrt(rpm / 1.2) 
        q_conv = h_eff * area * (Tg - Ts)
        
        # 2. RADYASYON (T^4) REVİZYONU
        # Mevcut çarpım (eps_g * eps_s) radyasyonu çok baskılıyor.
        # Gaz-Katı arasındaki radyasyon için gri gaz varsayımı:
        # q = sigma * area * (eps_g * Tg^4 - alpha_g_s * Ts^4)
        # Basitleştirilmiş ama daha hassas bir yaklaşım:
        eps_eff = (self.eps_g + self.eps_s) / 2.0  # Ortalama emisivite ile enerji akışını açıyoruz
        
        q_rad = eps_eff * self.sigma * area * (np.power(Tg, 4) - np.power(Ts, 4))
        
        return q_conv + q_rad