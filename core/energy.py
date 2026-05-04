import numpy as np

def safe_f(value):
    if isinstance(value, list): return float(value[0])
    return float(value)

class EnergyModel:
    def __init__(self, config):
        self.h_gs = safe_f(config['gas']['h_gs'])
        self.sigma = 5.67037e-8  # Stefan-Boltzmann
        self.eps_g = safe_f(config['gas'].get('emissivity_g', 0.3))
        self.eps_s = safe_f(config['material'].get('emissivity_s', 0.85))

    def heat_exchange(self, Tg, Ts, area):
        area = float(area)
        Tg = float(Tg)
        Ts = float(Ts)
        
        # Konveksiyon
        q_conv = self.h_gs * area * (Tg - Ts)
        
        # Radyasyon (T^4 farkı)
        # Efektif emite: 1 / (1/eps_g + 1/eps_s - 1) basitleştirmesi
        eps_eff = self.eps_g * self.eps_s
        q_rad = eps_eff * self.sigma * area * (np.power(Tg, 4) - np.power(Ts, 4))
        
        return q_conv + q_rad