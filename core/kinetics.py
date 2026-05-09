import numpy as np
from numba import njit

@njit
def compute_clinker_kinetics_numba(Ts, X, m_CaO, m_SiO2, m_C2S, m_C3A, m_C4AF,
                                  k0_vec, Ea_vec, R, T_min_vec):
    """
    Vektörize Çimento Faz Kinetiği (Fiziksel Rampalı ve Kilitli Sürüm)
    """
    N = len(Ts)
    rates = np.zeros((3, N))
    
    # 1. KALSİNASYON (CO2 Çıkışı)
    # Kalsinasyon hızı k0 damping ile (0.15) sınırlandı, plato kararlılığı sağlandı.
    mask_calc = (Ts >= T_min_vec[0]) & (X < 1.0)
    k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R * Ts))
    rates[0, mask_calc] = np.minimum(k_calc[mask_calc] * (1.0 - X[mask_calc]), 0.15)

    # 2. BELİT (C2S) OLUŞUMU
    # KALSİNASYON KİLİDİ: X > 0.85 olmadan C2S enerjiyi çalmasın.
    sigmoid_c2s = 1.0 / (1.0 + np.exp(-0.03 * (Ts - T_min_vec[1])))
    k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R * Ts))
    
    raw_rate_c2s = k_c2s * (m_CaO**2) * m_SiO2 * sigmoid_c2s
    # Kalsinasyon baskınken C2S hızını fiziksel olarak kısıtla
    lock_c2s = np.where(X > 0.85, 1.0, 0.0) 
    rates[1, :] = np.minimum(raw_rate_c2s * lock_c2s, 0.005)

    # 3. ALİT (C3S) OLUŞUMU
    # SIVI FAZ KİLİDİ: Kalsinasyon TAM bitmeden (X > 0.98) Alit başlamaz.
    sigmoid_c3s = 1.0 / (1.0 + np.exp(-0.015 * (Ts - T_min_vec[2])))
    flux_phase = m_C3A + m_C4AF + 0.01 # Sıvı faz yoksa reaksiyon durma noktasına gelir
    
    k_c3s = k0_vec[2] * np.exp(-Ea_vec[2] / (R * Ts))
    raw_rate_c3s = k_c3s * m_C2S * m_CaO * flux_phase * sigmoid_c3s
    
    # Tam kalsinasyon kilidi ve damping
    lock_c3s = np.where(X > 0.98, 1.0, 0.0)
    rates[2, :] = np.minimum(raw_rate_c3s * lock_c3s, 0.0005) 

    # Stoikiometrik Kesici: Malzeme bittiğinde hızı sıfırla
    rates[1, (m_CaO < 1e-4) | (m_SiO2 < 1e-4)] = 0
    rates[2, (m_CaO < 1e-4) | (m_C2S < 1e-4)] = 0

    return rates

class CalcinationKinetics:
    def __init__(self, config):
        # Kinetik parametreleri config'den al, yoksa güvenli varsayılanları kullan
        self.k0 = np.array([
            float(config['kinetics'].get('k0', 1.0e7)), 
            float(config['kinetics'].get('k0_c2s', 1.2e6)),
            float(config['kinetics'].get('k0_c3s', 8.0e5))
        ])
        
        self.Ea = np.array([
            float(config['kinetics'].get('Ea', 1.7e5)),
            float(config['kinetics'].get('Ea_c2s', 1.8e5)),
            float(config['kinetics'].get('Ea_c3s', 2.6e5))
        ])
        
        self.T_min = np.array([
            float(config['kinetics'].get('T_min_rxn', 800.0)),
            float(config['kinetics'].get('T_min_c2s', 1100.0)),
            float(config['kinetics'].get('T_min_c3s', 1450.0))
        ])

        self.R = 8.314
        self.dH_calc = float(config['kinetics'].get('dH', 1.78e6)) # J/kg
        self.dH_c2s = float(config['kinetics'].get('dH_c2s', -7.0e5)) # Ekzotermik (-)
        self.dH_c3s = float(config['kinetics'].get('dH_c3s', 5.0e4))  # Endotermik (+)

    def compute_all_rates(self, state):
        # State içindeki dizileri Numba çekirdeğine gönderir
        return compute_clinker_kinetics_numba(
            state.Ts, state.X, state.m_CaO, state.m_SiO2, state.m_C2S, 
            state.m_C3A, state.m_C4AF,
            self.k0, self.Ea, self.R, self.T_min
        )