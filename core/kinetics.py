import numpy as np
from numba import njit

@njit
def compute_clinker_kinetics_numba(Ts, X, m_CaO, m_SiO2, m_C2S, 
                                  k0_vec, Ea_vec, R, T_min_vec):
    """
    Vektörize Çimento Faz Kinetiği:
    0: Kalsinasyon (CaCO3 -> CaO)
    1: Belit Oluşumu (2CaO + SiO2 -> C2S)
    2: Alit Oluşumu (C2S + CaO -> C3S)
    """
    N = len(Ts)
    rates = np.zeros((3, N))
    
    # Sıcaklık bazlı maskeleme (T > T_min)
    mask_calc = (Ts >= T_min_vec[0]) & (X < 1.0)
    mask_c2s  = (Ts >= T_min_vec[1]) & (m_CaO > 0.01) & (m_SiO2 > 0.005)
    mask_c3s  = (Ts >= T_min_vec[2]) & (m_C2S > 0.01) & (m_CaO > 0.005)

    # 1. Kalsinasyon Hızı (r0)
    k0 = k0_vec[0] * np.exp(-Ea_vec[0] / (R * Ts))
    rates[0, mask_calc] = np.minimum(k0[mask_calc] * (1.0 - X[mask_calc]), 0.01)

    # 2. Belit (C2S) Oluşum Hızı (r1)
    # Katı faz diffüzyon limitli (Jander veya Ginstling-Brounshtein benzeri basitleştirme)
    k1 = k0_vec[1] * np.exp(-Ea_vec[1] / (R * Ts))
    rates[1, mask_c2s] = k1[mask_c2s] * (m_CaO[mask_c2s]**2) * m_SiO2[mask_c2s]

    # 3. Alit (C3S) Oluşum Hızı (r2)
    # Not: C3S oluşumu sıvı faz varlığında (genellikle >1250C) hızlanır
    k2 = k0_vec[2] * np.exp(-Ea_vec[2] / (R * Ts))
    rates[2, mask_c3s] = k2[mask_c3s] * m_C2S[mask_c3s] * m_CaO[mask_c3s]

    return rates

class CalcinationKinetics:
    def __init__(self, config):
        # Parametreleri vektör formunda tutuyoruz (Hız için)
        # Sırasıyla: [Kalsinasyon, C2S, C3S]
        self.k0 = np.array([
            float(config['kinetics']['k0']), 
            float(config['kinetics'].get('k0_c2s', 1.5e6)),
            float(config['kinetics'].get('k0_c3s', 2.0e7))
        ])
        
        self.Ea = np.array([
            float(config['kinetics']['Ea']),
            float(config['kinetics'].get('Ea_c2s', 1.8e5)), # J/mol
            float(config['kinetics'].get('Ea_c3s', 2.2e5))  # J/mol
        ])
        
        self.T_min = np.array([
            float(config['kinetics']['T_min_rxn']),
            float(config['kinetics'].get('T_min_c2s', 1100.0)),
            float(config['kinetics'].get('T_min_c3s', 1550.0)) # Sinterleşme bölgesi
        ])

        self.R = float(config['kinetics']['R'])
        
        # Isıl Etkiler (J/kg)
        self.dH_calc = float(config['kinetics']['dH'])     # Endotermik (+)
        self.dH_c2s = float(config['kinetics'].get('dH_c2s', -5.0e5)) # Ekzotermik (-)
        self.dH_c3s = float(config['kinetics'].get('dH_c3s', -1.2e5)) # Ekzotermik (-)

    def compute_all_rates(self, state):
        """Çözücüden çağrılan ana metod."""
        return compute_clinker_kinetics_numba(
            state.Ts, state.X, state.m_CaO, state.m_SiO2, state.m_C2S,
            self.k0, self.Ea, self.R, self.T_min
        )