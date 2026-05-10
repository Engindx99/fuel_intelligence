import numpy as np
from numba import njit

@njit
def compute_clinker_kinetics_numba(Ts, X, m_CaO, m_SiO2, m_C2S, m_C3A, m_C4AF,
                                  k0_vec, Ea_vec, R, T_min_vec):
    """
    Vektörize Çimento Faz Kinetiği (Solver ile Uyumlu, Darboğazsız Sürüm)
    """
    N = len(Ts)
    rates = np.zeros((3, N))
    
    # 1. KALSİNASYON (CO2 Çıkışı)
    # Kireç oluşum hızı kararlı bir plato için 0.10 ile sınırlandırıldı.
    mask_calc = (Ts >= T_min_vec[0]) & (X < 1.0)
    k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R * Ts))
    rates[0, mask_calc] = np.minimum(k_calc[mask_calc] * (1.0 - X[mask_calc]), 0.10)

    # 2. BELİT (C2S) OLUŞUMU
    # Kalsinasyon %75'e ulaştığında Belit oluşumu ivmelenir.
    # Sigmoid geçişi ile termal süreksizlik önlenir.
    sigmoid_c2s = 1.0 / (1.0 + np.exp(-0.03 * (Ts - T_min_vec[1])))
    k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R * Ts))
    
    # Solver'daki kütle dengesini beslemek için hız dampingi 0.02'ye çıkarıldı.
    raw_rate_c2s = k_c2s * m_CaO * m_SiO2 * sigmoid_c2s
    lock_c2s = 1.0 / (1.0 + np.exp(-15.0 * (X - 0.75))) # %75 kalsinasyon kilidi
    rates[1, :] = np.minimum(raw_rate_c2s * lock_c2s, 0.02)

    # 3. ALİT (C3S) OLUŞUMU
    # Kalsinasyon %92'den itibaren Alit yolu açılır, serbest kireç harcanmaya başlar.
    sigmoid_c3s = 1.0 / (1.0 + np.exp(-0.015 * (Ts - T_min_vec[2])))
    flux_phase = m_C3A + m_C4AF + 0.05 # Sıvı faz katalizör etkisi
    
    k_c3s = k0_vec[2] * np.exp(-Ea_vec[2] / (R * Ts))
    raw_rate_c3s = k_c3s * m_C2S * m_CaO * flux_phase * sigmoid_c3s
    
    # Sert %98 kilidi yerine %92 tabanlı yumuşak kilit (Free Lime tüketimi için kritik)
    lock_c3s = 1.0 / (1.0 + np.exp(-20.0 * (X - 0.92)))
    # Alit hızı kireci bitirebilmek için 0.015 tavanına yükseltildi.
    rates[2, :] = np.minimum(raw_rate_c3s * lock_c3s, 0.015) 

    # Stoikiometrik Kesici: Malzeme bittiğinde hızı fiziksel olarak sıfırla
    # Mikro eşik (1e-5) sayısal salınımı önler.
    rates[1, (m_CaO < 1e-5) | (m_SiO2 < 1e-5)] = 0.0
    rates[2, (m_CaO < 1e-5) | (m_C2S < 1e-5)] = 0.0

    return rates

class CalcinationKinetics:
    def __init__(self, config):
        # Kinetik parametreleri config'den al, yoksa fiziksel güvenli varsayılanları kullan
        self.k0 = np.array([
            float(config['kinetics'].get('k0', 1.0e7)), 
            float(config['kinetics'].get('k0_c2s', 1.5e6)), # Belit frekansı artırıldı
            float(config['kinetics'].get('k0_c3s', 1.2e6))  # Alit frekansı artırıldı
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
        """
        State içindeki dizileri Numba çekirdeğine göndererek reaksiyon hızlarını hesaplar.
        """
        return compute_clinker_kinetics_numba(
            state.Ts, state.X, state.m_CaO, state.m_SiO2, state.m_C2S, 
            state.m_C3A, state.m_C4AF,
            self.k0, self.Ea, self.R, self.T_min
        )