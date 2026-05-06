import numpy as np
from numba import njit

@njit
def compute_clinker_kinetics_numba(Ts, X, m_CaO, m_SiO2, m_C2S, m_C3A, m_C4AF,
                                  k0_vec, Ea_vec, R, T_min_vec):
    """
    Vektörize Çimento Faz Kinetiği (Fiziksel Rampalı Sürüm)
    """
    N = len(Ts)
    rates = np.zeros((3, N))
    
    # 1. KALSİNASYON (Stabilizasyon için maske devam ediyor)
    mask_calc = (Ts >= T_min_vec[0]) & (X < 1.0)
    k0 = k0_vec[0] * np.exp(-Ea_vec[0] / (R * Ts))
    rates[0, mask_calc] = np.minimum(k0[mask_calc] * (1.0 - X[mask_calc]), 0.01)

    # 2. BELİT (C2S) - Yumuşak Geçiş (Beta=0.03)
    # Reaksiyon 1100K civarında yavaşça başlar
    sigmoid_c2s = 1.0 / (1.0 + np.exp(-0.03 * (Ts - T_min_vec[1])))
    k1 = k0_vec[1] * np.exp(-Ea_vec[1] / (R * Ts))
    rates[1, :] = k1 * (m_CaO**2) * m_SiO2 * sigmoid_c2s

    # 3. ALİT (C3S) - "Dan" Diye Yükselmeyi Engelleyen Strateji
    # Strateji A: Sigmoid sertliğini (beta) 0.1'den 0.015'e düşürerek 150-200K'e yaydık.
    # Strateji B: Hızı doğrudan sıvı faz kütlesine (flux) çarpan olarak bağladık.
    # Strateji C: Sayısal damping (np.minimum) ekledik.
    
    # Sıcaklık rampası (Çok daha geniş bir yayılım)
    sigmoid_c3s = 1.0 / (1.0 + np.exp(-0.015 * (Ts - T_min_vec[2])))
    
    # Sıvı faz (melt) etkisi - C3A ve C4AF arttıkça Alit hızı doğal bir ivme kazanır
    flux_phase = m_C3A + m_C4AF
    
    k2 = k0_vec[2] * np.exp(-Ea_vec[2] / (R * Ts))
    
    # Ham hız hesabı
    raw_rate_c3s = k2 * m_C2S * m_CaO * flux_phase * sigmoid_c3s
    
    # [DAMPING] Sayısal Damping: Bir adımda oluşabilecek maksimum Alit miktarını 
    # sınırlayarak grafiksel "duvar" oluşumunu fiziksel bir rampaya dönüştürür.
    rates[2, :] = np.minimum(raw_rate_c3s, 0.0003) 

    # Stoikiometrik Kesici: Malzeme bittiğinde hızı sıfırla
    rates[1, m_CaO < 1e-4] = 0
    rates[2, (m_CaO < 1e-4) | (m_C2S < 1e-4)] = 0

    return rates

class CalcinationKinetics:
    def __init__(self, config):
        # Kinetik sabitlerdeki aşırı agresifliği frenledik
        self.k0 = np.array([
            float(config['kinetics']['k0']), 
            float(config['kinetics'].get('k0_c2s', 1.2e6)),
            float(config['kinetics'].get('k0_c3s', 8.0e5))  # Düşürüldü
        ])
        
        self.Ea = np.array([
            float(config['kinetics']['Ea']),
            float(config['kinetics'].get('Ea_c2s', 1.8e5)),
            float(config['kinetics'].get('Ea_c3s', 2.6e5))  # Artırıldı (Daha ağır tepki)
        ])
        
        self.T_min = np.array([
            float(config['kinetics']['T_min_rxn']),
            float(config['kinetics'].get('T_min_c2s', 1100.0)),
            float(config['kinetics'].get('T_min_c3s', 1500.0))  # Biraz erkene çekildi
        ])

        self.R = float(config['kinetics']['R'])
        self.dH_calc = float(config['kinetics']['dH'])
        self.dH_c2s = float(config['kinetics'].get('dH_c2s', -7.5e5))
        self.dH_c3s = float(config['kinetics'].get('dH_c3s', 4.0e4))

    def compute_all_rates(self, state):
        return compute_clinker_kinetics_numba(
            state.Ts, state.X, state.m_CaO, state.m_SiO2, state.m_C2S, 
            state.m_C3A, state.m_C4AF,
            self.k0, self.Ea, self.R, self.T_min
        )