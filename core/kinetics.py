import numpy as np
from numba import njit

@njit
def compute_clinker_kinetics_numba(Ts, X, m_CaO, m_SiO2, m_C2S, m_m_Al2O3, m_Fe2O3, m_C3A, m_C4AF,
                                  k0_vec, Ea_vec, R, T_min_vec, 
                                  pre_factors, kiln_length, dz, burning_zone_factor):
    """
    YAML hiyerarşisine ve Bölgesel (Zoned) modele tam uyumlu Numba çekirdeği.
    """
    N = len(Ts)
    rates = np.zeros((5, N)) # 0:Calc, 1:C2S, 2:C3S, 3:C3A, 4:C4AF
    
    for i in range(N):
        x_ratio = (i * dz) / kiln_length
        
        # 1. KALSİNASYON (CaCO3 -> CaO)
        # Bölge: Calcination (0.35 - 0.60) ve sonrası
        if x_ratio >= 0.35:
            if Ts[i] >= T_min_vec[0] and X[i] < 1.0:
                # Soft-lock kalsinasyonu düzgün başlatmak için
                k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R * Ts[i]))
                rates[0, i] = np.minimum(k_calc * (1.0 - X[i]), 0.015)

        # 2. BELİT (C2S) OLUŞUMU
        # Bölge: Transition (0.60 - 0.80) ve Burning (0.80 - 1.0)
        if x_ratio >= 0.60:
            if Ts[i] >= T_min_vec[1] and m_CaO[i] > 1e-6 and m_SiO2[i] > 1e-6:
                # %75 Kalsinasyon şartı (Soft-lock) - Fiziksel tutarlılık için
                lock_c2s = 1.0 / (1.0 + np.exp(-10.0 * (X[i] - 0.75)))
                k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R * Ts[i]))
                rates[1, i] = np.minimum(k_c2s * m_CaO[i] * m_SiO2[i] * lock_c2s, 0.030)

        # 3. ALİT (C3S) OLUŞUMU
        # Bölge: Sadece Burning Zone (0.80 - 1.0)
        if x_ratio >= 0.80:
            if Ts[i] >= T_min_vec[2] and m_CaO[i] > 1e-6 and m_C2S[i] > 0.05:
                # %92 Kalsinasyon şartı (Soft-lock)
                lock_c3s = 1.0 / (1.0 + np.exp(-15.0 * (X[i] - 0.92)))
                # Sıvı fazın (C3A+C4AF) katalizör etkisi
                liquid_effect = (m_C3A[i] + m_C4AF[i] + 0.05) * 8.0
                k_c3s = k0_vec[2] * np.exp(-Ea_vec[2] / (R * Ts[i]))
                
                # Burning zone residence behavior amplifikasyonu uygulanıyor
                rates[2, i] = k_c3s * m_C2S[i] * m_CaO[i] * liquid_effect * lock_c3s * burning_zone_factor

        # 4. YARDIMCI FAZLAR (Flux/Melt)
        # Sadece geçiş ve yanma bölgelerinde aktif
        if x_ratio >= 0.60:
            if Ts[i] >= 1350.0 and m_m_Al2O3[i] > 1e-6:
                rates[3, i] = pre_factors[0] * m_m_Al2O3[i] # C3A formation
            if Ts[i] >= 1300.0 and m_Fe2O3[i] > 1e-6:
                rates[4, i] = pre_factors[1] * m_Fe2O3[i] # C4AF formation

    return rates

class CalcinationKinetics:
    def __init__(self, config):
        self.cfg = config
        k_cfg = config.get('kinetics', {})
        
        # YAML'daki isimlendirmelerle birebir eşleşme
        self.k0 = np.array([
            float(k_cfg.get('k0', 3.0e5)),         # Kalsinasyon
            float(k_cfg.get('k0_c2s', 3.5e5)),     # Belit
            float(k_cfg.get('k0_c3s', 2.3e5))      # Alit
        ])
        
        self.Ea = np.array([
            float(k_cfg.get('Ea', 1.95e5)),
            float(k_cfg.get('Ea_c2s', 2.0e5)),
            float(k_cfg.get('Ea_c3s', 3.0e5))
        ])
        
        self.T_min = np.array([
            float(k_cfg.get('T_min_rxn', 1073.0)), # CaCO3 -> 1073K
            float(k_cfg.get('T_min_c2s', 1220.0)),
            float(k_cfg.get('T_min_c3s', 1380.0))
        ])

        self.pre_factors = np.array([
            float(k_cfg.get('pre_factor_c3a', 8.0e-4)),
            float(k_cfg.get('pre_factor_c4af', 5.0e-4))
        ])

        self.R = float(k_cfg.get('R', 8.314))
        
        # Isıl Etkiler (Ekzotermik/Endotermik Entalpiler)
        self.dH_calc = float(k_cfg.get('dH', 1.78e6))
        self.dH_c2s = float(k_cfg.get('dH_c2s', -1.10e6))
        self.dH_c3s = float(k_cfg.get('dH_c3s', -6.0e5))
        
        # Fırın Geometrisi (Bölge hesabı için gerekli)
        self.kiln_length = float(config['kiln']['length'])
        nodes = int(config['kiln']['nodes'])
        self.dz = self.kiln_length / nodes
        
        # Yanma bölgesi çarpanı (Residence time amplifikasyonu için)
        self.burning_zone_factor = 2.5 

    def compute_all_rates(self, state):
        """
        State nesnesinden gelen verilerle Numba çekirdeğini çalıştırır.
        """
        return compute_clinker_kinetics_numba(
            state.Ts, state.X, state.m_CaO, state.m_SiO2, state.m_C2S, 
            state.m_Al2O3, state.m_Fe2O3, state.m_C3A, state.m_C4AF,
            self.k0, self.Ea, self.R, self.T_min, self.pre_factors,
            self.kiln_length, self.dz, self.burning_zone_factor
        )

    def get_enthalpy_vector(self):
        """
        Solver içinde dTs hesaplamasında kullanılan reaksiyon ısıları vektörü.
        """
        # [Calc, C2S, C3S, C3A, C4AF]
        return np.array([
            self.dH_calc, 
            self.dH_c2s, 
            self.dH_c3s, 
            -125000.0, # C3A tahmini dH
            -100000.0  # C4AF tahmini dH
        ])