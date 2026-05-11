import numpy as np
from numba import njit

@njit
def compute_clinker_kinetics_numba(Ts, X, m_CaO, m_SiO2, m_C2S, m_Al2O3, m_Fe2O3, m_C3A, m_C4AF,
                                  k0_vec, Ea_vec, R, T_min_vec, 
                                  pre_factors, kiln_length, dz, burning_zone_factor):
    """
    Saf fiziksel kinetik model. Uydurma kilitler kaldırıldı.
    Reaksiyonlar sadece termodinamik eşikler ve kütle konsantrasyonları ile yürür.
    """
    N = len(Ts)
    rates = np.zeros((5, N)) 
    
    for i in range(N):
        T_curr = max(300.0, Ts[i])
        
        # 1. KALSİNASYON (Endotermik)
        # Sadece CaCO3 mevcudiyetine ve sıcaklığa bağlıdır.
        if T_curr >= T_min_vec[0] and X[i] < 1.0:
            k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R * T_curr))
            rates[0, i] = k_calc * (1.0 - X[i])

        # 2. BELİT (C2S) OLUŞUMU (Ekzotermik)
        # CaO ve SiO2 arasındaki katı faz reaksiyonu.
        if T_curr >= T_min_vec[1]:
            if m_CaO[i] > 1e-5 and m_SiO2[i] > 1e-5:
                k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R * T_curr))
                # İkinci mertebe reaksiyon kinetiği
                rates[1, i] = k_c2s * m_CaO[i] * m_SiO2[i]

        # 3. ALİT (C3S) OLUŞUMU (Ekzotermik)
        # Mekanizma: C2S + CaO -> C3S (Sıvı faz varlığında)
        if T_curr >= T_min_vec[2]:
            if m_CaO[i] > 1e-5 and m_C2S[i] > 1e-5:
                # Sıvı faz (flux) miktarı reaksiyon hızını doğrudan belirler
                liquid_phase = m_C3A[i] + m_C4AF[i]
                
                # Fiziksel gerçeklik: Sıvı faz yoksa reaksiyon hızı ihmal edilebilir düzeydedir.
                if liquid_phase > 0.01:
                    k_c3s = k0_vec[2] * np.exp(-Ea_vec[2] / (R * T_curr))
                    # Alit oluşumu, ortamdaki sıvı fazın (melt) taşıyıcılığı ile orantılıdır.
                    rates[2, i] = k_c3s * m_CaO[i] * m_C2S[i] * liquid_phase * burning_zone_factor

        # 4. YARDIMCI FAZLAR (C3A ve C4AF)
        # Alüminat ve Ferrit fazları sıvı fazın ana bileşenleridir.
        if T_curr >= 1200.0:
            if m_Al2O3[i] > 1e-5:
                rates[3, i] = pre_factors[0] * m_Al2O3[i]
            if m_Fe2O3[i] > 1e-5:
                rates[4, i] = pre_factors[1] * m_Fe2O3[i]

    return rates

class CalcinationKinetics:
    def __init__(self, config):
        self.cfg = config
        k_cfg = config.get('kinetics', {})
        
        # Fiziksel sabitler: k0 ve Ea değerleri literatürdeki (Taylor, 1997) 
        # klinkerleşme enerjilerine yaklaştırıldı.
        self.k0 = np.array([
            float(k_cfg.get('k0', 1.5e6)),     
            float(k_cfg.get('k0_c2s', 2.0e6)), 
            float(k_cfg.get('k0_c3s', 3.5e6))  
        ])
        
        self.Ea = np.array([
            float(k_cfg.get('Ea', 1.9e5)),     
            float(k_cfg.get('Ea_c2s', 2.1e5)), 
            float(k_cfg.get('Ea_c3s', 2.4e5)) # Alit bariyeri fiziksel olarak en yükseğidir.
        ])
        
        self.T_min = np.array([
            float(k_cfg.get('T_min_rxn', 1073.0)), 
            float(k_cfg.get('T_min_c2s', 1150.0)),
            float(k_cfg.get('T_min_c3s', 1450.0)) # Alit için 1250C (1523K) daha gerçekçidir, 1450K alt sınır.
        ])

        self.pre_factors = np.array([
            float(k_cfg.get('pre_factor_c3a', 5.0e-4)),
            float(k_cfg.get('pre_factor_c4af', 4.0e-4))
        ])

        self.R = 8.314
        self.dH_calc = float(k_cfg.get('dH', 1.78e6))
        self.dH_c2s = float(k_cfg.get('dH_c2s', -1.10e6))
        self.dH_c3s = float(k_cfg.get('dH_c3s', -6.0e5))
        
        self.kiln_length = float(config['kiln']['length'])
        self.dz = self.kiln_length / int(config['kiln']['nodes'])
        self.burning_zone_factor = 1.0 # Fiziksel modelde yapay çarpan 1.0 olmalı.

    def compute_all_rates(self, state):
        return compute_clinker_kinetics_numba(
            state.Ts, state.X, state.m_CaO, state.m_SiO2, state.m_C2S, 
            state.m_Al2O3, state.m_Fe2O3, state.m_C3A, state.m_C4AF,
            self.k0, self.Ea, self.R, self.T_min, self.pre_factors,
            self.kiln_length, self.dz, self.burning_zone_factor
        )

    def get_enthalpy_vector(self):
        return np.array([self.dH_calc, self.dH_c2s, self.dH_c3s, -1.0e5, -0.8e5])