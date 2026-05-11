import numpy as np
from numba import njit

@njit
def compute_clinker_kinetics_numba(Ts, X, m_CaO, m_SiO2, m_C2S, m_m_Al2O3, m_Fe2O3, m_C3A, m_C4AF,
                                  k0_vec, Ea_vec, R, T_min_vec, 
                                  pre_factors, kiln_length, dz, burning_zone_factor):
    """
    YAML hiyerarşisine ve Bölgesel (Zoned) modele tam uyumlu Numba çekirdeği.
    Termal satürasyonu kırmak için aktivasyon bariyerleri optimize edildi.
    """
    N = len(Ts)
    rates = np.zeros((5, N)) # 0:Calc, 1:C2S, 2:C3S, 3:C3A, 4:C4AF
    
    for i in range(N):
        x_ratio = (i * dz) / kiln_length
        # Sıcaklık değerini stabilite için alt sınırda tut
        T_curr = max(300.0, Ts[i])
        
        # 1. KALSİNASYON (CaCO3 -> CaO)
        # Bölge: Calcination (0.35 - 0.60) ve sonrası
        # KRİTİK: T_min_vec[0] (1073K) altında hız çok düşük de olsa başlamalı 
        # ki sistem kendi ısısını üretmeye meyillensin (Warm-up desteği).
        if x_ratio >= 0.35:
            # Eşik sıcaklığının %10 altında bile çok küçük bir kalsinasyon başlatarak 
            # sayısal "donmayı" engelliyoruz.
            if T_curr >= (T_min_vec[0] * 0.9) and X[i] < 1.0:
                arg = -Ea_vec[0] / (R * T_curr)
                # Arrhenius terimini clipleyerek patlamayı önle
                exp_term = np.exp(max(-80.0, min(20.0, arg)))
                k_calc = k0_vec[0] * exp_term
                
                # Soft-lock kalsinasyonu düzgün başlatmak için (1-X) terimi lineerdir.
                rates[0, i] = np.minimum(k_calc * (1.0 - X[i]), 0.025) # Hız sınırı hafif artırıldı

        # 2. BELİT (C2S) OLUŞUMU
        # Bölge: Transition (0.60 - 0.80) ve Burning (0.80 - 1.0)
        if x_ratio >= 0.60:
            if T_curr >= T_min_vec[1] and m_CaO[i] > 1e-6 and m_SiO2[i] > 1e-6:
                # %75 Kalsinasyon şartı (Soft-lock) - Fiziksel tutarlılık için
                # Sigmoid fonksiyonu ile geçişi yumuşatıyoruz.
                lock_c2s = 1.0 / (1.0 + np.exp(-10.0 * (X[i] - 0.70))) # Eşik %70'e çekildi
                
                arg = -Ea_vec[1] / (R * T_curr)
                exp_term = np.exp(max(-80.0, min(20.0, arg)))
                k_c2s = k0_vec[1] * exp_term
                
                rates[1, i] = np.minimum(k_c2s * m_CaO[i] * m_SiO2[i] * lock_c2s, 0.040)

        # 3. ALİT (C3S) OLUŞUMU
        # Bölge: Sadece Burning Zone (0.80 - 1.0)
        if x_ratio >= 0.80:
            if T_curr >= T_min_vec[2] and m_CaO[i] > 1e-6 and m_C2S[i] > 0.05:
                # %92 Kalsinasyon şartı (Soft-lock)
                lock_c3s = 1.0 / (1.0 + np.exp(-15.0 * (X[i] - 0.90))) # Eşik %90'a çekildi
                
                # Sıvı fazın (C3A+C4AF) katalizör etkisi
                # 0.05 taban değeri reaksiyonun "kuru" da olsa çok yavaş başlamasını sağlar
                liquid_effect = (m_C3A[i] + m_C4AF[i] + 0.05) * 10.0 # Çarpan 8'den 10'a çıkarıldı
                
                arg = -Ea_vec[2] / (R * T_curr)
                exp_term = np.exp(max(-80.0, min(20.0, arg)))
                k_c3s = k0_vec[2] * exp_term
                
                # Burning zone factor amplifikasyonu
                rates[2, i] = k_c3s * m_C2S[i] * m_CaO[i] * liquid_effect * lock_c3s * burning_zone_factor

        # 4. YARDIMCI FAZLAR (Flux/Melt)
        if x_ratio >= 0.60:
            # 1350K eşiği altına düşülse bile klinkerleşme için sıvı faz hayati.
            # Geçiş bölgesinde (1200K+) yavaş oluşum başlatılıyor.
            if T_curr >= 1250.0 and m_m_Al2O3[i] > 1e-6:
                rates[3, i] = pre_factors[0] * m_m_Al2O3[i] # C3A
            if T_curr >= 1200.0 and m_Fe2O3[i] > 1e-6:
                rates[4, i] = pre_factors[1] * m_Fe2O3[i] # C4AF

    return rates

class CalcinationKinetics:
    def __init__(self, config):
        self.cfg = config
        k_cfg = config.get('kinetics', {})
        
        # YAML'daki isimlendirmelerle birebir eşleşme
        # Satürasyonu kırmak için k0 (Frekans Faktörü) değerleri optimize edildi.
        self.k0 = np.array([
            float(k_cfg.get('k0', 5.0e6)),         # Kalsinasyon (Hızlandırıldı)
            float(k_cfg.get('k0_c2s', 4.0e6)),     # Belit
            float(k_cfg.get('k0_c3s', 2.8e6))      # Alit
        ])
        
        self.Ea = np.array([
            float(k_cfg.get('Ea', 1.80e5)),        # Aktivasyon enerjisi hafif düşürüldü (J/mol)
            float(k_cfg.get('Ea_c2s', 1.90e5)),
            float(k_cfg.get('Ea_c3s', 2.85e5))
        ])
        
        self.T_min = np.array([
            float(k_cfg.get('T_min_rxn', 1050.0)), # Eşik sıcaklığı 1073'ten 1050'ye çekildi
            float(k_cfg.get('T_min_c2s', 1200.0)),
            float(k_cfg.get('T_min_c3s', 1350.0))
        ])

        self.pre_factors = np.array([
            float(k_cfg.get('pre_factor_c3a', 1.0e-3)),
            float(k_cfg.get('pre_factor_c4af', 8.0e-4))
        ])

        self.R = float(k_cfg.get('R', 8.314))
        
        # Isıl Etkiler (Ekzotermik/Endotermik Entalpiler)
        self.dH_calc = float(k_cfg.get('dH', 1.78e6))  # CaCO3 -> CaO (Isı Alan +)
        self.dH_c2s = float(k_cfg.get('dH_c2s', -1.10e6)) # Ekzotermik (-)
        self.dH_c3s = float(k_cfg.get('dH_c3s', -6.0e5))  # Ekzotermik (-)
        
        # Fırın Geometrisi
        self.kiln_length = float(config['kiln']['length'])
        nodes = int(config['kiln']['nodes'])
        self.dz = self.kiln_length / nodes
        
        self.burning_zone_factor = 3.5 # Yanma bölgesi etkisi güçlendirildi

    def compute_all_rates(self, state):
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
        return np.array([
            self.dH_calc, 
            self.dH_c2s, 
            self.dH_c3s, 
            -150000.0, # C3A ekzotermik katkısı artırıldı
            -120000.0  # C4AF ekzotermik katkısı artırıldı
        ])