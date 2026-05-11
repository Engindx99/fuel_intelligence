import numpy as np

class KilnState:
    def __init__(self, n_nodes):
        self.N = n_nodes
        
        # --- Sıcaklık Profilleri ---
        self.Tg = np.zeros(n_nodes, dtype=float) 
        self.Ts = np.zeros(n_nodes, dtype=float) 
        self.Tw = np.zeros(n_nodes, dtype=float) 
        
        # --- Kimyasal Kompozisyon ---
        self.X = np.zeros(n_nodes, dtype=float)          
        self.m_CaO = np.zeros(n_nodes, dtype=float)      
        self.m_SiO2 = np.zeros(n_nodes, dtype=float)     
        # m_SiO2_locked kaldırıldı veya 0 yapıldı (Saturasyonu aşmak için)
        self.m_Al2O3 = np.zeros(n_nodes, dtype=float)    
        self.m_Fe2O3 = np.zeros(n_nodes, dtype=float)    
        
        # --- Klinker Fazları ---
        self.m_C2S  = np.zeros(n_nodes, dtype=float)     
        self.m_C3S  = np.zeros(n_nodes, dtype=float)     
        self.m_C3A  = np.zeros(n_nodes, dtype=float)     
        self.m_C4AF = np.zeros(n_nodes, dtype=float)     
        
        # --- Kütle Takibi ---
        self.m_co2_released = np.zeros(n_nodes, dtype=float) 
        self.total_solid_mass = np.ones(n_nodes, dtype=float) 

        # --- Fiziksel Parametreler ---
        self.rho_g = np.zeros(n_nodes, dtype=float)      
        self.v_s = np.zeros(n_nodes, dtype=float)        
        self.zones = np.zeros(n_nodes, dtype=int)        

    @property
    def total_mass(self):
        return self.total_solid_mass

    @total_mass.setter
    def total_mass(self, value):
        self.total_solid_mass = value

    def initialize_profiles(self, T_ambient=300.0, T_gas_inlet=2400.0, raw_meal_comp=None):
        """
        Saturasyonu kırmak için optimize edilmiş başlangıç koşulları.
        """
        self.Ts.fill(float(T_ambient))
        self.Tw.fill(float(T_ambient + 100.0))
        
        # Gaz profili: Daha agresif bir üstel veya logaritmik başlangıç 
        # (Saturasyonu aşmak için soğuk ucu 900K bandına yaklaştırıyoruz)
        self.Tg = np.linspace(900.0, float(T_gas_inlet), self.N)
        
        self.X.fill(0.0)
        self.m_co2_released.fill(0.0)
        
        if raw_meal_comp is None:
            raw_meal_comp = {
                'CaCO3': 0.760, # Standart klinker hedefi için %76
                'SiO2': 0.140, 
                'Al2O3': 0.050, 
                'Fe2O3': 0.030
            }

        # Tüm silisi reaktif kabul ediyoruz (Saturasyonu aşmak için kritik)
        self.m_SiO2.fill(float(raw_meal_comp['SiO2']))
        self.m_Al2O3.fill(float(raw_meal_comp['Al2O3']))
        self.m_Fe2O3.fill(float(raw_meal_comp['Fe2O3']))
        
        self.m_CaO.fill(1e-9)
        self.m_C2S.fill(0.0)   
        self.m_C3S.fill(0.0)
        self.m_C4AF.fill(0.0)
        self.m_C3A.fill(0.0)
        
        self.total_solid_mass.fill(1.0) 

        self.update_gas_density()
        
        # Bölge tanımları
        for i in range(self.N):
            x_ratio = i / (self.N - 1)
            if x_ratio < 0.30: self.zones[i] = 0 # Preheat biraz kısaltıldı
            elif x_ratio < 0.65: self.zones[i] = 1 # Calcination uzatıldı
            elif x_ratio < 0.85: self.zones[i] = 2 
            else: self.zones[i] = 3 

    def update_gas_density(self, mw_gas=29e-3, pressure=101325.0):
        R_universal = 8.314
        safe_Tg = np.maximum(300.0, self.Tg)
        self.rho_g = (pressure * mw_gas) / (R_universal * safe_Tg)

    def get_clinker_quality(self):
        idx = -1 
        # Bölü sıfır hatası korumalı kalite metrikleri
        c2s = max(1e-9, self.m_C2S[idx])
        return {
            "C3S": self.m_C3S[idx],
            "C2S": c2s,
            "fCaO": self.m_CaO[idx],
            "LSF": (self.m_CaO[idx]) / (2.8 * self.m_SiO2[idx] + 1.1 * self.m_Al2O3[idx] + 0.7 * self.m_Fe2O3[idx] + 1e-9)
        }