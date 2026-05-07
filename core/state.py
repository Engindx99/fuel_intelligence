import numpy as np

class KilnState:
    def __init__(self, n_nodes):
        self.N = n_nodes
        
        # --- Sıcaklık Profilleri ---
        self.Tg = np.zeros(n_nodes, dtype=float)
        self.Ts = np.zeros(n_nodes, dtype=float)
        self.Tw = np.zeros(n_nodes, dtype=float)
        
        # --- Kimyasal Kompozisyon ve Reaksiyon Derecesi ---
        self.X = np.zeros(n_nodes, dtype=float)  # Kalsinasyon derecesi (0.0 - 1.0)
        self.m_CaO = np.zeros(n_nodes, dtype=float)
        
        # Oksitler
        self.m_SiO2        = np.zeros(n_nodes, dtype=float)
        self.m_SiO2_locked = np.zeros(n_nodes, dtype=float)  # Kilitli silis takibi için eklendi
        self.m_Al2O3       = np.zeros(n_nodes, dtype=float)
        self.m_Fe2O3       = np.zeros(n_nodes, dtype=float)
        
        # Klinker Fazları
        self.m_C2S  = np.zeros(n_nodes, dtype=float) 
        self.m_C3S  = np.zeros(n_nodes, dtype=float) 
        self.m_C3A  = np.zeros(n_nodes, dtype=float) 
        self.m_C4AF = np.zeros(n_nodes, dtype=float) 
        
        # --- Fiziksel Parametreler ---
        self.rho_g = np.zeros(n_nodes, dtype=float)
        self.v_s = np.zeros(n_nodes, dtype=float)
        self.total_mass = np.ones(n_nodes, dtype=float) 

    def initialize_profiles(self, T_ambient=300.0, T_gas_inlet=2200.0, raw_meal_comp=None):
        # 1. Sıcaklıklar
        self.Ts.fill(float(T_ambient))
        self.Tw.fill(float(T_ambient))
        self.Tg = np.linspace(float(T_ambient), float(T_gas_inlet), self.N)
        
        # 2. Reaksiyon ve Kütle Başlangıcı
        self.X.fill(0.0)
        self.total_mass.fill(1.0) 
        
        if raw_meal_comp is None:
            raw_meal_comp = {'CaCO3': 0.77, 'SiO2': 0.16, 'Al2O3': 0.045, 'Fe2O3': 0.025}

        # 3. Bileşenlerin İlklendirilmesi
        # Silis'i ikiye bölerek başlatıyoruz: %75 Reaktif, %25 Kilitli
        sio2_total = float(raw_meal_comp['SiO2'])
        self.m_SiO2.fill(sio2_total * 0.75)
        self.m_SiO2_locked.fill(sio2_total * 0.25)
        
        self.m_Al2O3.fill(float(raw_meal_comp['Al2O3']))
        self.m_Fe2O3.fill(float(raw_meal_comp['Fe2O3']))
        
        # Kireç ve Fazların Sıfırlanması
        self.m_CaO.fill(1e-5)
        self.m_C2S.fill(0.0)   
        self.m_C3S.fill(0.0)
        self.m_C3A.fill(0.0)
        self.m_C4AF.fill(0.0)

        self.update_gas_density()

    def update_gas_density(self, mw_gas=28.97e-3, pressure=101325.0):
        R_universal = 8.314
        safe_Tg = np.maximum(200.0, self.Tg)
        self.rho_g = (float(pressure) * float(mw_gas)) / (R_universal * safe_Tg)