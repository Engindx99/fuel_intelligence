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
        self.m_SiO2  = np.zeros(n_nodes, dtype=float)
        self.m_Al2O3 = np.zeros(n_nodes, dtype=float)
        self.m_Fe2O3 = np.zeros(n_nodes, dtype=float)
        
        # Klinker Fazları
        self.m_C2S  = np.zeros(n_nodes, dtype=float) 
        self.m_C3S  = np.zeros(n_nodes, dtype=float) 
        self.m_C3A  = np.zeros(n_nodes, dtype=float) 
        self.m_C4AF = np.zeros(n_nodes, dtype=float) 
        
        # --- Fiziksel Parametreler ---
        self.rho_g = np.zeros(n_nodes, dtype=float)
        self.v_s = np.zeros(n_nodes, dtype=float)
        self.total_mass = np.ones(n_nodes, dtype=float) # Solver kütle kaybı için kullanır

    def initialize_profiles(self, T_ambient=300.0, T_gas_inlet=1400.0, raw_meal_comp=None):
        self.Ts.fill(float(T_ambient))
        self.Tw.fill(float(T_ambient))
        self.Tg = np.linspace(float(T_ambient), float(T_gas_inlet), self.N)
        
        self.X.fill(0.0)
        self.total_mass.fill(1.0)
        
        if raw_meal_comp is None:
            raw_meal_comp = {'CaCO3': 0.82, 'SiO2': 0.13, 'Al2O3': 0.03, 'Fe2O3': 0.02}

        self.m_SiO2.fill(raw_meal_comp['SiO2'])
        self.m_Al2O3.fill(raw_meal_comp['Al2O3'])
        self.m_Fe2O3.fill(raw_meal_comp['Fe2O3'])
        self.m_CaO.fill(0.0)
        
        for phase in ['m_C2S', 'm_C3S', 'm_C3A', 'm_C4AF']:
            getattr(self, phase).fill(0.0)

        self.update_gas_density()

    def update_gas_density(self, mw_gas=28.97e-3, pressure=101325.0):
        R_universal = 8.314
        safe_Tg = np.maximum(100.0, self.Tg)
        self.rho_g = (float(pressure) * float(mw_gas)) / (R_universal * safe_Tg)