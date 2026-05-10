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
        self.m_SiO2_locked = np.zeros(n_nodes, dtype=float) 
        self.m_Al2O3 = np.zeros(n_nodes, dtype=float)
        self.m_Fe2O3 = np.zeros(n_nodes, dtype=float)
        
        # --- Klinker Fazları ---
        self.m_C2S  = np.zeros(n_nodes, dtype=float) 
        self.m_C3S  = np.zeros(n_nodes, dtype=float) 
        self.m_C3A  = np.zeros(n_nodes, dtype=float) 
        self.m_C4AF = np.zeros(n_nodes, dtype=float) 
        
        # --- Kütle ve Gaz Takibi ---
        self.m_co2_released = np.zeros(n_nodes, dtype=float) 
        self.total_solid_mass = np.ones(n_nodes, dtype=float) 

        # --- Fiziksel Parametreler ---
        self.rho_g = np.zeros(n_nodes, dtype=float)
        self.v_s = np.zeros(n_nodes, dtype=float)

    # --- GERİYE DÖNÜK UYUMLULUK (BACKWARD COMPATIBILITY) ---
    @property
    def total_mass(self):
        """run_simulation.py içindeki eski çağrıları karşılamak için."""
        return self.total_solid_mass

    @total_mass.setter
    def total_mass(self, value):
        self.total_solid_mass = value

    def initialize_profiles(self, T_ambient=300.0, T_gas_inlet=2200.0, raw_meal_comp=None):
        self.Ts.fill(float(T_ambient))
        self.Tw.fill(float(T_ambient))
        self.Tg = np.linspace(float(T_ambient + 500.0), float(T_gas_inlet), self.N)
        
        self.X.fill(0.0)
        self.m_co2_released.fill(0.0)
        
        if raw_meal_comp is None:
            raw_meal_comp = {'CaCO3': 0.730, 'SiO2': 0.210, 'Al2O3': 0.040, 'Fe2O3': 0.020}

        sio2_total = float(raw_meal_comp['SiO2'])
        self.m_SiO2.fill(sio2_total * 0.50)
        self.m_SiO2_locked.fill(sio2_total * 0.50)
        
        self.m_Al2O3.fill(float(raw_meal_comp['Al2O3']))
        self.m_Fe2O3.fill(float(raw_meal_comp['Fe2O3']))
        
        self.m_CaO.fill(1e-6)
        self.m_C2S.fill(0.0)   
        self.m_C3S.fill(0.0)
        self.m_C4AF.fill(0.0)
        self.m_C3A.fill(0.0)
        
        self.total_solid_mass.fill(1.0) 

        self.update_gas_density()

    def update_gas_density(self, mw_gas=28.97e-3, pressure=101325.0):
        R_universal = 8.314
        safe_Tg = np.maximum(273.15, self.Tg)
        self.rho_g = (float(pressure) * float(mw_gas)) / (R_universal * safe_Tg)