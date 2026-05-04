import numpy as np

class KilnState:
    def __init__(self, n_nodes):
        self.N = n_nodes
        
        # --- Sıcaklık Profilleri ---
        self.Tg = np.zeros(n_nodes, dtype=float)
        self.Ts = np.zeros(n_nodes, dtype=float)
        self.Tw = np.zeros(n_nodes, dtype=float)
        
        # --- Kimyasal Kompozisyon (Kütle Oranları) ---
        # Ana reaktantlar
        self.X = np.zeros(n_nodes, dtype=float)  # Kalsinasyon derecesi
        self.m_CaCO3 = np.zeros(n_nodes, dtype=float)
        self.m_CaO = np.zeros(n_nodes, dtype=float)
        
        # Oksitler
        self.m_SiO2  = np.zeros(n_nodes, dtype=float)
        self.m_Al2O3 = np.zeros(n_nodes, dtype=float)
        self.m_Fe2O3 = np.zeros(n_nodes, dtype=float)
        
        # Klinker Fazları (Yeni eklenenler)
        self.m_C2S  = np.zeros(n_nodes, dtype=float) # Belit
        self.m_C3S  = np.zeros(n_nodes, dtype=float) # Alit
        self.m_C3A  = np.zeros(n_nodes, dtype=float) # Trikalsiyum Alüminat
        self.m_C4AF = np.zeros(n_nodes, dtype=float) # Tetrakalsiyum Alüminoferrit
        
        # --- Gaz Fazı Özellikleri ---
        self.rho_g = np.zeros(n_nodes, dtype=float)
        
        # --- Geometrik Değişkenler ---
        self.fill_degree = np.zeros(n_nodes, dtype=float)
        self.v_s = np.zeros(n_nodes, dtype=float)

    def initialize_profiles(self, T_ambient=300.0, T_gas_inlet=1400.0, raw_meal_comp=None):
        self.Ts.fill(float(T_ambient))
        self.Tg = np.linspace(float(T_ambient), float(T_gas_inlet), self.N)
        self.X.fill(0.0)
        
        if raw_meal_comp is None:
            raw_meal_comp = {
                'CaCO3': 0.78, 
                'SiO2': 0.14, 
                'Al2O3': 0.05, 
                'Fe2O3': 0.03
            }

        # Başlangıç hammadde oranları
        self.m_CaCO3.fill(raw_meal_comp['CaCO3'])
        self.m_SiO2.fill(raw_meal_comp['SiO2'])
        self.m_Al2O3.fill(raw_meal_comp['Al2O3'])
        self.m_Fe2O3.fill(raw_meal_comp['Fe2O3'])
        
        # Ürünler başlangıçta sıfır
        self.m_CaO.fill(0.0)
        self.m_C2S.fill(0.0)
        self.m_C3S.fill(0.0)
        self.m_C3A.fill(0.0)
        self.m_C4AF.fill(0.0)

        self.update_gas_density()

    def update_gas_density(self, mw_gas=28.97e-3, pressure=101325.0):
        R_universal = 8.314
        safe_Tg = np.where(self.Tg < 100, 300.0, self.Tg)
        self.rho_g = (float(pressure) * float(mw_gas)) / (R_universal * safe_Tg)

    def get_state_vector(self):
        """
        State vektörünü klinker fazlarını da içerecek şekilde (18-State) genişletiyoruz.
        """
        return np.concatenate([
            self.Ts,      # 1
            self.Tg,      # 2
            self.X,       # 3
            self.m_CaO,   # 4
            self.m_C2S,   # 5
            self.m_C3S,   # 6
            self.m_C3A,   # 7
            self.m_C4AF,  # 8
            self.m_SiO2,  # 9
            self.m_Al2O3, # 10
            self.m_Fe2O3  # 11
        ])

    def set_state_from_vector(self, vec):
        """Vektörden state geri yükleme (N düğüm sayısına göre dilimleme)."""
        n = self.N
        self.Ts      = vec[0   : n]
        self.Tg      = vec[n   : 2*n]
        self.X       = vec[2*n : 3*n]
        self.m_CaO   = vec[3*n : 4*n]
        self.m_C2S   = vec[4*n : 5*n]
        self.m_C3S   = vec[5*n : 6*n]
        self.m_C3A   = vec[6*n : 7*n]
        self.m_C4AF  = vec[7*n : 8*n]
        self.m_SiO2  = vec[8*n : 9*n]
        self.m_Al2O3 = vec[9*n : 10*n]
        self.m_Fe2O3 = vec[10*n: 11*n]