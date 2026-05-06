import numpy as np

class KilnState:
    def __init__(self, n_nodes):
        self.N = n_nodes
        
        # --- Sıcaklık Profilleri ---
        self.Tg = np.zeros(n_nodes, dtype=float)
        self.Ts = np.zeros(n_nodes, dtype=float)
        self.Tw = np.zeros(n_nodes, dtype=float)
        
        # --- Kimyasal Kompozisyon (Kütle Oranları) ---
        self.X = np.zeros(n_nodes, dtype=float)  # Kalsinasyon derecesi (0.0 - 1.0)
        self.m_CaCO3 = np.zeros(n_nodes, dtype=float)
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
        self.fill_degree = np.zeros(n_nodes, dtype=float)
        self.v_s = np.zeros(n_nodes, dtype=float)

    def initialize_profiles(self, T_ambient=300.0, T_gas_inlet=1400.0, raw_meal_comp=None):
        """Profil başlangıç değerlerini atar."""
        self.Ts.fill(float(T_ambient))
        self.Tw.fill(float(T_ambient))
        self.Tg = np.linspace(float(T_ambient), float(T_gas_inlet), self.N)
        self.X.fill(0.0)
        
        if raw_meal_comp is None:
            # Standart hammadde kompozisyonu
            raw_meal_comp = {
                'CaCO3': 0.82, 
                'SiO2': 0.13, 
                'Al2O3': 0.03, 
                'Fe2O3': 0.02
            }

        self.m_CaCO3.fill(raw_meal_comp['CaCO3'])
        self.m_SiO2.fill(raw_meal_comp['SiO2'])
        self.m_Al2O3.fill(raw_meal_comp['Al2O3'])
        self.m_Fe2O3.fill(raw_meal_comp['Fe2O3'])
        
        for phase in ['m_CaO', 'm_C2S', 'm_C3S', 'm_C3A', 'm_C4AF']:
            getattr(self, phase).fill(0.0)

        self.update_gas_density()

    def get_total_solid_mass(self):
        """
        Fiziksel Kütle Dengesi: 
        Kalsinasyon sırasında CO2 gazı (M_CO2 / M_CaCO3 = 0.44) sistemden ayrılır.
        Başlangıç kütlesi 1.0 birim kabul edildiğinde, kalan kütle kalsinasyon 
        oranına (X) bağlı olarak azalır.
        """
        co2_loss_factor = 0.44
        # Toplam kütle = 1.0 - (Başlangıç CaCO3 Payı * X * 0.44)
        # Girişteki CaCO3 miktarını referans alıyoruz (ilk düğüm değeri)
        m_CaCO3_initial = self.m_CaCO3[0] 
        return 1.0 - (m_CaCO3_initial * self.X * co2_loss_factor)

    def update_gas_density(self, mw_gas=28.97e-3, pressure=101325.0):
        """İdeal gaz yasasına göre yerel gaz yoğunluğunu günceller."""
        R_universal = 8.314
        safe_Tg = np.maximum(100.0, self.Tg)
        self.rho_g = (float(pressure) * float(mw_gas)) / (R_universal * safe_Tg)

    def get_state_vector(self):
        """MPC ve Kararlı Hal Analizi için eyalet vektörü."""
        return np.concatenate([
            self.Ts.copy(),      # 0
            self.Tg.copy(),      # 1
            self.Tw.copy(),      # 2
            self.X.copy(),       # 3
            self.m_CaO.copy(),   # 4
            self.m_C2S.copy(),   # 5
            self.m_C3S.copy(),   # 6
            self.m_C3A.copy(),   # 7
            self.m_C4AF.copy(),  # 8
            self.m_SiO2.copy(),  # 9
            self.m_Al2O3.copy(), # 10
            self.m_Fe2O3.copy()  # 11
        ])

    def set_state_from_vector(self, vec):
        """Vektörden durum verilerini geri yükler."""
        n = self.N
        self.Ts      = vec[0   : n].copy()
        self.Tg      = vec[n   : 2*n].copy()
        self.Tw      = vec[2*n : 3*n].copy()
        self.X       = vec[3*n : 4*n].copy()
        self.m_CaO   = vec[4*n : 5*n].copy()
        self.m_C2S   = vec[5*n : 6*n].copy()
        self.m_C3S   = vec[6*n : 7*n].copy()
        self.m_C3A   = vec[7*n : 8*n].copy()
        self.m_C4AF  = vec[8*n : 9*n].copy()
        self.m_SiO2  = vec[9*n : 10*n].copy()
        self.m_Al2O3 = vec[10*n: 11*n].copy()
        self.m_Fe2O3 = vec[11*n: 12*n].copy()