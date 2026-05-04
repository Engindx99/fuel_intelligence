import numpy as np

class KilnState:
    def __init__(self, n_nodes):
        self.N = n_nodes
        
        # --- Sıcaklık Profilleri ---
        self.Tg = np.zeros(n_nodes, dtype=float)
        self.Ts = np.zeros(n_nodes, dtype=float)
        self.Tw = np.zeros(n_nodes, dtype=float)
        
        # --- Kimyasal Kompozisyon ve Dönüşüm ---
        self.X = np.zeros(n_nodes, dtype=float)  # Kalsinasyon derecesi
        self.m_CaCO3 = np.zeros(n_nodes, dtype=float)
        self.m_CaO = np.zeros(n_nodes, dtype=float)
        
        # --- Eksik Oksit Bileşikleri (Kütle Oranları) ---
        self.m_SiO2  = np.zeros(n_nodes, dtype=float)
        self.m_Al2O3 = np.zeros(n_nodes, dtype=float)
        self.m_Fe2O3 = np.zeros(n_nodes, dtype=float)
        
        # --- Gaz Fazı Özellikleri ---
        self.P_CO2 = np.zeros(n_nodes, dtype=float)
        self.rho_g = np.zeros(n_nodes, dtype=float)
        
        # --- Geometrik Değişkenler ---
        self.fill_degree = np.zeros(n_nodes, dtype=float)
        self.v_s = np.zeros(n_nodes, dtype=float)

    def initialize_profiles(self, T_ambient=300.0, T_gas_inlet=1400.0, raw_meal_comp=None):
        """
        raw_meal_comp: Giriş hammadde kompozisyonu (sözlük)
        Örn: {'CaCO3': 0.78, 'SiO2': 0.14, 'Al2O3': 0.05, 'Fe2O3': 0.03}
        """
        self.Ts.fill(float(T_ambient))
        self.Tg = np.linspace(float(T_ambient), float(T_gas_inlet), self.N)
        self.X.fill(0.0)
        
        # Varsayılan hammadde kompozisyonu (Eğer belirtilmemişse)
        if raw_meal_comp is None:
            raw_meal_comp = {
                'CaCO3': 0.78, 
                'SiO2': 0.14, 
                'Al2O3': 0.05, 
                'Fe2O3': 0.03
            }

        # Tüm fırın boyunca başlangıç kütle oranlarını ata
        self.m_CaCO3.fill(raw_meal_comp['CaCO3'])
        self.m_SiO2.fill(raw_meal_comp['SiO2'])
        self.m_Al2O3.fill(raw_meal_comp['Al2O3'])
        self.m_Fe2O3.fill(raw_meal_comp['Fe2O3'])
        self.m_CaO.fill(0.0) # Başlangıçta CaO yok

        self.update_gas_density()

    def update_gas_density(self, mw_gas=28.97e-3, pressure=101325.0):
        R_universal = 8.314
        safe_Tg = np.where(self.Tg < 100, 300.0, self.Tg)
        self.rho_g = (float(pressure) * float(mw_gas)) / (R_universal * safe_Tg)

    def get_state_vector(self):
        """
        Optimizasyon ve kontrol için state vektörünü genişletiyoruz.
        Not: SiO2, Al2O3 ve Fe2O3 kalsinasyon sırasında (şimdilik) değişmediği için
        genellikle kontrol vektörüne eklenmezler ama izleme için eklenebilir.
        """
        return np.concatenate([self.Ts, self.Tg, self.X, self.m_CaO])

    def set_state_from_vector(self, vec):
        # Vektör boyutu get_state_vector ile uyumlu olmalı
        self.Ts = vec[0 : self.N]
        self.Tg = vec[self.N : 2 * self.N]
        self.X  = vec[2 * self.N : 3 * self.N]
        self.m_CaO = vec[3 * self.N : 4 * self.N]