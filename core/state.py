import numpy as np

class KilnState:
    def __init__(self, n_nodes):
        self.N = n_nodes
        
        # --- Sıcaklık Profilleri ---
        self.Tg = np.zeros(n_nodes, dtype=float) # Gaz Sıcaklığı
        self.Ts = np.zeros(n_nodes, dtype=float) # Malzeme Sıcaklığı
        self.Tw = np.zeros(n_nodes, dtype=float) # Fırın Duvar Sıcaklığı
        
        # --- Kimyasal Kompozisyon (Kütle Fraksiyonları) ---
        self.X = np.zeros(n_nodes, dtype=float)          # Kalsinasyon Derecesi (0.0 - 1.0)
        self.m_CaO = np.zeros(n_nodes, dtype=float)      # Serbest Kireç
        self.m_SiO2 = np.zeros(n_nodes, dtype=float)     # Serbest Silis
        self.m_SiO2_locked = np.zeros(n_nodes, dtype=float) # Reaksiyona Girmeyen Silis
        self.m_Al2O3 = np.zeros(n_nodes, dtype=float)    # Alümin
        self.m_Fe2O3 = np.zeros(n_nodes, dtype=float)    # Demir Oksit
        
        # --- Klinker Fazları ---
        self.m_C2S  = np.zeros(n_nodes, dtype=float)     # Belit
        self.m_C3S  = np.zeros(n_nodes, dtype=float)     # Alit
        self.m_C3A  = np.zeros(n_nodes, dtype=float)     # Trikalsiyum Alüminat
        self.m_C4AF = np.zeros(n_nodes, dtype=float)     # Tetrakalsiyum Alüminoferrit
        
        # --- Kütle ve Gaz Takibi ---
        self.m_co2_released = np.zeros(n_nodes, dtype=float) # Salınan CO2 kütlesi
        self.total_solid_mass = np.ones(n_nodes, dtype=float) # Toplam Katı Kütlesi (CO2 çıktıkça azalır)

        # --- Fiziksel Parametreler ---
        self.rho_g = np.zeros(n_nodes, dtype=float)      # Gaz Yoğunluğu
        self.v_s = np.zeros(n_nodes, dtype=float)        # Malzeme İlerleme Hızı

        # --- Bölge (Zone) Bilgisi ---
        self.zones = np.zeros(n_nodes, dtype=int)        # 0:Preheat, 1:Calc, 2:Trans, 3:Burn

    # --- GERİYE DÖNÜK UYUMLULUK (BACKWARD COMPATIBILITY) ---
    @property
    def total_mass(self):
        """run_simulation.py içindeki eski çağrıları karşılamak için."""
        return self.total_solid_mass

    @total_mass.setter
    def total_mass(self, value):
        self.total_solid_mass = value

    def initialize_profiles(self, T_ambient=300.0, T_gas_inlet=2200.0, raw_meal_comp=None):
        """
        Simülasyon başlangıç koşullarını ayarlar.
        """
        self.Ts.fill(float(T_ambient))
        self.Tw.fill(float(T_ambient))
        
        # Gaz sıcaklığı fırın boyunca lineer bir tahminle başlar (Solver bunu günceller)
        self.Tg = np.linspace(float(T_ambient + 500.0), float(T_gas_inlet), self.N)
        
        self.X.fill(0.0)
        self.m_co2_released.fill(0.0)
        
        if raw_meal_comp is None:
            raw_meal_comp = {
                'CaCO3': 0.730, 
                'SiO2': 0.210, 
                'Al2O3': 0.040, 
                'Fe2O3': 0.020
            }

        # Ham un bileşenlerinin atanması
        sio2_total = float(raw_meal_comp['SiO2'])
        # Silisin bir kısmı reaktif (m_SiO2), bir kısmı kristal yapıda kilitli varsayılabilir
        self.m_SiO2.fill(sio2_total * 0.70) 
        self.m_SiO2_locked.fill(sio2_total * 0.30)
        
        self.m_Al2O3.fill(float(raw_meal_comp['Al2O3']))
        self.m_Fe2O3.fill(float(raw_meal_comp['Fe2O3']))
        
        # Başlangıçta klinker fazları ve serbest kireç yoktur
        self.m_CaO.fill(1e-12)
        self.m_C2S.fill(0.0)   
        self.m_C3S.fill(0.0)
        self.m_C4AF.fill(0.0)
        self.m_C3A.fill(0.0)
        
        # Başlangıç kütlesi normalize edilmiştir
        self.total_solid_mass.fill(1.0) 

        # Gaz yoğunluğu ilk hesaplama
        self.update_gas_density()
        
        # Bölge bilgilerini güncelle (0.0 - 1.0 arası x_ratio'ya göre)
        for i in range(self.N):
            x_ratio = i / (self.N - 1)
            if x_ratio < 0.35:
                self.zones[i] = 0 # Preheat
            elif x_ratio < 0.60:
                self.zones[i] = 1 # Calcination
            elif x_ratio < 0.80:
                self.zones[i] = 2 # Transition
            else:
                self.zones[i] = 3 # Burning

    def update_gas_density(self, mw_gas=28.97e-3, pressure=101325.0):
        """
        İdeal gaz yasasına göre sıcaklığa bağlı gaz yoğunluğunu günceller.
        """
        R_universal = 8.314
        # Bölü sıfır hatasını engellemek için 273.15K (0C) alt sınırı
        safe_Tg = np.maximum(273.15, self.Tg)
        self.rho_g = (float(pressure) * float(mw_gas)) / (R_universal * safe_Tg)

    def get_clinker_quality(self):
        """
        Fırın sonundaki (çıkış node) ana fazları döndürür.
        """
        idx = -1 # Çıkış ucu
        return {
            "C3S": self.m_C3S[idx],
            "C2S": self.m_C2S[idx],
            "fCaO": self.m_CaO[idx],
            "Alite_Belite_Ratio": self.m_C3S[idx] / (self.m_C2S[idx] + 1e-9)
        }