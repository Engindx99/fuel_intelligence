import numpy as np

class KilnState:
    """
    Döner fırın durum değişkenlerini ve kütle dengesini yöneten sınıf.
    Birimler: Sıcaklık [K], Kütle Kesri [kg_i / kg_total_solid]
    """
    def __init__(self, N):
        self.N = N

        # --- Termal Profil ---
        self.Tg = np.zeros(N, dtype=np.float64)  # Gaz sıcaklığı
        self.Ts = np.zeros(N, dtype=np.float64)  # Malzeme (katı) sıcaklığı
        self.Tw = np.zeros(N, dtype=np.float64)  # Fırın iç duvar sıcaklığı

        # --- Katı Faz Bileşenleri (Kütle Kesri: 0.0 - 1.0) ---
        # Bu değerler katı kütlesi içindeki oranları temsil eder.
        self.CaCO3 = np.zeros(N, dtype=np.float64)
        self.CaO   = np.zeros(N, dtype=np.float64)
        self.SiO2  = np.zeros(N, dtype=np.float64)
        self.Al2O3 = np.zeros(N, dtype=np.float64)
        self.Fe2O3 = np.zeros(N, dtype=np.float64)

        # --- Klinker Fazları ---
        self.C2S = np.zeros(N, dtype=np.float64)
        self.C3S = np.zeros(N, dtype=np.float64)
        self.C3A = np.zeros(N, dtype=np.float64)
        self.C4AF = np.zeros(N, dtype=np.float64)

        # --- Gaz Fazı Ürünleri ---
        # CO2 burada gaz fazındaki kütle kesrini veya düğüm başına biriken kütleyi temsil edebilir.
        # Solver'daki taşınım mantığına göre kütle kesri olarak yönetilir.
        self.CO2 = np.zeros(N, dtype=np.float64) 

        # --- Fiziksel Parametreler ---
        self.rho_g = np.ones(N, dtype=np.float64) * 1.2 # Nominal hava yoğunluğu (kg/m3)

    @property
    def check_mass_fraction(self):
        """
        Katı fazdaki türlerin toplamını kontrol eder. 
        Kalsinasyon sonrası CO2 gaz fazına geçtiği için, 
        stokiyometrik denge solver içinde kütle kaybı olarak yönetilmelidir.
        """
        total_solid = (
            self.CaCO3 + self.CaO + self.SiO2 +
            self.Al2O3 + self.Fe2O3 +
            self.C2S + self.C3S + self.C3A + self.C4AF
        )
        return total_solid

    def initialize_profiles(self,
                            T_ambient=300.0,
                            T_gas_inlet=2000.0,
                            raw_meal_comp=None):
        """
        Başlangıç profillerini Solver'daki taşınım (Upwind) mantığına uygun başlatır.
        """
        if raw_meal_comp is None:
            raw_meal_comp = {
                "CaCO3": 0.76,
                "SiO2": 0.14,
                "Al2O3": 0.03,
                "Fe2O3": 0.02
            }

        # 1. Sıcaklık Başlatma
        # Ts: Besleme sıcaklığından başlar (Sayısal şokları önlemek için tüm düğümler T_ambient)
        self.Ts.fill(T_ambient)
        self.Tw.fill(T_ambient + 100.0) # Duvar biraz daha sıcak varsayılabilir
        
        # Tg: Solver içinde linspace ile ezilecek olsa da burada tutarlılık için atanır.
        self.Tg = np.linspace(T_ambient + 200.0, T_gas_inlet, self.N)

        # 2. Bileşenlerin Tüm Fırın Boyunca Başlatılması
        # ÖNEMLİ: Upwind şemasında [i] düğümü [i-1]'den beslendiği için,
        # fırının içini boş (0.0) başlatmak yerine ham karışımla dolu başlatmak 
        # simülasyonun kararlı hale gelme süresini (settling time) kısaltır.
        
        self.CaCO3.fill(raw_meal_comp.get("CaCO3", 0.0))
        self.SiO2.fill(raw_meal_comp.get("SiO2", 0.0))
        self.Al2O3.fill(raw_meal_comp.get("Al2O3", 0.0))
        self.Fe2O3.fill(raw_meal_comp.get("Fe2O3", 0.0))

        # Ürün fazları başlangıçta fırın içinde sıfırdır.
        other_phases = ["CaO", "C2S", "C3S", "C3A", "C4AF", "CO2"]
        for attr in other_phases:
            getattr(self, attr).fill(0.0)

    def get_solid_state_vector(self):
        """Kinetik hesaplamalar için toplu matris döndürür."""
        return np.vstack((
            self.CaCO3, self.CaO, self.SiO2, self.Al2O3, self.Fe2O3,
            self.C2S, self.C3S, self.C3A, self.C4AF
        ))