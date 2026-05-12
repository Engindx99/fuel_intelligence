import numpy as np

class KilnState:
    def __init__(self, N):
        self.N = N

        # --- Termal Profil ---
        self.Tg = np.zeros(N, dtype=np.float64)  
        self.Ts = np.zeros(N, dtype=np.float64)  
        self.Tw = np.zeros(N, dtype=np.float64)  

        # --- Katı Faz Bileşenleri (Kütle Kesri) ---
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

        # --- Gaz Fazı Takibi ---
        self.CO2_released = np.zeros(N, dtype=np.float64) # kg/s cinsinden her düğümde çıkan CO2

    @property
    def total_solid_fraction(self):
        """
        Kütle koruması kontrolü. 
        NOT: Kalsinasyon bölgesinde bu değer 1.0'ın altına düşecektir.
        Çünkü CO2 katı fazdan ayrılır.
        """
        return (self.CaCO3 + self.CaO + self.SiO2 + self.Al2O3 + self.Fe2O3 +
                self.C2S + self.C3S + self.C3A + self.C4AF)

    def initialize_profiles(self, config, raw_meal_comp=None):
        """
        Config dosyasından gelen parametrelerle profilleri başlatır.
        """
        # Config hiyerarşisine göre güvenli okuma
        mat_cfg = config.get("material", {})
        gas_cfg = config.get("gas", {})
        
        T_ambient = mat_cfg.get("temp_inlet", 300.0)
        T_gas_max = gas_cfg.get("temp_inlet", 2200.0)

        # 1. Sıcaklık Başlatma (Lineer bir tahmin nümerik yakınsamayı hızlandırır)
        self.Ts.fill(T_ambient)
        self.Tw.fill(T_ambient + 50.0)
        self.Tg = np.linspace(T_ambient + 100.0, T_gas_max, self.N)

        # 2. Bileşen Başlatma (Ham karışım normalizasyonu)
        if raw_meal_comp is None:
            raw_meal_comp = config.get("raw_meal_composition", {})

        # Toplamı 1.0 yapacak şekilde normalize et
        total_sum = sum(raw_meal_comp.values())
        if total_sum > 0:
            for key, val in raw_meal_comp.items():
                if hasattr(self, key):
                    getattr(self, key).fill(val / total_sum)

        # Diğer fazları temizle
        for attr in ["CaO", "C2S", "C3S", "C3A", "C4AF", "CO2_released"]:
            getattr(self, attr).fill(0.0)

    def get_solid_state_vector(self):
        """Kinetik çözücüye (ODE solver) gönderilecek vektör."""
        return np.array([
            self.CaCO3, self.CaO, self.SiO2, self.Al2O3, self.Fe2O3,
            self.C2S, self.C3S, self.C3A, self.C4AF
        ])
    
    def apply_solid_state_vector(self, vector):
        """Kinetik çözücüden dönen sonuçları düğümlere geri yazar."""
        (self.CaCO3, self.CaO, self.SiO2, self.Al2O3, self.Fe2O3,
         self.C2S, self.C3S, self.C3A, self.C4AF) = vector