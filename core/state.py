import numpy as np

class KilnState:
    """
    Döner fırın durum değişkenlerini ve kütle dengesini yöneten sınıf.
    Birimler: Sıcaklık [K], Kütle Kesri [kg_i/kg_total]
    """
    def __init__(self, N):
        self.N = N

        # Termal Profil
        self.Tg = np.zeros(N)  # Gaz sıcaklığı
        self.Ts = np.zeros(N)  # Malzeme sıcaklığı
        self.Tw = np.zeros(N)  # Duvar sıcaklığı

        # Katı Faz Bileşenleri (Kütle Kesri olarak tutulur: 0.0 - 1.0)
        self.CaCO3 = np.zeros(N)
        self.CaO   = np.zeros(N)
        self.SiO2  = np.zeros(N)
        self.Al2O3 = np.zeros(N)
        self.Fe2O3 = np.zeros(N)

        # Klinker Fazları
        self.C2S = np.zeros(N)
        self.C3S = np.zeros(N)
        self.C3A = np.zeros(N)
        self.C4AF = np.zeros(N)

        # Gaz Fazı Ürünleri
        self.CO2 = np.zeros(N)  # Kalsinasyon sonucu açığa çıkan CO2

        # Fiziksel Parametreler
        self.rho_g = np.ones(N)  # Gaz yoğunluğu
        self.total_mass_flow = np.zeros(N) # Toplam kütle akışı

    @property
    def check_mass_fraction(self):
        """
        Türlerin toplamının 1.0 olup olmadığını kontrol eder.
        Not: CO2 gaz fazına geçtiği için katı kütle dengesini CO2 dahil veya hariç 
        olarak iki farklı şekilde takip etmek isteyebilirsiniz.
        """
        total = (
            self.CaCO3 + self.CaO + self.SiO2 +
            self.Al2O3 + self.Fe2O3 +
            self.C2S + self.C3S + self.C3A + self.C4AF +
            self.CO2
        )
        return total

    def initialize_profiles(self,
                            T_ambient=300.0,
                            T_gas_inlet=2000.0,
                            feed_mass_flow=10.0,
                            raw_meal_comp=None):
        """
        Başlangıç profillerini ve sınır koşullarını atar.
        """
        if raw_meal_comp is None:
            raw_meal_comp = {
                "CaCO3": 0.76,
                "SiO2": 0.21,
                "Al2O3": 0.05,
                "Fe2O3": 0.03
            }

        # Sıcaklık Başlatma
        self.Ts.fill(T_ambient)
        self.Tw.fill(T_ambient + 50.0)
        self.Tg = np.linspace(900.0, T_gas_inlet, self.N)

        # Tüm fırın boyunca kütle akışını başlat
        self.total_mass_flow.fill(feed_mass_flow)

        # Bileşenleri sıfırla (CO2 dahil)
        comp_list = ["CaCO3", "CaO", "SiO2", "Al2O3", "Fe2O3", 
                     "C2S", "C3S", "C3A", "C4AF", "CO2"]
        for attr in comp_list:
            getattr(self, attr).fill(0.0)

        # Giriş düğümlerine besleme kompozisyonunu ata
        feed_nodes = max(1, int(0.05 * self.N)) 
        
        self.CaCO3[:feed_nodes] = raw_meal_comp["CaCO3"]
        self.SiO2[:feed_nodes]  = raw_meal_comp["SiO2"]
        self.Al2O3[:feed_nodes] = raw_meal_comp["Al2O3"]
        self.Fe2O3[:feed_nodes] = raw_meal_comp["Fe2O3"]