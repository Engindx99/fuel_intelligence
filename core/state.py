import numpy as np


class KilnState:
    """
    Rotary Kiln distributed state container.

    Endüstriyel standarda daha yakın sade/state-only yapı:
    - Her node fiziksel bir kontrol hacmidir
    - Başlangıçta yalnızca giriş bölgesi doludur
    - Locked/rezerv SiO2 sistemi kaldırılmıştır
    - Total mass doğrudan species toplamından hesaplanır
    """

    def __init__(self, n_nodes):
        self.N = n_nodes

        # ==========================================================
        # SICAKLIK PROFİLLERİ
        # ==========================================================
        self.Tg = np.zeros(n_nodes, dtype=float)   # Gas temperature
        self.Ts = np.zeros(n_nodes, dtype=float)   # Solid temperature
        self.Tw = np.zeros(n_nodes, dtype=float)   # Wall temperature

        # ==========================================================
        # HAM MADDE BİLEŞENLERİ
        # ==========================================================
        self.m_CaCO3 = np.zeros(n_nodes, dtype=float)

        self.m_CaO = np.zeros(n_nodes, dtype=float)
        self.m_SiO2 = np.zeros(n_nodes, dtype=float)
        self.m_Al2O3 = np.zeros(n_nodes, dtype=float)
        self.m_Fe2O3 = np.zeros(n_nodes, dtype=float)

        # ==========================================================
        # KLİNKER FAZLARI
        # ==========================================================
        self.m_C2S = np.zeros(n_nodes, dtype=float)
        self.m_C3S = np.zeros(n_nodes, dtype=float)
        self.m_C3A = np.zeros(n_nodes, dtype=float)
        self.m_C4AF = np.zeros(n_nodes, dtype=float)

        # ==========================================================
        # REAKSİYON / GAZ ÇIKIŞI
        # ==========================================================
        self.m_CO2_released = np.zeros(n_nodes, dtype=float)

        # ==========================================================
        # FİZİKSEL PARAMETRELER
        # ==========================================================
        self.rho_g = np.zeros(n_nodes, dtype=float)
        self.v_s = np.zeros(n_nodes, dtype=float)

        # 0: Drying / Heating
        # 1: Calcination
        # 2: Burning / Belite
        # 3: Alite Formation
        self.zones = np.zeros(n_nodes, dtype=int)

    # ==============================================================
    # TOTAL MASS
    # ==============================================================

    @property
    def total_mass(self):
        return (
            self.m_CaCO3
            + self.m_CaO
            + self.m_SiO2
            + self.m_Al2O3
            + self.m_Fe2O3
            + self.m_C2S
            + self.m_C3S
            + self.m_C3A
            + self.m_C4AF
        )

    # ==============================================================
    # INITIALIZATION
    # ==============================================================

    def initialize_profiles(
        self,
        T_ambient=300.0,
        T_gas_inlet=2000.0,
        raw_meal_comp=None,
        feed_fill_ratio=0.15
    ):
        """
        Cold-start initialization.

        x=0   -> Feed end
        x=L   -> Burner end
        """

        # ----------------------------------------------------------
        # DEFAULT RAW MEAL
        # ----------------------------------------------------------

        if raw_meal_comp is None:
            raw_meal_comp = {
                "CaCO3": 0.76,
                "SiO2": 0.21,
                "Al2O3": 0.05,
                "Fe2O3": 0.03
            }

        # ----------------------------------------------------------
        # TEMPERATURE INITIALIZATION
        # ----------------------------------------------------------

        self.Ts.fill(float(T_ambient))

        # Refractory genelde ortamdan daha sıcak olur
        self.Tw.fill(float(T_ambient + 50.0))

        # Counter-current gas profile
        self.Tg = np.linspace(
            900.0,
            float(T_gas_inlet),
            self.N
        )

        # ----------------------------------------------------------
        # INITIAL EMPTY KILN
        # ----------------------------------------------------------

        self.m_CaCO3.fill(0.0)
        self.m_CaO.fill(0.0)
        self.m_SiO2.fill(0.0)
        self.m_Al2O3.fill(0.0)
        self.m_Fe2O3.fill(0.0)

        self.m_C2S.fill(0.0)
        self.m_C3S.fill(0.0)
        self.m_C3A.fill(0.0)
        self.m_C4AF.fill(0.0)

        self.m_CO2_released.fill(0.0)

        # ----------------------------------------------------------
        # ONLY FEED-END CONTAINS MATERIAL
        # ----------------------------------------------------------

        feed_nodes = max(1, int(feed_fill_ratio * self.N))

        self.m_CaCO3[:feed_nodes] = raw_meal_comp["CaCO3"]
        self.m_SiO2[:feed_nodes] = raw_meal_comp["SiO2"]
        self.m_Al2O3[:feed_nodes] = raw_meal_comp["Al2O3"]
        self.m_Fe2O3[:feed_nodes] = raw_meal_comp["Fe2O3"]

        # ----------------------------------------------------------
        # GAS DENSITY
        # ----------------------------------------------------------

        self.update_gas_density()

        # ----------------------------------------------------------
        # ZONE DEFINITIONS
        # ----------------------------------------------------------

        for i in range(self.N):

            x_ratio = i / (self.N - 1)

            # Drying / Heating
            if x_ratio < 0.25:
                self.zones[i] = 0

            # Calcination
            elif x_ratio < 0.60:
                self.zones[i] = 1

            # Burning / Belite formation
            elif x_ratio < 0.85:
                self.zones[i] = 2

            # Alite formation / hottest zone
            else:
                self.zones[i] = 3

    # ==============================================================
    # GAS DENSITY
    # ==============================================================

    def update_gas_density(
        self,
        mw_gas=29e-3,
        pressure=101325.0
    ):
        """
        Ideal gas density.
        """

        R_universal = 8.314

        safe_Tg = np.maximum(300.0, self.Tg)

        self.rho_g = (
            pressure * mw_gas
        ) / (
            R_universal * safe_Tg
        )

    # ==============================================================
    # QUALITY METRICS
    # ==============================================================

    def get_clinker_quality(self):

        idx = -1

        free_cao = self.m_CaO[idx]

        sio2 = max(1e-9, self.m_SiO2[idx])
        al2o3 = max(1e-9, self.m_Al2O3[idx])
        fe2o3 = max(1e-9, self.m_Fe2O3[idx])

        lsf_denom = (
            2.8 * sio2
            + 1.18 * al2o3
            + 0.65 * fe2o3
        )

        lsf = free_cao / max(lsf_denom, 1e-9)

        return {
            "C3S": self.m_C3S[idx],
            "C2S": self.m_C2S[idx],
            "fCaO": free_cao,
            "LSF": lsf
        }