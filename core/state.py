import numpy as np


class KilnState:
    def __init__(self, N):
        self.N = N

        # --- Termal Profil ---
        self.Tg = np.zeros(N, dtype=np.float64)
        self.Ts = np.zeros(N, dtype=np.float64)
        self.Tw = np.zeros(N, dtype=np.float64)

        # --- Katı Faz Bileşenleri ---
        self.CaCO3 = np.zeros(N, dtype=np.float64)
        self.CaO   = np.zeros(N, dtype=np.float64)
        self.SiO2  = np.zeros(N, dtype=np.float64)
        self.Al2O3 = np.zeros(N, dtype=np.float64)
        self.Fe2O3 = np.zeros(N, dtype=np.float64)

        # --- Klinker Fazları ---
        self.C2S  = np.zeros(N, dtype=np.float64)
        self.C3S  = np.zeros(N, dtype=np.float64)
        self.C3A  = np.zeros(N, dtype=np.float64)
        self.C4AF = np.zeros(N, dtype=np.float64)

        # ======================================================
        # 🔧 FIX: CO2 tutarlılığı (solver uyumu için)
        # ======================================================
        self.CO2 = np.zeros(N, dtype=np.float64)

        # ======================================================
        # 🔥 TRANSIENT HISTORY BUFFER
        # ======================================================
        self.history_enabled = True
        self._time_history = []
        self._Ts_history = []
        self._Tg_history = []
        self._CaCO3_history = []
        self._CaO_history = []
        self._C2S_history = []
        self._C3S_history = []

    # ======================================================
    # GLOBAL MASS CHECK
    # ======================================================
    @property
    def total_solid_fraction(self):
        return (
            self.CaCO3 + self.CaO + self.SiO2 + self.Al2O3 + self.Fe2O3 +
            self.C2S + self.C3S + self.C3A + self.C4AF
        )

    # ======================================================
    # INITIALIZATION
    # ======================================================
    def initialize_profiles(self, config, raw_meal_comp=None):

        mat_cfg = config.get("material", {})
        gas_cfg = config.get("gas", {})

        T_ambient = mat_cfg.get("temp_inlet", 300.0)
        T_gas_max = gas_cfg.get("temp_inlet", 2200.0)

        self.Ts.fill(T_ambient)
        self.Tw.fill(T_ambient + 50.0)
        self.Tg = np.linspace(T_ambient + 100.0, T_gas_max, self.N)

        if raw_meal_comp is None:
            raw_meal_comp = config.get("raw_meal_composition", {})

        total_sum = sum(raw_meal_comp.values())

        if total_sum > 0:
            for key, val in raw_meal_comp.items():
                if hasattr(self, key):
                    getattr(self, key).fill(val / total_sum)

        # reset products
        self.CaO.fill(0.0)
        self.C2S.fill(0.0)
        self.C3S.fill(0.0)
        self.C3A.fill(0.0)
        self.C4AF.fill(0.0)
        self.CO2.fill(0.0)

    # ======================================================
    # SAFE VECTOR INTERFACE (FIXED)
    # ======================================================
    def get_solid_state_vector(self):
        return np.stack([
            self.CaCO3,
            self.CaO,
            self.SiO2,
            self.Al2O3,
            self.Fe2O3,
            self.C2S,
            self.C3S,
            self.C3A,
            self.C4AF
        ])

    def apply_solid_state_vector(self, vector):
        assert len(vector) == 9, "State vector must have 9 components"

        (
            self.CaCO3,
            self.CaO,
            self.SiO2,
            self.Al2O3,
            self.Fe2O3,
            self.C2S,
            self.C3S,
            self.C3A,
            self.C4AF
        ) = vector

    # ======================================================
    # SNAPSHOT SYSTEM (UNCHANGED)
    # ======================================================
    def snapshot(self, t):
        if not self.history_enabled:
            return

        self._time_history.append(t)
        self._Ts_history.append(self.Ts.copy())
        self._Tg_history.append(self.Tg.copy())
        self._CaCO3_history.append(self.CaCO3.copy())
        self._CaO_history.append(self.CaO.copy())
        self._C2S_history.append(self.C2S.copy())
        self._C3S_history.append(self.C3S.copy())

    # ======================================================
    # ACCESSORS
    # ======================================================
    @property
    def time_history(self):
        return np.array(self._time_history)

    @property
    def Ts_history(self):
        return np.array(self._Ts_history)

    @property
    def Tg_history(self):
        return np.array(self._Tg_history)

    @property
    def CaCO3_history(self):
        return np.array(self._CaCO3_history)

    @property
    def CaO_history(self):
        return np.array(self._CaO_history)

    @property
    def C2S_history(self):
        return np.array(self._C2S_history)

    @property
    def C3S_history(self):
        return np.array(self._C3S_history)