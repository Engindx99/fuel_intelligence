import numpy as np

class KilnState:
    def __init__(self, N):
        self.N = N

        self.Tg = np.zeros(N)
        self.Ts = np.zeros(N)
        self.Tw = np.zeros(N)

        self.CaCO3 = np.zeros(N)
        self.CaO   = np.zeros(N)
        self.SiO2  = np.zeros(N)
        self.Al2O3 = np.zeros(N)
        self.Fe2O3 = np.zeros(N)

        self.C2S = np.zeros(N)
        self.C3S = np.zeros(N)
        self.C3A = np.zeros(N)
        self.C4AF = np.zeros(N)

        self.rho_g = np.ones(N)

    @property
    def total_mass(self):
        return (
            self.CaCO3 + self.CaO + self.SiO2 +
            self.Al2O3 + self.Fe2O3 +
            self.C2S + self.C3S + self.C3A + self.C4AF
        )

    def initialize_profiles(self,
                            T_ambient=300.0,
                            T_gas_inlet=2000.0,
                            raw_meal_comp=None):

        if raw_meal_comp is None:
            raw_meal_comp = {
                "CaCO3": 0.76,
                "SiO2": 0.21,
                "Al2O3": 0.05,
                "Fe2O3": 0.03
            }

        self.Ts.fill(T_ambient)
        self.Tw.fill(T_ambient + 50.0)
        self.Tg = np.linspace(900.0, T_gas_inlet, self.N)

        self.CaCO3.fill(0.0)
        self.CaO.fill(0.0)
        self.SiO2.fill(0.0)
        self.Al2O3.fill(0.0)
        self.Fe2O3.fill(0.0)

        self.C2S.fill(0.0)
        self.C3S.fill(0.0)
        self.C3A.fill(0.0)
        self.C4AF.fill(0.0)

        feed_nodes = max(1, int(0.15 * self.N))

        self.CaCO3[:feed_nodes] = raw_meal_comp["CaCO3"]
        self.SiO2[:feed_nodes]  = raw_meal_comp["SiO2"]
        self.Al2O3[:feed_nodes] = raw_meal_comp["Al2O3"]
        self.Fe2O3[:feed_nodes] = raw_meal_comp["Fe2O3"]