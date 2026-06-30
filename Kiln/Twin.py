import numpy as np
from Kiln.GlobalState import GlobalState
from Kiln.Preheater import Preheater
from Kiln.Calcination import Calcination
from Kiln.Burning import Burning
from Kiln.Cooler import Cooler


class Twin:

    def __init__(self):

        # ======================================================
        # SUBSYSTEMS
        # ======================================================
        self.state = GlobalState()

        self.preheater = Preheater()
        self.calcination = Calcination()
        self.burning = Burning()
        self.cooler = Cooler()

        self.current_time = 0.0
        self.inputs = {}
        self.outputs = {}

        # ======================================================
        # FUEL CORE (ONLY SOURCE)
        # ======================================================
        self.LHV_petcoke = 32e6
        self.LHV_lignite = 18e6
        self.LHV_RDF = 25e6

        self.O2_opt = 3.5
        self.O2_sigma2 = 25.0

        self.eps = 1e-12

    # ======================================================
    def combustion_efficiency(self, O2):
        return np.exp(-((O2 - self.O2_opt) ** 2) / self.O2_sigma2)

    # ======================================================
    def compute_fuel_energy(self, inputs):

        p = inputs.get("Petcoke", 0.6)
        a = inputs.get("RDF_Fuel", 0.2)
        l = max(1.0 - p - a, 0.0)

        norm = p + a + l + self.eps
        p, a, l = p / norm, a / norm, l / norm

        fuel_rate = inputs.get("Fuel_rate", 1.0)
        O2 = inputs.get("O2", 3.5)

        eta = self.combustion_efficiency(O2)

        LHV_mix = p * self.LHV_petcoke + l * self.LHV_lignite + a * self.LHV_RDF

        Q_in = fuel_rate * LHV_mix * eta

        return Q_in

    # ======================================================
    def initialize(self):

        self.current_time = 0.0
        self.state.t = 0.0

        self.inputs = {
            "Feed_rate": self.state.Feed_rate,
            "Fuel_rate": self.state.Fuel_rate,
            "Kiln_speed": self.state.Kiln_speed,
            "RDF_Fuel": 0.2,
            "O2": 3.5,
        }

    # ======================================================
    def _get_Q_in(self):

        Q_in = self.compute_fuel_energy(self.inputs)
        print(f"[DEBUG] Q_in = {Q_in:.2e}")
        return Q_in

    # ======================================================

    def step(self, inputs=None):

        print("[DEBUG] TWIN STEP CALLED")

        if inputs is not None:
            self.inputs.update(inputs)

        dt = self.state.dt

        # ================= ENERGY =================
        Q_in = self._get_Q_in()

        print("[DEBUG] Q_in =", Q_in)

        # ================= ZONE INPUTS =================
        preheater_inputs = {**self.inputs, "Q_in": Q_in}
        calc_inputs = {**self.inputs, "Q_in": Q_in}
        burning_inputs = {**self.inputs, "Q_burning": Q_in}
        cooler_inputs = {**self.inputs, "Q_in": Q_in}

        print("[TYPE burning_inputs]", type(burning_inputs))
        print("[VALUE burning_inputs]", burning_inputs)

        # ================= EXECUTION =================
        self.state = self.preheater.apply(self.state, preheater_inputs, dt)
        self.state = self.calcination.apply(self.state, calc_inputs, dt)
        self.state = self.burning.apply(self.state, burning_inputs, dt)
        self.state = self.cooler.apply(self.state, cooler_inputs, dt)

        self.current_time += dt
        self.state.t = self.current_time

        self.outputs = {"t": self.current_time, "Q_in": Q_in}

        return self.state


print("OUTSIDE TWIN")

twin = Twin()
twin.initialize()

for i in range(10):
    state = twin.step()
