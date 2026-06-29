from Kiln.GlobalState import GlobalState
from Kiln.Preheater import Preheater
from Kiln.Calcination import Calcination
from Kiln.Burning import Burning
from Kiln.Cooler import Cooler


class Twin:

    def __init__(self):

        self.state = GlobalState()

        self.preheater = Preheater()
        self.calcination = Calcination()
        self.burning = Burning()
        self.cooler = Cooler()

        self.current_time = 0.0
        self.inputs = {}
        self.outputs = {}

    # ======================================================
    def initialize(self):

        self.current_time = 0.0
        self.state.t = 0.0

        self.inputs = {
            "Feed_rate": self.state.Feed_rate,
            "Fuel_rate": self.state.Fuel_rate,
            "Kiln_speed": self.state.Kiln_speed,
            "RDF_Fuel": 0.2,  # FIXED (NOT duplicated fuel_rate)
        }

        self._initialize_mass_energy_balance()

    # ======================================================
    def step(self, inputs: dict = None):

        if inputs is not None:
            self.inputs.update(inputs)

        dt = self.state.dt

        # ======================================================
        # 1. BURNING (SOURCE)
        # ======================================================
        self.state = self.burning.apply(self.state, self.inputs, dt)

        Tg_b_out = self.state.Tg_burning[-1]
        Ts_b_out = self.state.Ts_burning[-1]
        Tw_b_out = self.state.Tw_burning[-1]

        # ======================================================
        # 2. CALCINATION INPUT (SMOOTHED COUPLING)
        # ======================================================
        alpha = 0.7

        if not hasattr(self.state, "Tg_in"):
            self.state.Tg_in = Tg_b_out
            self.state.Ts_in = Ts_b_out
            self.state.Tw_in = Tw_b_out

        self.state.Tg_in = alpha * Tg_b_out + (1 - alpha) * self.state.Tg_in
        self.state.Ts_in = alpha * Ts_b_out + (1 - alpha) * self.state.Ts_in
        self.state.Tw_in = alpha * Tw_b_out + (1 - alpha) * self.state.Tw_in

        calc_inputs = self.inputs.copy()
        calc_inputs["Tg_in"] = self.state.Tg_in
        calc_inputs["Ts_in"] = self.state.Ts_in
        calc_inputs["Tw_in"] = self.state.Tw_in

        # ======================================================
        # 3. CALCINATION (SINK)
        # ======================================================
        self.state = self.calcination.apply(self.state, calc_inputs, dt)

        Q_sink = getattr(self.state, "Calcination_Q_sink", 0.0)

        # ======================================================
        # 4. PREHEATER
        # ======================================================
        self.state = self.preheater.apply(self.state, self.inputs, dt)

        # ======================================================
        # 5. COOLER
        # ======================================================
        self.state = self.cooler.apply(self.state, self.inputs, dt)

        # ======================================================
        # 6. TIME UPDATE
        # ======================================================
        self.current_time += dt
        self.state.t = self.current_time

        # ======================================================
        # 7. MASS + ENERGY BALANCE UPDATE
        # ======================================================
        self.state.Total_enthalpy -= Q_sink * dt  # REAL FEEDBACK

        # ======================================================
        # 8. OUTPUT
        # ======================================================
        self.outputs = {
            "t": self.current_time,
            "Tg_b_out": Tg_b_out,
            "Ts_b_out": Ts_b_out,
            "Q_sink_calcination": Q_sink,
            "Tg_in_calcination": self.state.Tg_in,
        }

        return self.state

    # ======================================================
    def _initialize_mass_energy_balance(self):

        self.state.Total_mass = (
            self.state.CaCO3
            + self.state.CaO
            + self.state.SiO2
            + self.state.Al2O3
            + self.state.Fe2O3
        )

        self.state.Total_enthalpy = self.state.Fuel_energy
        self.state.Mass_balance_error = 0.0
        self.state.Energy_balance_error = 0.0
