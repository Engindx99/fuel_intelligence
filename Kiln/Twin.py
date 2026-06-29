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
            "RDF_Fuel": self.state.Fuel_rate,  # net isimlendirme
        }

        self._initialize_mass_energy_balance()
        self.outputs = {}

    # ======================================================
    def step(self, inputs: dict = None):

        # ================= INPUT UPDATE =================
        if inputs is not None:
            self.inputs.update(inputs)

        dt = self.state.dt

        # ======================================================
        # 1. BURNING (SOURCE SYSTEM)
        # ======================================================
        self.state = self.burning.apply(self.state, self.inputs, dt)

        # ======================================================
        # 2. EXTRACT BURNING OUTLET (COUPLING SOURCE)
        # ======================================================
        Tg_b_out = self.state.Tg_burning[-1]
        Ts_b_out = self.state.Ts_burning[-1]

        # ======================================================
        # 3. CALCINATION (SINK SYSTEM)
        # ======================================================
        calc_inputs = self.inputs.copy()
        calc_inputs["Tg_in"] = Tg_b_out
        calc_inputs["Ts_in"] = Ts_b_out

        self.state = self.calcination.apply(self.state, calc_inputs, dt)

        # ======================================================
        # 4. ENERGY SINK (FROM CALCINATION)
        # ======================================================
        # Q_sink = getattr(self.state, "Calcination_Q_sink", 0.0)
        Q_sink = 0.0

        #  IMPORTANT:
        # Burning’e tekrar apply YOK.
        # Enerji etkisi state içinde already accounted olmalı
        # (calcination heat removal already reduces Tg/Ts)

        # ======================================================
        # 5. PREHEATER + COOLER
        # ======================================================
        self.state = self.preheater.apply(self.state, self.inputs, dt)
        self.state = self.cooler.apply(self.state, self.inputs, dt)

        # ======================================================
        # 6. TIME UPDATE
        # ======================================================
        self.current_time += dt
        self.state.t = self.current_time

        # ======================================================
        # 7. OUTPUT
        # ======================================================
        self.outputs = {
            "t": self.current_time,
            "Feed_rate": self.state.Feed_rate,
            "Fuel_rate": self.state.Fuel_rate,
            "RDF_Fuel": self.inputs.get("RDF_Fuel", 0.0),
            "Q_sink_calcination": Q_sink,
            "Total_mass": self.state.Total_mass,
            "Total_enthalpy": self.state.Total_enthalpy,
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


twin = Twin()
twin.initialize()

for i in range(1000):
    twin.step()

    if i % 50 == 0:
        print(
            twin.state.t,
            twin.state.Tg_burning[-1],
            getattr(twin.state, "Calcination_Q_sink", 0.0),
        )
