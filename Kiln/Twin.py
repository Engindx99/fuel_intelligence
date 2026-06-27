from dataclasses import dataclass, field
from typing import Dict
from Kiln.GlobalState import GlobalState
from Kiln.Preheater import Preheater
from Kiln.Calcination import Calcination
from Kiln.Burning import Burning
from Kiln.Cooler import Cooler

# ==========================================================
# TWIN CORE
# ==========================================================


class Twin:

    def __init__(self):

        # ======================================================
        # GLOBAL STATE
        # ======================================================
        self.state = GlobalState()

        # ======================================================
        # ZONES
        # ======================================================
        self.preheater = None
        self.calcination = None
        self.burning = None
        self.cooler = None

        # ======================================================
        # SIMULATION
        # ======================================================
        self.current_time = 0.0
        self.time_step = self.state.dt

        # ======================================================
        # API
        # ======================================================
        self.inputs = {}
        self.outputs = {}

    # ======================================================
    # INITIALIZATION
    # ======================================================
    def initialize(self):

        # TIME RESET
        self.current_time = 0.0
        self.state.t = 0.0

        # ZONES INITIALIZATION
        self.preheater = Preheater()
        self.calcination = Calcination()
        self.burning = Burning()
        self.cooler = Cooler()

        # INPUTS
        self.inputs = {
            "Feed_rate": self.state.Feed_rate,
            "Fuel_rate": self.state.Fuel_rate,
            "Kiln_speed": self.state.Kiln_speed,
        }

        # MASS & ENERGY INIT
        self._initialize_mass_energy_balance()

        # OUTPUT STRUCTURE
        self.outputs = {}

    # ======================================================
    # STEP FUNCTION
    # ======================================================
    def step(self, inputs: dict = None):

        # ======================================================
        # 0. INPUT UPDATE
        # ======================================================
        if inputs is not None:
            self.inputs.update(inputs)

        self.state.Feed_rate = self.inputs.get("Feed_rate", self.state.Feed_rate)
        self.state.Fuel_rate = self.inputs.get("Fuel_rate", self.state.Fuel_rate)
        self.state.Kiln_speed = self.inputs.get("Kiln_speed", self.state.Kiln_speed)

        dt = self.state.dt

        # ======================================================
        # 1. ZONE EXECUTION (SEQUENTIAL OPERATOR SPLITTING)
        # ======================================================
        self.state = self.preheater.apply(self.state, self.inputs, dt)
        self.state = self.calcination.apply(self.state, self.inputs, dt)
        self.state = self.burning.apply(self.state, self.inputs, dt)
        self.state = self.cooler.apply(self.state, self.inputs, dt)

        # ======================================================
        # 2. GLOBAL MASS (SIMPLIFIED CONSISTENCY CHECK)
        # ======================================================
        self.state.Total_mass = (
            self.state.CaCO3
            + self.state.CaO
            + self.state.SiO2
            + self.state.Al2O3
            + self.state.Fe2O3
        )

        self.state.Mass_balance_error = 0.0  # placeholder (flux model later)

        # ======================================================
        # 3. GLOBAL ENERGY (PLACEHOLDER CONSISTENCY)
        # ======================================================
        self.state.Total_enthalpy = self.state.Fuel_energy

        self.state.Energy_balance_error = (
            self.state.Fuel_energy - self.state.Total_enthalpy - self.state.Heat_loss
        )

        # ======================================================
        # 4. TIME UPDATE
        # ======================================================
        self.current_time += dt
        self.state.t = self.current_time

        # ======================================================
        # 5. OUTPUT UPDATE
        # ======================================================
        self.outputs = {
            "t": self.current_time,
            "Feed_rate": self.state.Feed_rate,
            "Fuel_rate": self.state.Fuel_rate,
            "Total_mass": self.state.Total_mass,
            "Total_enthalpy": self.state.Total_enthalpy,
            "mass_error": self.state.Mass_balance_error,
            "energy_error": self.state.Energy_balance_error,
        }

        return self.state

    # ======================================================
    # MASS & ENERGY INIT
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
