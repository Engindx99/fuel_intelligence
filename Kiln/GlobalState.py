from dataclasses import dataclass, field
from typing import Dict


# ==========================================================
# GLOBAL STATE
# ==========================================================
@dataclass
class GlobalState:

    # SIMULATION
    t: float = 0.0
    dt: float = 0.05

    # OPERATION
    Feed_rate: float = 40.0
    Fuel_rate: float = 3.0
    Kiln_speed: float = 3.5

    # GAS PHASE
    Gas_flow: float = 0.0
    O2: float = 0.0
    CO2: float = 0.0
    H2O: float = 0.0
    N2: float = 0.0

    # SOLID PHASE
    Solid_flow: float = 0.0

    # CHEMISTRY
    CaCO3: float = 0.0
    CaO: float = 0.0
    SiO2: float = 0.0
    Al2O3: float = 0.0
    Fe2O3: float = 0.0

    C3S: float = 0.0
    C2S: float = 0.0
    C3A: float = 0.0
    C4AF: float = 0.0

    Free_Lime: float = 0.0

    # TEMPERATURES
    Tg_preheater: float = 25.0
    Ts_preheater: float = 25.0

    Tg_calcination: float = 850.0
    Ts_calcination: float = 850.0

    Tg_burning: float = 1450.0
    Ts_burning: float = 1400.0

    Tg_cooler: float = 150.0
    Ts_cooler: float = 120.0

    # GLOBAL MASS & ENERGY
    Total_mass: float = 0.0
    Total_enthalpy: float = 0.0

    Fuel_energy: float = 0.0
    Heat_loss: float = 0.0

    Mass_balance_error: float = 0.0
    Energy_balance_error: float = 0.0

    # MPC / RL / API
    Inputs: Dict = field(default_factory=dict)
    Outputs: Dict = field(default_factory=dict)
