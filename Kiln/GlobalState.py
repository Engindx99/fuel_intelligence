from dataclasses import dataclass, field
from typing import Dict
import numpy as np


# ==========================================================
# GLOBAL STATE
# ==========================================================
@dataclass
class GlobalState:

    # ======================================================
    # SIMULATION
    # ======================================================
    t: float = 0.0
    dt: float = 0.05

    # ======================================================
    # OPERATION
    # ======================================================
    Feed_rate: float = 40.0
    Fuel_rate: float = 3.0
    Kiln_speed: float = 3.5

    # ======================================================
    # GAS PHASE
    # ======================================================
    Gas_flow: float = 0.0
    O2: float = 3.5
    CO2: float = 0.0
    H2O: float = 0.0
    N2: float = 0.0

    # ======================================================
    # SOLID PHASE
    # ======================================================
    Solid_flow: float = 0.0

    # ======================================================
    # ENERGY
    # ======================================================

    E_preheater: float = 0.0
    E_calcination: float = 0.0
    E_burning: float = 0.0
    E_cooler: float = 0.0

    # ======================================================
    # CHEMISTRY
    # ======================================================
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

    # ======================================================
    # PREHEATER STATES
    # ======================================================
    Tg_preheater: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (300.0 + 273.15)
    )

    Ts_preheater: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (300.0 + 273.15)
    )

    Tw_preheater: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (250.0 + 273.15)
    )

    # ======================================================
    # CALCINATION STATES
    # ======================================================
    Tg_calcination: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (1200.0 + 273.15)
    )

    Ts_calcination: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (950.0 + 273.15)
    )

    Tw_calcination: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (500.0 + 273.15)
    )

    # ======================================================
    # BURNING STATES
    # ======================================================
    Tg_burning: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (1500.0 + 273.15)
    )

    Ts_burning: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (1100.0 + 273.15)
    )

    Tw_burning: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (600.0 + 273.15)
    )

    # ======================================================
    # COOLER STATES
    # ======================================================
    Tg_cooler: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (150.0 + 273.15)
    )

    Ts_cooler: np.ndarray = field(
        default_factory=lambda: np.ones(80) * (120.0 + 273.15)
    )

    Tw_cooler: np.ndarray = field(default_factory=lambda: np.ones(80) * (80.0 + 273.15))

    # ======================================================
    # GLOBAL MASS & ENERGY
    # ======================================================
    Total_mass: float = 0.0
    Total_enthalpy: float = 0.0

    Fuel_energy: float = 0.0
    Heat_loss: float = 0.0

    Mass_balance_error: float = 0.0
    Energy_balance_error: float = 0.0

    # ======================================================
    # API
    # ======================================================
    Inputs: Dict = field(default_factory=dict)
    Outputs: Dict = field(default_factory=dict)
