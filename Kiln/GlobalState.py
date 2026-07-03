from dataclasses import dataclass, field
from typing import Dict
import numpy as np


# ==========================================================
# GLOBAL STATE (CONSISTENT 5-CELL SYSTEM)
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
    Feed_rate: float = 40.0  # ton/h
    Fuel_rate: float = 3.0  # ton/h
    Kiln_speed: float = 3.5  # rpm

    # ======================================================
    # GAS PHASE
    # ======================================================
    Gas_flow: float = 0.0
    O2: float = 3.5  # %
    CO2: float = 0.0
    H2O: float = 0.0
    N2: float = 0.0

    # ======================================================
    # SOLID PHASE
    # ======================================================
    Solid_flow: float = 0.0

    # ======================================================
    # DIAGNOSTIC ENERGY STATES (NOT IN PDE)
    # ======================================================
    E_preheater: float = 0.0
    E_calcination: float = 0.0
    E_burning: float = 0.0
    E_cooler: float = 0.0
    
    # ======================================================
    # REACTION STATES
    # ======================================================

    alpha_calcination: float = 0.0

    Calcination_Q_sink: float = 0.0

    CO2_generation_rate: float = 0.0

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
    # PREHEATER STATES (5 CELL) (KELVIN)
    # ======================================================
    Tg_preheater: np.ndarray = field(default_factory=lambda: np.ones(5) * 573.15)

    Ts_preheater: np.ndarray = field(default_factory=lambda: np.ones(5) * 573.15)

    Tw_preheater: np.ndarray = field(default_factory=lambda: np.ones(5) * 523.15)
    
    # ======================================================
    # TRANSITION STATES (5 CELL) (KELVIN)
    # ======================================================
    Tg_transition: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1650.0
    )

    Ts_transition: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 1400.0
    )

    Tw_transition: np.ndarray = field(
        default_factory=lambda: np.ones(5) * 800.0
    )

    Hgas_transition_out: float = 0.0

    Transition_Q_sink: float = 0.0

    # ======================================================
    # CALCINER STATES (5 CELL) (KELVIN)
    # ======================================================
    Tg_calcination: np.ndarray = field(default_factory=lambda: np.ones(5) * (1473.15))

    Ts_calcination: np.ndarray = field(default_factory=lambda: np.ones(5) * (1223.15))

    Tw_calcination: np.ndarray = field(default_factory=lambda: np.ones(5) * (773.15))

    # ======================================================
    # BURNING STATES (5 CELL) (KELVIN)
    # ======================================================
    Tg_burning: np.ndarray = field(default_factory=lambda: np.ones(5) * (1773.15))

    Ts_burning: np.ndarray = field(default_factory=lambda: np.ones(5) * (1673.15))

    Tw_burning: np.ndarray = field(default_factory=lambda: np.ones(5) * (873.15))

    # ======================================================
    # COOLER STATES (5 CELL) (KELVIN)
    # ======================================================
    Tg_cooler: np.ndarray = field(default_factory=lambda: np.ones(5) * (423.15))

    Ts_cooler: np.ndarray = field(default_factory=lambda: np.ones(5) * (393.15))

    Tw_cooler: np.ndarray = field(default_factory=lambda: np.ones(5) * (353.15))

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
