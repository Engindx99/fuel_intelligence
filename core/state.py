"""
State and control definition for rotary kiln model.

This file is the SINGLE SOURCE OF TRUTH.
Do NOT modify variable names in other modules.
"""

from enum import IntEnum


# -------------------------------------------------
# 1. STATE VECTOR
# -------------------------------------------------

class StateIdx(IntEnum):
    # Thermal states
    T_S = 0
    T_G = 1

    # Major solid species
    CaCO3 = 2
    CaO   = 3

    # Clinker phases
    C2S  = 4
    C3S  = 5
    C3A  = 6
    C4AF = 7

    # Gas species (extended)
    CO2 = 8
    O2  = 9
    CO  = 10
    H2O = 11
    N2  = 12

    # Structural states
    PHI      = 13   # bed packing / porosity (to be defined physically)
    EPSILON  = 14   # void fraction / holdup

    # Oxides (feed composition tracking)
    SiO2  = 15
    Al2O3 = 16
    Fe2O3 = 17


# total number of states (SAFE)
N_STATES = len(StateIdx)


# -------------------------------------------------
# 2. CONTROL VECTOR
# -------------------------------------------------

class ControlIdx(IntEnum):
    FUEL    = 0
    FAN     = 1
    FEED    = 2
    REACTOR = 3


N_CONTROLS = len(ControlIdx)


# Legacy/int aliases (same integers as enums; used by main, mpc, tests)
IDX_T_S = StateIdx.T_S
IDX_T_G = StateIdx.T_G
IDX_CaCO3 = StateIdx.CaCO3
IDX_CaO = StateIdx.CaO
IDX_C2S = StateIdx.C2S
IDX_C3S = StateIdx.C3S
IDX_C3A = StateIdx.C3A
IDX_C4AF = StateIdx.C4AF
IDX_CO2 = StateIdx.CO2
IDX_O2 = StateIdx.O2
IDX_CO = StateIdx.CO
IDX_H2O = StateIdx.H2O
IDX_N2 = StateIdx.N2
IDX_PHI = StateIdx.PHI
IDX_EPSILON = StateIdx.EPSILON
IDX_SiO2 = StateIdx.SiO2
IDX_Al2O3 = StateIdx.Al2O3
IDX_Fe2O3 = StateIdx.Fe2O3

IDX_FUEL = ControlIdx.FUEL
IDX_FAN = ControlIdx.FAN
IDX_FEED = ControlIdx.FEED
IDX_REACTOR = ControlIdx.REACTOR


# -------------------------------------------------
# 3. GROUPINGS
# -------------------------------------------------

THERMAL_STATES = [
    StateIdx.T_S,
    StateIdx.T_G
]

SOLID_SPECIES = [
    StateIdx.CaCO3,
    StateIdx.CaO,
    StateIdx.C2S,
    StateIdx.C3S,
    StateIdx.C3A,
    StateIdx.C4AF,
    StateIdx.SiO2,
    StateIdx.Al2O3,
    StateIdx.Fe2O3
]

GAS_SPECIES = [
    StateIdx.CO2,
    StateIdx.O2,
    StateIdx.CO,
    StateIdx.H2O,
    StateIdx.N2
]

STRUCTURAL_STATES = [
    StateIdx.PHI,
    StateIdx.EPSILON
]


# -------------------------------------------------
# 4. STATE NAMES (LOGGING / DEBUG)
# -------------------------------------------------

STATE_NAMES = [s.name for s in StateIdx]
CONTROL_NAMES = [c.name.lower() for c in ControlIdx]


# -------------------------------------------------
# 5. HELPERS
# -------------------------------------------------

def create_zero_state():
    return [0.0] * N_STATES


def create_zero_control():
    return [0.0] * N_CONTROLS


def state_vector_to_dict(x):
    """Debug için okunabilir mapping"""
    return {StateIdx(i).name: x[i] for i in range(N_STATES)}