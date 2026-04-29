# state.py

"""
State and control definition for rotary kiln model.

This file is the SINGLE SOURCE OF TRUTH.
Do NOT modify variable names in other modules.
"""


# -------------------------------------------------
# 1. STATE VECTOR
# -------------------------------------------------

# x = [
#     T_s, T_g,
#     C_CaCO3, C_CaO,
#     C_C2S, C_C3S, C_C3A, C_C4AF,
#     CO2, O2,
#     phi, epsilon_l
# ]

IDX_T_S = 0
IDX_T_G = 1

IDX_CaCO3 = 2
IDX_CaO   = 3

IDX_C2S  = 4
IDX_C3S  = 5
IDX_C3A  = 6
IDX_C4AF = 7

IDX_CO2 = 8
IDX_O2  = 9

IDX_PHI     = 10
IDX_EPSILON = 11


# total number of states
N_STATES = 12


# -------------------------------------------------
# 2. CONTROL VECTOR
# -------------------------------------------------

# u = [
#     fuel_rate,
#     fan_rpm,
#     feed_rate,
#     reactor_rpm
# ]

IDX_FUEL    = 0
IDX_FAN     = 1
IDX_FEED    = 2
IDX_REACTOR = 3

N_CONTROLS = 4


# -------------------------------------------------
# 3. GROUPINGS (VERY IMPORTANT)
# -------------------------------------------------

THERMAL_STATES = [IDX_T_S, IDX_T_G]

SOLID_SPECIES = [
    IDX_CaCO3,
    IDX_CaO,
    IDX_C2S,
    IDX_C3S,
    IDX_C3A,
    IDX_C4AF
]

GAS_SPECIES = [
    IDX_CO2,
    IDX_O2
]

STRUCTURAL_STATES = [
    IDX_PHI,
    IDX_EPSILON
]


# -------------------------------------------------
# 4. STATE NAMES (DEBUG / LOGGING)
# -------------------------------------------------

STATE_NAMES = [
    "T_s", "T_g",
    "C_CaCO3", "C_CaO",
    "C_C2S", "C_C3S", "C_C3A", "C_C4AF",
    "CO2", "O2",
    "phi", "epsilon_l"
]

CONTROL_NAMES = [
    "fuel_rate",
    "fan_rpm",
    "feed_rate",
    "reactor_rpm"
]


# -------------------------------------------------
# 5. OPTIONAL HELPERS
# -------------------------------------------------

def create_zero_state():
    return [0.0] * N_STATES


def create_zero_control():
    return [0.0] * N_CONTROLS