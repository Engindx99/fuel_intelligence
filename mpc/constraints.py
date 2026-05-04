# mpc/constraints.py

"""
Constraints definition for Model Predictive Control.

Defines:
- State constraints (Hard/Soft safety limits)
- Control constraints (Actuator saturation)
- Rate-of-change constraints (Slew rate limits)

NO:
- Direct optimization logic (handled in solver)
- Physical constants (imported from configs)
"""

import numpy as np
from core.state import *

# -------------------------------------------------
# 1. STATE CONSTRAINTS (x_min, x_max)
# -------------------------------------------------

def get_state_constraints():
    """
    Returns the lower and upper bounds for the state vector.
    Prevents unrealistic temperatures or species concentrations.
    """
    x_min = np.zeros(N_STATES)
    x_max = np.inf * np.ones(N_STATES)

    # Temperature Limits (Kelvin)
    x_min[IDX_T_S] = 300.0    # Solid minimum (Ambient)
   # x_max[IDX_T_S] = 1800.0   # Artırıldı: Klinkerleşme için daha yüksek pay
    
    x_min[IDX_T_G] = 500.0    # Gas minimum
    #x_max[IDX_T_G] = 2600.0   # Artırıldı: Yanma bölgesi esnekliği

    # Species Limits (Normalized concentration 0.0 - 1.0)
    x_max[IDX_CaCO3 : IDX_C4AF + 1] = 1.0
    x_max[IDX_CO2 : IDX_O2 + 1] = 1.0

    # Structural Limits
    x_min[IDX_PHI] = 0.05     # Min filling (Empty kiln safety)
    x_max[IDX_PHI] = 0.25     # Max filling (Overload protection)
    
    x_min[IDX_EPSILON] = 0.2
    x_max[IDX_EPSILON] = 0.8

    return x_min, x_max


# -------------------------------------------------
# 2. CONTROL CONSTRAINTS (u_min, u_max)
# -------------------------------------------------

def get_control_constraints():
    """
    Returns the saturation limits for the actuators.
    Based on physical equipment specs (Fan power, motor torque).
    """
    u_min = np.zeros(N_CONTROLS)
    u_max = np.zeros(N_CONTROLS)

    # Fuel Rate (kg/s)
    u_min[IDX_FUEL] = 0.5
    u_max[IDX_FUEL] = 15.0

    # Fan RPM
    u_min[IDX_FAN] = 300.0
    u_max[IDX_FAN] = 6000.0

    # Feed Rate (kg/s)
    u_min[IDX_FEED] = 10.0
    u_max[IDX_FEED] = 80.0

    # Reactor (Kiln) RPM
    u_min[IDX_REACTOR] = 0.5
    u_max[IDX_REACTOR] = 5.0

    return u_min, u_max


# -------------------------------------------------
# 3. SLEW RATE CONSTRAINTS (du_max)
# -------------------------------------------------

def get_slew_rate_constraints():
    """
    Limits how fast an actuator can move in one control step (dt_mpc).
    Essential for preventing mechanical wear and thermal shock.
    """
    # Values per second (multiplied by dt in solver)
    du_max = np.zeros(N_CONTROLS)

    du_max[IDX_FUEL]    = 0.2   # kg/s^2
    du_max[IDX_FAN]     = 50.0  # RPM/s
    du_max[IDX_FEED]    = 1.0   # kg/s^2
    du_max[IDX_REACTOR] = 0.1   # RPM/s

    return du_max