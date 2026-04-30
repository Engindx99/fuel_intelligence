import numpy as np
from core.state import IDX_FAN, IDX_REACTOR, IDX_FEED, IDX_EPSILON


# -----------------------------
# GAS FLOW
# -----------------------------
def v_gas_model(fan_rpm, epsilon):

    k = 0.001

    eps = max(epsilon, 0.05)

    # pressure-driven + porosity resistance
    return -(k * fan_rpm) / eps


# -----------------------------
# SOLID FLOW
# -----------------------------
def v_solid_model(reactor_rpm, feed_rate, epsilon):

    slope = 0.03
    D = 4.5
    k = 0.02

    # load damping
    load = 1.0 / (1.0 + 0.001 * feed_rate)

    # packing resistance (nonlinear stabilized)
    eps = np.clip(epsilon, 0.0, 0.9)
    packing = (1.0 - eps)**1.5

    return k * D * slope * reactor_rpm * load * packing


# -----------------------------
# VELOCITY FIELD
# -----------------------------
def compute_velocities(x, u):

    fan = u[IDX_FAN]
    reactor = u[IDX_REACTOR]
    feed = u[IDX_FEED]

    epsilon = x[IDX_EPSILON]

    v_g = v_gas_model(fan, epsilon)
    v_s = v_solid_model(reactor, feed, epsilon)

    return v_s, v_g


# -----------------------------
# POROSITY DYNAMICS
# -----------------------------
def compute_porosity(x, u):

    CaCO3 = x[2]   # IDX_CaCO3
    CaO = x[3]     # IDX_CaO

    # reaction-driven expansion
    k = 0.05

    return k * max(CaCO3, 0.0) * (1.0 - max(CaO, 0.0))