import os
import yaml
import numpy as np
from core.state import StateIdx, IDX_FAN, IDX_REACTOR, IDX_FEED

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = os.path.join(_ROOT, "configs", "model_config.yaml")
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _flow_cfg = yaml.safe_load(f) or {}

_g = _flow_cfg.get("kiln_geometry") or {}
_t = _flow_cfg.get("thermal") or {}
_FLOW_D = float(_g.get("diameter", 4.5))
_FLOW_RHO_G = float(_t.get("rho_gas", 1.2))
_MDOT_G_BASE = float(_t.get("mdot_gas_base", 20.0))
_MDOT_G_PER_FAN = float(_t.get("mdot_gas_per_fan", 0.05))
_CROSS_A = np.pi * (_FLOW_D / 2.0) ** 2


def v_gas_model(fan_rpm, epsilon):
    """
    Interstitial axial gas speed [m/s], counter-current (negative toward z decreasing).

    Superficial: u_sup = dot(m)_g / (rho_g A). Interstitial divides by epsilon.
    Gives ~few–few×10 seconds gas residence through a 60 m kiln with plant-scale mdot,
    replacing the empirical k*fan/eps surrogate that overstated tau by orders.
    """
    eps = np.maximum(np.asarray(epsilon, dtype=np.float64), 0.05)
    mdot = _MDOT_G_BASE + _MDOT_G_PER_FAN * float(fan_rpm)
    v_sup = mdot / (_FLOW_RHO_G * _CROSS_A + 1e-18)
    v_int = -(v_sup / eps)
    return np.clip(v_int, -120.0, -0.3)


def v_solid_model(reactor_rpm, feed_rate, epsilon):
    """
    Düzeltilmiş katı faz eksenel hız modeli (granular flow surrogate).
    """

    R = _FLOW_D / 2.0

    omega = 2.0 * np.pi * reactor_rpm / 60.0

    v_t = omega * R

    theta = float(_g.get("slope", 0.03))
    slope_effect = np.tan(theta)

    eps = np.clip(np.asarray(epsilon, dtype=np.float64), 0.05, 0.95)
    packing = eps / (eps + 0.3)

    load_effect = np.exp(-0.0005 * feed_rate)

    slip_factor = np.tanh(v_t * slope_effect)

    v_s = v_t * slope_effect * slip_factor * packing * load_effect

    return np.clip(v_s, 0.0, 0.25)


def compute_velocities(x, u):
    """
    State + control → phase velocities (scalars matching engine RTD midpoint usage).
    """
    fan = u[IDX_FAN]
    reactor = u[IDX_REACTOR]
    feed = u[IDX_FEED]
    epsilon = x[StateIdx.EPSILON]

    v_g_raw = v_gas_model(fan, epsilon)
    v_g = float(np.asarray(v_g_raw, dtype=np.float64).ravel()[0])

    v_s_raw = v_solid_model(reactor, feed, epsilon)
    v_s = float(np.asarray(v_s_raw, dtype=np.float64).ravel()[0])

    return v_s, v_g


def compute_porosity(x, u):
    """
    CaCO3 → CaO dönüşümüne bağlı gözeneklilik değişimi.
    """
    CaCO3 = x[StateIdx.CaCO3]
    CaO = x[StateIdx.CaO]

    k = 0.05

    return k * np.maximum(CaCO3, 0.0) * (1.0 - np.maximum(CaO, 0.0))
