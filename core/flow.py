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
_FEED_NOMINAL_KG_S = float(_t.get("feed_rate_nominal_kg_s", 10.0))
_K_RPM_AX = float(_t.get("solid_axial_rpm_coeff", 0.0105))
_K_SLOPE_AX = float(_t.get("solid_axial_slope_coeff", 0.07))
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
    Katı eksenel hız [m/s] — dönüş (RPM), tambur çapı, packing ve besleme ile ölçeklenir.

    Eski `tanh(v_t * slope)` RPM'i fiilen sıfıra yakın bir faktörde bastırıyordu; kontrol çıktıları
    anlamlı değişmiyordu. Yüksek RPM → daha büyük v_s → daha kısa rezidans → eksende daha az ısınma.
    """

    R = _FLOW_D / 2.0
    rpm = float(max(reactor_rpm, 0.05))
    omega = 2.0 * np.pi * rpm / 60.0
    v_t = omega * R

    theta = float(_g.get("slope", 0.03))
    slope_effect = float(np.tan(theta))

    eps = np.clip(np.asarray(epsilon, dtype=np.float64), 0.05, 0.95)
    packing = eps / (eps + 0.3)

    feed = float(max(feed_rate, 1e-6))
    load_factor = feed / (_FEED_NOMINAL_KG_S + 1e-9)
    load_factor = float(np.clip(load_factor, 0.15, 4.0))

    # Ana terim: RPM ve çap ile doğrudan ölçek (yatak dönüşü → eksenel taşınım sürrogatı).
    v_rotate = _K_RPM_AX * rpm * _FLOW_D * packing * load_factor
    # Eğim + teğet hızdan küçük ek (düşük RPM'de bile sıfırlanmasın diye tanh(1) sabit eğim akışı).
    v_slope = v_t * slope_effect * packing * load_factor * _K_SLOPE_AX * 0.7615941559557649  # tanh(1)

    v_s = v_rotate + v_slope
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
