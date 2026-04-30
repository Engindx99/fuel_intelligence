import logging
import numpy as np

from core.kinetics import compute_reaction_rates_vec
from core.physics import mass_terms_vec
from core.state import *

# -----------------------------
# LOGGER SETUP (ROBUST FIX)
# -----------------------------
logger = logging.getLogger("PhysicsValidation")

# 🔥 kritik: eski handlerları temizle (IDE / notebook bug fix)
logger.handlers.clear()
logger.propagate = False

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

# set level
logger.setLevel(logging.WARNING)

# -----------------------------
# CONSTANTS
# -----------------------------
C3S_MIN_T_K = 1200.0

PERF_GOOD = 3e-3
WARN = 7e-3
FAIL = 2e-2


def validate_state_vec(
    X,
    dCdt,
    R,
    T_s,
    dt=0.04,
    strict=False,
    closed_system=False
):
    T_s = np.asarray(T_s)
    violations = []

    # -----------------------------
    # 1. Physical constraint
    # -----------------------------
    if np.any((T_s < C3S_MIN_T_K) & (R["r_C3S"] > 1e-6)):
        msg = "C3S formation below physical threshold (T < 1200K)"
        violations.append(msg)
        logger.warning(msg)

    # -----------------------------
    # 2. Molecular weights
    # -----------------------------
    mw = np.zeros(X.shape[1], dtype=float)

    mw[IDX_CaCO3] = 0.10009
    mw[IDX_CaO]   = 0.05608
    mw[IDX_SiO2]  = 0.06008
    mw[IDX_C2S]   = 0.17224
    mw[IDX_C3S]   = 0.22832
    mw[IDX_CO2]   = 0.04401
    mw[IDX_C3A]   = 0.27000
    mw[IDX_C4AF]  = 0.48600
    mw[IDX_Al2O3] = 0.10200
    mw[IDX_Fe2O3] = 0.16000

    # -----------------------------
    # 3. Mass conservation residual
    # -----------------------------
    mass_rate = dCdt * mw
    mass_residual = np.sum(mass_rate, axis=1)

    total_mass = np.sum(X * mw, axis=1) + 1e-12

    rel_err = np.max(np.abs(mass_residual * dt) / total_mass)

    # -----------------------------
    # 4. Closed system check
    # -----------------------------
    if closed_system:
        net = np.sum(mass_residual)
        if np.abs(net) > 1e-6:
            msg = f"Closed system mass imbalance: {net:.3e}"
            violations.append(msg)
            logger.warning(msg)

    # -----------------------------
    # 5. PERFORMANCE LEVEL (GOOD)
    # -----------------------------
    if rel_err < PERF_GOOD:
        logger.debug(f"Mass conservation good (rel residual={rel_err:.2e})")

    # -----------------------------
    # 6. WARNING LEVEL
    # -----------------------------
    if rel_err > WARN:
        msg = f"Mass drift warning (rel residual={rel_err:.2e})"
        violations.append(msg)
        logger.warning(msg)

    # -----------------------------
    # 7. FAILURE LEVEL
    # -----------------------------
    if rel_err > FAIL:
        msg = f"Mass conservation FAILURE (rel residual={rel_err:.2e})"
        violations.append(msg)
        logger.error(msg)

        if strict:
            raise RuntimeError(msg)

    # -----------------------------
    # 8. Numerical stability
    # -----------------------------
    if not np.all(np.isfinite(X)):
        msg = "NaN/Inf detected in state"
        violations.append(msg)
        logger.error(msg)

        if strict:
            raise RuntimeError(msg)

    # -----------------------------
    # 9. STRICT MODE
    # -----------------------------
    if strict and violations:
        raise RuntimeError(f"Physics violations: {violations}")

    return len(violations) == 0


def validate_state(X, step=None, u=None):
    R = compute_reaction_rates_vec(X)

    if u is None:
        u = np.zeros(N_CONTROLS, dtype=float)

    dCdt = mass_terms_vec(X, u)

    T_s = np.maximum(X[:, IDX_T_S], 300.0)

    return validate_state_vec(X, dCdt, R, T_s)