import logging
import numpy as np

from core.kinetics import compute_reaction_rates_vec
from core.physics import mass_terms_vec
from core.state import *


logger = logging.getLogger("PhysicsValidation")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)


C3S_MIN_T_K = 1200.0


def validate_state_vec(X, dCdt, R, T_s, dt=0.04, strict=False):

    T_s = np.asarray(T_s)
    violations = []

    if np.any((T_s < C3S_MIN_T_K) & (R["r_C3S"] > 1e-6)):
        msg = "C3S formation below physical threshold (expect r_C3S=0 if T < 1200 K)"
        violations.append(msg)
        logger.warning(msg)

    M = {
        IDX_CaCO3: 0.10009,
        IDX_CaO: 0.05608,
        IDX_C2S: 0.17224,
        IDX_C3S: 0.17234,
        IDX_C3A: 0.27000,
        IDX_C4AF: 0.48600,
        IDX_SiO2: 0.06008,
        IDX_Al2O3: 0.10200,
        IDX_Fe2O3: 0.16000,
        IDX_CO2: 0.04401,
    }

    mass_before = np.zeros(X.shape[0])
    mass_after = np.zeros(X.shape[0])

    for idx, mw in M.items():
        mass_before += X[:, idx] * mw
        mass_after += (X[:, idx] + dCdt[:, idx] * dt) * mw

    rel_err = np.max(
        np.abs(mass_after - mass_before)
        / (mass_before + 1e-12)
    )

    if rel_err > 1e-3:
        msg = f"Mass conservation violation (approx kg basis; rel error={rel_err:.2e})"
        violations.append(msg)
        logger.warning(msg)

    if np.any(~np.isfinite(X)):
        msg = "NaN/Inf detected in state"
        violations.append(msg)
        logger.error(msg)
        if strict:
            raise RuntimeError(msg)

    if strict and len(violations) > 0:
        raise RuntimeError(f"Physics violations: {violations}")

    return len(violations) == 0


def validate_state(X, step=None, u=None):
    R = compute_reaction_rates_vec(X)

    if u is None:
        u = np.zeros(N_CONTROLS, dtype=float)

    dCdt = mass_terms_vec(X, u)

    T_s = np.maximum(X[:, IDX_T_S], 300.0)
    return validate_state_vec(X, dCdt, R, T_s)
