"""
Full rotary kiln PDE system assembly (Vectorized & 18-State Compatible).
Equation: ∂x/∂t + v * ∂x/∂z = source_terms
"""

import numpy as np
from core.state import *
from core.physics import energy_terms_vec, mass_terms_vec
from core.kinetics import compute_reaction_rates
from core.flow import compute_velocities, compute_porosity


# -------------------------------------------------
# MAIN SYSTEM
# -------------------------------------------------

def kiln_pde_system(x, u, dx_dz):
    """
    x     : State vector — 1D (N_STATES,) veya 2D (N_cells, N_STATES).
    dx_dz : Mekansal türev ile aynı şekil.
    """

    x = np.asarray(x, dtype=np.float64)
    dx_dz_a = np.asarray(dx_dz, dtype=np.float64)
    singleton = x.ndim == 1
    if singleton:
        x = x.reshape(1, -1)
        dx_dz_a = dx_dz_a.reshape(1, -1)
    xc = x[0] if x.shape[0] == 1 else x[x.shape[0] // 2]

    v_s, v_g = compute_velocities(xc, u)

    d_epsilon = compute_porosity(xc, u)
    d_phi = 0.01 * u[IDX_FEED]

    dE_s, dE_g = energy_terms_vec(x, u, t=None)
    M_mat = mass_terms_vec(x, u)
    R_full = compute_reaction_rates(x)
    reaction_vec = np.asarray(R_full[0], dtype=np.float64) if singleton else np.mean(R_full, axis=0)

    dx_dt = np.zeros_like(x)

    dx_dt[:, IDX_T_S] = -v_s * dx_dz_a[:, IDX_T_S] + dE_s
    dx_dt[:, IDX_T_G] = -v_g * dx_dz_a[:, IDX_T_G] + dE_g

    for idx in SOLID_SPECIES:
        dx_dt[:, idx] = -v_s * dx_dz_a[:, idx] + M_mat[:, idx]

    for idx in GAS_SPECIES:
        dx_dt[:, idx] = -v_g * dx_dz_a[:, idx] + M_mat[:, idx]

    dx_dt[:, IDX_PHI] = d_phi
    deps_col = np.full((x.shape[0],), float(d_epsilon), dtype=np.float64)
    if M_mat.ndim == 2:
        deps_col = deps_col + M_mat[:, IDX_EPSILON]
    dx_dt[:, IDX_EPSILON] = deps_col

    out_dx_dt = dx_dt[0].copy() if singleton else dx_dt
    mass_dict = {StateIdx(ix).name: float(M_mat[0, ix]) for ix in range(N_STATES)} if singleton else {}

    return {
        "dx_dt": out_dx_dt,
        "velocities": {"solid": v_s, "gas": v_g},
        "terms": {
            "reaction": reaction_vec,
            "energy": {"solid_energy": float(dE_s[0]), "gas_energy": float(dE_g[0])}
            if singleton
            else {"solid_energy": dE_s, "gas_energy": dE_g},
            "mass": mass_dict if singleton else M_mat,
        },
    }


def zeros_like(x):
    """Geriye dönük uyumluluk için, ancak dahili olarak NumPy kullanır."""
    return np.zeros_like(x)
