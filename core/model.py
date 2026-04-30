"""
Full rotary kiln PDE system assembly.

dx/dt + v * dx/dz = source_terms
"""

from core.state import *
from core.physics import *
from core.flow import *


# -------------------------------------------------
# MAIN SYSTEM
# -------------------------------------------------

def kiln_pde_system(x, u, dx_dz):
    """
    x     : state vector
    u     : control vector
    dx_dz : spatial derivative ∂x/∂z
    """

    # -----------------------------
    # 1. FLOW FIELD
    # -----------------------------
    v_s, v_g = compute_velocities(x, u)

    # structural dynamics
    d_epsilon = compute_porosity(x, u)
    d_phi     = phi_feed_effect(u[IDX_FEED])  # FIX: was missing correct coupling

    # -----------------------------
    # 2. PHYSICS TERMS
    # -----------------------------
    R_terms = compute_reaction_rates(x)
    E_terms = energy_terms(x, u)
    M_terms = mass_terms(x, u)

    # -----------------------------
    # 3. INIT STATE VECTOR
    # -----------------------------
    dx_dt = zeros_like(x)

    # =============================
    # ENERGY
    # =============================
    dx_dt[IDX_T_S] = (
        - v_s * dx_dz[IDX_T_S]
        + E_terms["solid_energy"]
    )

    dx_dt[IDX_T_G] = (
        - v_g * dx_dz[IDX_T_G]
        + E_terms["gas_energy"]
    )

    # =============================
    # SOLID SPECIES
    # =============================
    dx_dt[IDX_CaCO3] = - v_s * dx_dz[IDX_CaCO3] + M_terms["CaCO3"]
    dx_dt[IDX_CaO]   = - v_s * dx_dz[IDX_CaO]   + M_terms["CaO"]
    dx_dt[IDX_C2S]   = - v_s * dx_dz[IDX_C2S]   + M_terms["C2S"]
    dx_dt[IDX_C3S]   = - v_s * dx_dz[IDX_C3S]   + M_terms["C3S"]
    dx_dt[IDX_C3A]   = - v_s * dx_dz[IDX_C3A]   + M_terms["C3A"]
    dx_dt[IDX_C4AF]  = - v_s * dx_dz[IDX_C4AF]  + M_terms["C4AF"]

    dx_dt[IDX_SiO2]  = - v_s * dx_dz[IDX_SiO2]  + M_terms["SiO2"]
    dx_dt[IDX_Al2O3] = - v_s * dx_dz[IDX_Al2O3] + M_terms["Al2O3"]
    dx_dt[IDX_Fe2O3] = - v_s * dx_dz[IDX_Fe2O3] + M_terms["Fe2O3"]

    # =============================
    # GAS SPECIES
    # =============================
    dx_dt[IDX_CO2] = - v_g * dx_dz[IDX_CO2] + M_terms["CO2"]
    dx_dt[IDX_O2]  = - v_g * dx_dz[IDX_O2]  + M_terms["O2"]

    # =============================
    # STRUCTURE
    # =============================
    dx_dt[IDX_PHI]     = d_phi
    dx_dt[IDX_EPSILON] = d_epsilon + M_terms["epsilon"]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    return {
        "dx_dt": dx_dt,
        "velocities": {
            "solid": v_s,
            "gas": v_g
        },
        "terms": {
            "reaction": R_terms,
            "energy": E_terms,
            "mass": M_terms
        }
    }


# -------------------------------------------------
# UTILITY
# -------------------------------------------------

def zeros_like(x):
    return [0.0 for _ in x]