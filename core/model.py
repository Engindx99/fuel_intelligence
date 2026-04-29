# model.py

"""
Full rotary kiln PDE system assembly.

Combines:
- state definition
- physics (reaction, energy, mass)
- flow (velocity, structure)

Outputs structured PDE form:

dx/dt + v * dx/dz = source_terms

NO:
- discretization
- numerical solving
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

    Returns structured PDE system
    """

    # -----------------------------
    # 1. FLOW FIELD
    # -----------------------------
    v = compute_velocity(x, u)

    # structural dynamics
    d_epsilon = compute_porosity(x, u)
    d_phi     = compute_solid_fraction(x, u)

    # -----------------------------
    # 2. PHYSICS TERMS
    # -----------------------------
    R_terms = compute_reaction_rates(x)
    E_terms = energy_terms(x, u)
    M_terms = mass_terms(x, u)

    # -----------------------------
    # 3. BUILD dx/dt
    # -----------------------------
    dx_dt = zeros_like(x)

    # --- ENERGY STATES ---
    dx_dt[IDX_T_S] = (
        - v * dx_dz[IDX_T_S]
        + E_terms["solid_energy"]
    )

    dx_dt[IDX_T_G] = (
        - v * dx_dz[IDX_T_G]
        + E_terms["gas_energy"]
    )

    # --- SOLID SPECIES ---
    dx_dt[IDX_CaCO3] = (
        - v * dx_dz[IDX_CaCO3]
        + M_terms["CaCO3"]
    )

    dx_dt[IDX_CaO] = (
        - v * dx_dz[IDX_CaO]
        + M_terms["CaO"]
    )

    dx_dt[IDX_C2S] = (
        - v * dx_dz[IDX_C2S]
        + M_terms["C2S"]
    )

    dx_dt[IDX_C3S] = (
        - v * dx_dz[IDX_C3S]
        + M_terms["C3S"]
    )

    dx_dt[IDX_C3A] = (
        - v * dx_dz[IDX_C3A]
        + M_terms["C3A"]
    )

    dx_dt[IDX_C4AF] = (
        - v * dx_dz[IDX_C4AF]
        + M_terms["C4AF"]
    )

    # --- GAS SPECIES ---
    dx_dt[IDX_CO2] = (
        - v * dx_dz[IDX_CO2]
        + M_terms["CO2"]
    )

    dx_dt[IDX_O2] = (
        - v * dx_dz[IDX_O2]
        + M_terms["O2"]
    )

    # --- STRUCTURAL STATES ---
    dx_dt[IDX_PHI] = d_phi
    dx_dt[IDX_EPSILON] = d_epsilon

    # -----------------------------
    # 4. OUTPUT STRUCTURE
    # -----------------------------
    return {
        "dx_dt": dx_dt,
        "velocity": v,
        "terms": {
            "reaction": R_terms,
            "energy": E_terms,
            "mass": M_terms
        }
    }


# -------------------------------------------------
# UTIL PLACEHOLDER
# -------------------------------------------------

def zeros_like(x):
    return [0 for _ in x]