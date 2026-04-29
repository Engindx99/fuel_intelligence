# physics.py

"""
Symbolic physics module for rotary kiln digital twin.

Includes:
- Reaction kinetics (symbolic)
- Energy balance terms
- Mass balance source terms

NO:
- numerical constants
- discretization
- velocity definition (handled in flow.py)
"""

from core.state import *


# -------------------------------------------------
# 1. REACTION KINETICS
# -------------------------------------------------

def compute_reaction_rates(x):
    """
    Returns reaction rates R for all species.

    All rates are symbolic functions of:
    - T_s
    - concentrations
    """

    T_s = x[IDX_T_S]

    C_CaCO3 = x[IDX_CaCO3]
    C_CaO   = x[IDX_CaO]

    C_C2S  = x[IDX_C2S]
    C_C3S  = x[IDX_C3S]
    C_C3A  = x[IDX_C3A]
    C_C4AF = x[IDX_C4AF]

    # --- Primary calcination reaction ---
    # CaCO3 → CaO + CO2
    r_calcination = k_calcination(T_s) * C_CaCO3

    # --- Clinker formation reactions (symbolic placeholders) ---
    r_C2S  = k_C2S(T_s)  * C_CaO
    r_C3S  = k_C3S(T_s)  * C_CaO
    r_C3A  = k_C3A(T_s)  * C_CaO
    r_C4AF = k_C4AF(T_s) * C_CaO

    return {
        "r_calcination": r_calcination,
        "r_C2S": r_C2S,
        "r_C3S": r_C3S,
        "r_C3A": r_C3A,
        "r_C4AF": r_C4AF
    }


# -------------------------------------------------
# 2. ENERGY TERMS
# -------------------------------------------------

def energy_terms(x, u):
    """
    Returns symbolic energy contributions for:
    - solid temperature (T_s)
    - gas temperature (T_g)
    """

    T_s = x[IDX_T_S]
    T_g = x[IDX_T_G]

    fuel_rate = u[IDX_FUEL]

    R = compute_reaction_rates(x)

    # --- Reaction heat source (symbolic) ---
    Q_reaction = (
        H_calcination() * R["r_calcination"] +
        H_C2S()  * R["r_C2S"] +
        H_C3S()  * R["r_C3S"] +
        H_C3A()  * R["r_C3A"] +
        H_C4AF() * R["r_C4AF"]
    )

    # --- Gas-solid heat exchange ---
    Q_exchange = h_gs(x) * (T_g - T_s)

    # --- Fuel heat input (affects gas phase) ---
    Q_fuel = Q_fuel_source(fuel_rate)

    return {
        "solid_energy": Q_reaction + Q_exchange,
        "gas_energy": Q_fuel - Q_exchange
    }


# -------------------------------------------------
# 3. MASS BALANCE TERMS
# -------------------------------------------------

def mass_terms(x, u):
    """
    Returns source terms for all species.

    Transport (advection) is NOT included here.
    Only reaction contributions.
    """

    R = compute_reaction_rates(x)

    # --- Solid species ---
    dC_CaCO3 = -R["r_calcination"]
    dC_CaO   = (
        + R["r_calcination"]
        - R["r_C2S"]
        - R["r_C3S"]
        - R["r_C3A"]
        - R["r_C4AF"]
    )

    dC_C2S  = +R["r_C2S"]
    dC_C3S  = +R["r_C3S"]
    dC_C3A  = +R["r_C3A"]
    dC_C4AF = +R["r_C4AF"]

    # --- Gas species ---
    dCO2 = +R["r_calcination"]
    dO2  = -O2_consumption(u)  # linked to combustion

    return {
        "CaCO3": dC_CaCO3,
        "CaO": dC_CaO,
        "C2S": dC_C2S,
        "C3S": dC_C3S,
        "C3A": dC_C3A,
        "C4AF": dC_C4AF,
        "CO2": dCO2,
        "O2": dO2
    }


# -------------------------------------------------
# 4. SYMBOLIC PLACEHOLDERS (INTENTIONALLY EMPTY)
# -------------------------------------------------

def k_calcination(T): pass
def k_C2S(T): pass
def k_C3S(T): pass
def k_C3A(T): pass
def k_C4AF(T): pass

def H_calcination(): pass
def H_C2S(): pass
def H_C3S(): pass
def H_C3A(): pass
def H_C4AF(): pass

def h_gs(x): pass
def Q_fuel_source(fuel_rate): pass

def O2_consumption(u): pass