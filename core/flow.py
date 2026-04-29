# flow.py

"""
Flow and structural dynamics module for rotary kiln.

Includes:
- axial velocity field
- porosity evolution (epsilon_l)
- solid fraction evolution (phi)

NO:
- discretization
- numerical constants
- direct reaction implementation (use concentrations only)
"""

from core.state import *


# -------------------------------------------------
# 1. AXIAL VELOCITY
# -------------------------------------------------

def compute_velocity(x, u):
    """
    Symbolic axial velocity field v(z,t).

    Depends on:
    - fan_rpm (gas driving force)
    - feed_rate (solid loading)
    - porosity (epsilon_l)
    """

    fan_rpm   = u[IDX_FAN]
    feed_rate = u[IDX_FEED]

    epsilon_l = x[IDX_EPSILON]

    # Gas-driven contribution
    v_gas = v_gas_model(fan_rpm, epsilon_l)

    # Solid loading resistance
    v_solid = v_solid_model(feed_rate, epsilon_l)

    # Combined effective velocity
    v = v_gas - v_solid

    return v


# -------------------------------------------------
# 2. POROSITY EVOLUTION (epsilon_l)
# -------------------------------------------------

def compute_porosity(x, u):
    """
    Porosity evolution (epsilon_l).

    Influenced by:
    - reaction progress (conversion)
    - reactor rotation (mixing)
    """

    reactor_rpm = u[IDX_REACTOR]

    C_CaCO3 = x[IDX_CaCO3]
    C_CaO   = x[IDX_CaO]

    # Reaction-induced structural change
    eps_reaction = porosity_reaction_effect(C_CaCO3, C_CaO)

    # Mixing / agitation effect
    eps_mixing = porosity_mixing_effect(reactor_rpm)

    d_epsilon = eps_reaction + eps_mixing

    return d_epsilon


# -------------------------------------------------
# 3. SOLID FRACTION (phi)
# -------------------------------------------------

def compute_solid_fraction(x, u):
    """
    Solid fraction (phi) evolution.

    Influenced by:
    - feed_rate (mass inflow)
    - reaction (mass redistribution)
    """

    feed_rate = u[IDX_FEED]

    C_CaCO3 = x[IDX_CaCO3]
    C_CaO   = x[IDX_CaO]

    # Inflow contribution
    phi_in = phi_feed_effect(feed_rate)

    # Reaction redistribution
    phi_rxn = phi_reaction_effect(C_CaCO3, C_CaO)

    d_phi = phi_in + phi_rxn

    return d_phi


# -------------------------------------------------
# 4. SYMBOLIC PLACEHOLDERS
# -------------------------------------------------

def v_gas_model(fan_rpm, epsilon_l): pass
def v_solid_model(feed_rate, epsilon_l): pass

def porosity_reaction_effect(C_CaCO3, C_CaO): pass
def porosity_mixing_effect(reactor_rpm): pass

def phi_feed_effect(feed_rate): pass
def phi_reaction_effect(C_CaCO3, C_CaO): pass