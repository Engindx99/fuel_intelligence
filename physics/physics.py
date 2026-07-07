import numpy as np


# ======================================================
# FLOW
# ======================================================

def solid_mass_flow(feed_rate):

    m_dot_s = feed_rate # Solid feed rate [kg/s]

    return m_dot_s


def gas_mass_balance(fuel_rate_total, O2, eps):
    
    return fuel_rate_total * (1.0 + 0.8 * O2) # Combustion stoichiometry (SI units)


def residence_time(L, D, slope_deg, fill_fraction, rpm, eps):

    theta = np.deg2rad(slope_deg)

    tau = (
        1.77
        * L
        / (
            D
            * rpm
            * np.tan(theta)
            * (fill_fraction + eps)
        )
    )

    return tau


def solid_axial_velocity(L, D, slope_deg, fill_fraction, rpm, eps):

    tau = residence_time(
        L,
        D,
        slope_deg,
        fill_fraction,
        rpm,
        eps,
    )

    return L / (tau + eps)


def gas_axial_velocity(m_dot_g, rho_g, A_cross, eps):

    return m_dot_g / (rho_g * A_cross + eps)

# ======================================================
# COMBUSTION
# ======================================================
def combustion_efficiency(O2, O2_opt, O2_sigma2):

    return np.exp(
        -((O2 - O2_opt) ** 2) / O2_sigma2
    )
    
def fuel_heat_release(
    fuel_rate_total,     # kg/s
    O2,
    O2_opt,
    O2_sigma2,
    LHV,
    inputs,
    eps,
):

    # ================= FUEL MIX =================
    p = inputs.get("Petcoke_ratio", 0.50)
    c = inputs.get("Coal_ratio", 0.30)
    r = inputs.get("RDF_ratio", 0.15)
    h = inputs.get("H2_ratio", 0.05)

    norm = p + c + r + h + eps

    p /= norm
    c /= norm
    r /= norm
    h /= norm

    # ================= FUEL FLOW =================
    Q_petcoke = fuel_rate_total * p * LHV["petcoke"]
    Q_coal    = fuel_rate_total * c * LHV["coal"]
    Q_RDF     = fuel_rate_total * r * LHV["rdf"]
    Q_H2      = fuel_rate_total * h * LHV["h2"]

    # ================= COMBUSTION =================
    eta = combustion_efficiency(
        O2,
        O2_opt,
        O2_sigma2,
    )


    Q_burning = eta * (
        Q_petcoke
        + Q_coal
        + Q_RDF
        + Q_H2
    )

    return (
        Q_petcoke,
        Q_coal,
        Q_RDF,
        Q_H2,
        Q_burning,
    )

# ======================================================
# HEAT TRANSFER
# ======================================================

"""
k_eff is an effective radiation scaling factor representing
unresolved radiative effects, including view factors,
participating media, gas absorption, flame radiation,
and other complex heat transfer mechanisms.
"""

ZONE_RAD_CONFIG = {
    "burning": {
        "eps": 0.90,
        "k_eff": 0.70,
    },
    "transition": {
        "eps": 0.85,
        "k_eff": 0.35,
    },
    "calciner": {
        "eps": 0.80,
        "k_eff": 0.30,
    },
    "preheater": {
        "eps": 0.75,
        "k_eff": 0.22,
    },
    "cooler": {
        "eps": 0.60,
        "k_eff": 0.15,
    },
}

ZONE_HT_CONFIG = {
    "burning": {
        "hv_gs": 700.0,
        "hv_gw": 180.0,
        "hv_ws": 220.0,
    },
    "transition": {
        "hv_gs": 450.0,
        "hv_gw": 150.0,
        "hv_ws": 180.0,
    },
    "calciner": {
        "hv_gs": 350.0,
        "hv_gw": 120.0,
        "hv_ws": 150.0,
    },
    "preheater": {
        "hv_gs": 220.0,
        "hv_gw": 90.0,
        "hv_ws": 110.0,
    },
    "cooler": {
        "hv_gs": 180.0,
        "hv_gw": 70.0,
        "hv_ws": 90.0,
    },
}


sigma = 5.670374419e-8

def radiation(T1, T2, zone, area=1.0):
    """ Stefan–Boltzmann radiation model with zone-dependent tuning."""
    

    cfg = ZONE_RAD_CONFIG[zone]

    eps = cfg["eps"]
    k_eff = cfg["k_eff"]
    

    q_rad = k_eff * eps * sigma * area * (T1**4 - T2**4)


    return q_rad


def heat_transfer(Tg, Ts, Tw, hv_gs, hv_gw, hv_ws, a_gs, a_gw, a_ws, zone=None):

    # ======================================================
    # CONVECTION
    # ======================================================
    q_gs_conv = hv_gs * a_gs * (Tg - Ts)
    q_gw_conv = hv_gw * a_gw * (Tg - Tw)
    q_ws_conv = hv_ws * a_ws * (Ts - Tw)

    # ======================================================
    # RADIATION (STEFAN–BOLTZMANN)
    # ======================================================
    q_gs_rad = radiation(Tg, Ts, zone, area=a_gs)
    q_gw_rad = radiation(Tg, Tw, zone, area=a_gw)
    q_ws_rad = radiation(Ts, Tw, zone, area=a_ws)

    # ======================================================
    # TOTAL HEAT TRANSFER
    # ======================================================
    q_gs = q_gs_conv + q_gs_rad
    q_gw = q_gw_conv + q_gw_rad
    q_ws = q_ws_conv + q_ws_rad

    return q_gs, q_gw, q_ws

    
def wall_losses(
    Tw,
    h_ext,
    A_wall_cell,
    V_cell,
    T_amb,
    A_wall_total,
    N,
    refractory_thickness,
    refractory_conductivity,
    eps,
    insulation_factor=0.27,   # default calibration
):

    # ======================================================
    # THERMAL RESISTANCE NETWORK
    # ======================================================
    R_ref, R_conv, R_total = wall_thermal_resistance(
        refractory_thickness=refractory_thickness,
        refractory_conductivity=refractory_conductivity,
        h_ext=h_ext,
        A_wall_cell=A_wall_cell,
    )

    # ======================================================
    # HEAT LOSS (cell-wise)
    # ======================================================
    wall_loss_cells = (Tw - T_amb) / (R_total + eps)

    # ======================================================
    # TOTAL HEAT LOSS
    # ======================================================
    wall_loss_raw = np.sum(wall_loss_cells)

    # ======================================================
    # APPLY INSULATION FACTOR
    # ======================================================
    insulation_factor = np.clip(insulation_factor, 0.1, 1.0)

    wall_loss = insulation_factor * wall_loss_raw

    # ======================================================
    # VOLUMETRIC LOSS
    # ======================================================
    q_loss = insulation_factor * wall_loss_cells / (V_cell + eps)

    # ======================================================
    # DEBUG
    # ======================================================
    wall_debug = {
        "R_ref": float(R_ref),
        "R_conv": float(R_conv),
        "R_total": float(R_total),

        "insulation_factor": float(insulation_factor),

        "q_loss_mean": float(np.mean(q_loss)),
        "q_loss_max": float(np.max(q_loss)),

        "wall_loss_mean": float(np.mean(wall_loss_cells)),
        "wall_loss_total_raw": float(wall_loss_raw),
        "wall_loss_total": float(wall_loss),

        "A_wall": float(A_wall_total),
        "A_wall_cell": float(A_wall_cell),
        "V_cell": float(V_cell),
        "N": int(N),
    }

    return q_loss, wall_loss, wall_debug
    
# ======================================================
# THERMAL CAPACITIES
# ======================================================
def thermal_capacities(
    rho_g_Vcell_Cp_g,
    rho_s_Vcell_Cp_s,
    rho_wall_Vwall_cell_Cp,
    effective=0.01,
):

    C_s = rho_s_Vcell_Cp_s
    effective_C_s = effective * C_s

    C_g = rho_g_Vcell_Cp_g
    C_w = rho_wall_Vwall_cell_Cp

    return (
        C_g,
        effective_C_s,
        C_w,
    )
    
# ======================================================
# GEOMETRY
# ======================================================
def kiln_geometry(D, L, N):

    # Cross-sectional area (m²)
    A_cross = np.pi * D**2 / 4.0

    # Total kiln volume (m³)
    V_total = A_cross * L

    # Computational cell volume (m³)
    V_cell = V_total / N

    return (
        A_cross,
        V_total,
        V_cell,
    )
# ======================================================
# WALL GEOMETRY
# ======================================================
def wall_geometry(
    D,
    L,
    N,
    V_cell,
    refractory_thickness=0.05,
):

    # Kiln inner perimeter (m)
    wall_perimeter = np.pi * D

    # Total inner wall area (m²)
    A_wall_total = wall_perimeter * L

    # Wall area per computational cell (m²)
    A_wall_cell = A_wall_total / N

    # Gas-wall interfacial area density (m²/m³)
    a_gw = A_wall_cell / V_cell

    # Refractory wall volume (m³)
    V_wall = A_wall_total * refractory_thickness

    return (
        wall_perimeter,
        A_wall_total,
        A_wall_cell,
        a_gw,
        V_wall,
    )
    
# ======================================================
# INTERFACIAL AREAS
# ======================================================
def interfacial_areas(
    D,
    epsilon_bed,
    k_interfacial=1.0,
):

    # Gas-solid interfacial area density (m²/m³)
    a_gs_base = (
        6.0
        * (1.0 - epsilon_bed)
        / D
    )

    a_gs = k_interfacial * a_gs_base

    # Wall-solid interfacial area density (m²/m³)
    a_ws = 0.6 * a_gs

    return (
        a_gs,
        a_ws,
    )

# ======================================================
# WALL THERMAL RESISTANCE
# ======================================================
def wall_thermal_resistance(
    refractory_thickness,
    refractory_conductivity,
    h_ext,
    A_wall_cell,
):

    # Refractory conduction resistance (K/W)
    R_ref = (
        refractory_thickness
        / (
            refractory_conductivity
            * A_wall_cell
        )
    )

    # External convection resistance (K/W)
    R_conv = (
        1.0
        / (
            h_ext
            * A_wall_cell
        )
    )

    # Total thermal resistance (K/W)
    R_total = R_ref + R_conv

    return (
        R_ref,
        R_conv,
        R_total,
    )
    
