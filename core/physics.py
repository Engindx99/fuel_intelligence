import numpy as np
import yaml
import os

from core.state import StateIdx, N_STATES
from core.kinetics import compute_reaction_rates, dX_kin


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = os.path.join(_ROOT, "configs", "model_config.yaml")


def load_config():
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()


# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------

_r_gas = cfg["physics"]["r_gas"]

_diff_cfg = cfg["physics"]["diffusion"]
_D0 = _diff_cfg["D0"]
_tau = _diff_cfg["tau"]

_rho_s = cfg["thermal"]["rho_solid"]
_cp_s = cfg["thermal"]["cp_solid"]
_rho_g = cfg["thermal"]["rho_gas"]
_cp_g = cfg["thermal"]["cp_gas"]

_lhv = cfg["thermal"]["lhv_fuel"]
_h_base = cfg["thermal"]["h_gs_base"]
_eps_rad = cfg["thermal"]["emissivity"]
_sigma = cfg["thermal"]["sigma_sb"]

_dH_calc = cfg["thermal"]["enthalpy_calc"]
_dH_c3s = cfg["thermal"]["enthalpy_c3s"]

_geom = cfg["kiln_geometry"]
_D = float(_geom["diameter"])
_A_cs = np.pi * (_D / 2.0) ** 2
_length = float(_geom["length"])

_rho_cp_s = _rho_s * _cp_s
_rho_cp_g = _rho_g * _cp_g
_kiln_vol = _A_cs * _length


# -------------------------------------------------
# ENERGY
# -------------------------------------------------

def energy_terms_vec(X, u):

    X = np.asarray(X, dtype=np.float64)

    T_s = np.clip(X[:, StateIdx.T_S], 200, 3000)
    T_g = np.clip(X[:, StateIdx.T_G], 200, 3000)

    phi = np.maximum(X[:, StateIdx.PHI], 0.0)

    # kinetics
    r = compute_reaction_rates(X, T_s)

    r_calc = r[:, 0]
    r_c3s = r[:, 2]

    MW_CaCO3 = 0.10009
    MW_C3S = 0.1723

    Q_rxn = (_dH_calc / MW_CaCO3) * r_calc - (_dH_c3s / MW_C3S) * r_c3s

    # heat transfer
    h_gs = _h_base * (1.0 + 0.5 * phi)

    Q_conv = h_gs * (T_g - T_s)
    Q_rad = _sigma * _eps_rad * (T_g**4 - T_s**4)

    Q_xfer = Q_conv + Q_rad

    # burner (no linspace allocation)
    N = X.shape[0]
    z = np.arange(N, dtype=np.float64) / (N - 1 + 1e-12)

    fuel_rate = u[0]

    q_dist = np.exp(-(1.0 - z)**2 / 0.05)
    q_dist /= (np.mean(q_dist) + 1e-12)

    Q_fuel = fuel_rate * _lhv * q_dist / _kiln_vol

    solid_dT = (-Q_rxn + Q_xfer + 0.05 * Q_fuel) / _rho_cp_s
    gas_dT = (0.95 * Q_fuel - Q_xfer) / _rho_cp_g

    return solid_dT, gas_dT


# -------------------------------------------------
# MASS
# -------------------------------------------------

def mass_terms_vec(X, u):

    X = np.asarray(X, dtype=np.float64)

    r = compute_reaction_rates(X, T=X[:, StateIdx.T_S])
    dX = dX_kin(r)

    dC = np.zeros((X.shape[0], N_STATES), dtype=np.float64)

    kin_idx = np.array([
        StateIdx.CaCO3,
        StateIdx.CaO,
        StateIdx.SiO2,
        StateIdx.C2S,
        StateIdx.C3S,
        StateIdx.CO2
    ], dtype=np.intp)

    dC[:, kin_idx] = dX

    # inert
    dC[:, StateIdx.C3A] = 0.0
    dC[:, StateIdx.C4AF] = 0.0
    dC[:, StateIdx.Al2O3] = 0.0
    dC[:, StateIdx.Fe2O3] = 0.0

    # porosity
    dC[:, StateIdx.EPSILON] = 0.05 * r[:, 0]

    return dC


# -------------------------------------------------
# SINGLE CELL WRAPPERS
# -------------------------------------------------

def energy_terms(x, u):
    x = np.asarray(x).reshape(1, -1)
    u = np.asarray(u)

    s, g = energy_terms_vec(x, u)

    return {
        "solid_energy": float(s[0]),
        "gas_energy": float(g[0])
    }


def mass_terms(x, u):
    x = np.asarray(x).reshape(1, -1)
    u = np.asarray(u)

    dC = mass_terms_vec(x, u)[0]

    return {i: float(dC[i]) for i in range(len(dC))}