"""
Mass-conserving R1–R3 kinetics (STABLE KILN VERSION)
"""

import os
import numpy as np
import yaml
from core.state import StateIdx

# -------------------------------------------------
# CONFIG LOAD
# -------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = os.path.join(_ROOT, "configs", "model_config.yaml")

def load_config():
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)

_cfg = load_config()

R_gas = _cfg["physics"]["r_gas"]
A_calc = _cfg["physics"]["calcination"]["A"]
Ea_calc = _cfg["physics"]["calcination"]["Ea"]
A_c2s = _cfg["physics"]["c2s_formation"]["A"]
Ea_c2s = _cfg["physics"]["c2s_formation"]["Ea"]
A_c3s = _cfg["physics"]["c3s_formation"]["A"]
Ea_c3s = _cfg["physics"]["c3s_formation"]["Ea"]

_C3S_GATE_TEMP_K = 1200.0

# -------------------------------------------------
# MASS-BASED STOICHIOMETRY
# -------------------------------------------------
MW_CaCO3 = 0.10009
MW_CaO   = 0.05608
MW_SiO2  = 0.06008
MW_C2S   = 0.17224
MW_C3S   = 0.22832
MW_CO2   = 0.04401

S = np.array([
    [-MW_CaCO3,      0.0,       0.0],   
    [ MW_CaO,   -2*MW_CaO,  -MW_CaO],   
    [ 0.0,       -MW_SiO2,      0.0],   
    [ 0.0,        MW_C2S,   -MW_C2S],   
    [ 0.0,           0.0,    MW_C3S],   
    [ MW_CO2,        0.0,       0.0],   
], dtype=np.float64)

# -------------------------------------------------
# REACTION RATES
# -------------------------------------------------
def safe_exp(x):
    return np.exp(np.clip(x, -50.0, 50.0))

def compute_reaction_rates(X, T=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if T is None:
        T = X[:, StateIdx.T_S]
    else:
        T = np.asarray(T, dtype=np.float64).reshape(-1)

    T = np.maximum(T, 300.0)
    invT = 1.0 / T
    Xp = np.maximum(X, 0.0)

    CaCO3 = Xp[:, StateIdx.CaCO3]
    CaO   = Xp[:, StateIdx.CaO]
    SiO2  = Xp[:, StateIdx.SiO2]
    C2S   = Xp[:, StateIdx.C2S]

    k1 = A_calc * safe_exp(-Ea_calc * invT / R_gas)
    k2 = A_c2s  * safe_exp(-Ea_c2s  * invT / R_gas)
    k3 = A_c3s  * safe_exp(-Ea_c3s  * invT / R_gas)

    r1 = k1 * CaCO3
    r2 = k2 * CaO * SiO2
    gate = (T >= _C3S_GATE_TEMP_K).astype(np.float64)
    r3 = gate * k3 * np.minimum(C2S, CaO)

    return np.stack([r1, r2, r3], axis=1)

# -------------------------------------------------
# IMPORT ERROR'I ÇÖZEN EKSİK FONKSİYON
# -------------------------------------------------
def compute_reaction_rates_vec(X):
    """Energy ve Validation modülleri için sözlük yapısında çıktı üretir."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    r = compute_reaction_rates(X, T=X[:, StateIdx.T_S])

    return {
        "r_calcination": r[:, 0],
        "r_C2S": r[:, 1],
        "r_C3S": r[:, 2],
        "r_C3A": np.zeros(len(r)),
        "r_C4AF": np.zeros(len(r)),
    }

# -------------------------------------------------
# INTEGRATION
# -------------------------------------------------
def dX_kin(r):
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        r = r.reshape(1, -1)
    return r @ S.T

def kinetics_step(X, T):
    r = compute_reaction_rates(X, T)
    dx = dX_kin(r)
    return dx, r

# Aliaslar (Diğer modüllerle uyumluluk için)
dXdt_kinetic_subspace = dX_kin