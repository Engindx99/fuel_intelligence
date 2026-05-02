"""
Mass-conserving R1–R5 kinetics (18-STATE COMPATIBLE & STABILIZED).
R1-R3 primary line; R4/R5 simplified aluminate and ferrite channels.
"""

import os
import numpy as np
import yaml
from core.state import StateIdx, N_STATES

# -------------------------------------------------
# CONFIG LOAD
# -------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = os.path.join(_ROOT, "configs", "model_config.yaml")


def load_config():
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_cfg = load_config()

# Fiziksel Sabitler
R_gas = float(_cfg["physics"]["r_gas"])
A_calc = float(_cfg["physics"]["calcination"]["A"])
Ea_calc = float(_cfg["physics"]["calcination"]["Ea"])
A_c2s = float(_cfg["physics"]["c2s_formation"]["A"])
Ea_c2s = float(_cfg["physics"]["c2s_formation"]["Ea"])
A_c3s = float(_cfg["physics"]["c3s_formation"]["A"])
Ea_c3s = float(_cfg["physics"]["c3s_formation"]["Ea"])
_ph = _cfg["physics"]
A_c3a = float(_ph["c3a_formation"]["A"])
Ea_c3a = float(_ph["c3a_formation"]["Ea"])
A_c4af = float(_ph["c4af_formation"]["A"])
Ea_c4af = float(_ph["c4af_formation"]["Ea"])

# -------------------------------------------------
# MASS-BASED STOICHIOMETRY (NORMALIZED)
# -------------------------------------------------
MW_CaCO3 = 0.10009
MW_CaO = 0.05608
MW_SiO2 = 0.06008
MW_C2S = 0.17224
MW_C3S = 0.22832
MW_CO2 = 0.04401
MW_Al2O3 = 0.10200
MW_Fe2O3 = 0.16000
MW_C3A = 0.27000
MW_C4AF = 0.48600

N_RXN = 5
S = np.zeros((N_STATES, N_RXN), dtype=np.float64)

# R1: Kalsinasyon (CaCO3 -> CaO + CO2)
S[StateIdx.CaCO3, 0] = -1.0
S[StateIdx.CaO, 0] = MW_CaO / MW_CaCO3
S[StateIdx.CO2, 0] = MW_CO2 / MW_CaCO3

# R2: Belit (2CaO + SiO2 -> C2S)
m_total_r2 = (2 * MW_CaO + MW_SiO2)
S[StateIdx.CaO, 1] = -(2 * MW_CaO) / m_total_r2
S[StateIdx.SiO2, 1] = -MW_SiO2 / m_total_r2
S[StateIdx.C2S, 1] = 1.0

# R3: Alit (C2S + CaO -> C3S)
m_total_r3 = (MW_C2S + MW_CaO)
S[StateIdx.C2S, 2] = -MW_C2S / m_total_r3
S[StateIdx.CaO, 2] = -MW_CaO / m_total_r3
S[StateIdx.C3S, 2] = 1.0

# R4: Tricalcium aluminate skeleton (3CaO + Al2O3 -> C3A), mass-conserving lump
m_total_r4 = (3 * MW_CaO + MW_Al2O3)
S[StateIdx.CaO, 3] = -(3 * MW_CaO) / m_total_r4
S[StateIdx.Al2O3, 3] = -MW_Al2O3 / m_total_r4
S[StateIdx.C3A, 3] = 1.0

# R5: Ferrite lump (4CaO + Al2O3 + Fe2O3 -> C4AF)
m_total_r5 = (4 * MW_CaO + MW_Al2O3 + MW_Fe2O3)
S[StateIdx.CaO, 4] = -(4 * MW_CaO) / m_total_r5
S[StateIdx.Al2O3, 4] = -MW_Al2O3 / m_total_r5
S[StateIdx.Fe2O3, 4] = -MW_Fe2O3 / m_total_r5
S[StateIdx.C4AF, 4] = 1.0

# -------------------------------------------------
# REACTION RATES
# -------------------------------------------------
def safe_exp(x):
    return np.exp(np.clip(x, -100.0, 50.0), dtype=np.float64)


def compute_reaction_rates(X, T=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if T is None:
        T = X[:, StateIdx.T_S]
    else:
        T = np.asarray(T, dtype=np.float64).reshape(-1)

    T = np.clip(T, 300.0, 2500.0)
    invT = 1.0 / T
    Xp = np.maximum(X, 0.0)

    CaCO3 = Xp[:, StateIdx.CaCO3]
    CaO = Xp[:, StateIdx.CaO]
    SiO2 = Xp[:, StateIdx.SiO2]
    C2S = Xp[:, StateIdx.C2S]
    Al2O3 = Xp[:, StateIdx.Al2O3]
    Fe2O3 = Xp[:, StateIdx.Fe2O3]

    k1 = A_calc * safe_exp(-Ea_calc * invT / R_gas)
    k2 = A_c2s * safe_exp(-Ea_c2s * invT / R_gas)
    k3 = A_c3s * safe_exp(-Ea_c3s * invT / R_gas)

    # R4/R5 onset above belit formation band; stagger alumina-rich then ferrite uptake
    k4 = A_c3a * safe_exp(-Ea_c3a * invT / R_gas)
    k5 = A_c4af * safe_exp(-Ea_c4af * invT / R_gas)

    r1 = k1 * CaCO3
    r2 = k2 * CaO * SiO2
    switch_r3 = 1.0 / (1.0 + np.exp(-0.05 * (T - 1250.0)))
    r3 = k3 * np.minimum(C2S, CaO) * switch_r3

    switch_r4 = 1.0 / (1.0 + np.exp(-0.045 * (T - 1320.0)))
    # Aluminate competes weakly until free CaO,Si pair less dominant (soft cap via availability)
    r4 = k4 * CaO * Al2O3 * switch_r4

    switch_r5 = 1.0 / (1.0 + np.exp(-0.04 * (T - 1390.0)))
    r5 = k5 * CaO * np.maximum(Fe2O3, 0.0) * np.maximum(Al2O3, 0.0) * switch_r5

    return np.stack([r1, r2, r3, r4, r5], axis=1)


def dX_kin(r):
    """(N, 5) @ (5, N_STATES)"""
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        r = r.reshape(1, -1)
    return r @ S.T


# --- Legacy 6-species × R1–R3 stoichiometry (directive / unit tests only) ---
STOICHIOMETRY_MATRIX_S = np.array(
    [
        [-1, 0, 0],
        [1, -1, -1],
        [0, -1, 0],
        [0, 1, -1],
        [0, 0, 1],
        [1, 0, 0],
    ],
    dtype=np.float64,
)


def dXdt_kinetic_subspace(r: np.ndarray) -> np.ndarray:
    """(N,3) reaction rates → (N,6) increments for [CaCO3,CaO,SiO2,C2S,C3S,CO2]."""
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        r = r.reshape(1, -1)
    if r.shape[1] != 3:
        raise ValueError("dXdt_kinetic_subspace expects r shape (N, 3)")
    return r @ STOICHIOMETRY_MATRIX_S.T


def compute_reaction_rates_vec(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    r = compute_reaction_rates(X)
    return {
        "r_calcination": r[:, 0],
        "r_C2S": r[:, 1],
        "r_C3S": r[:, 2],
        "r_C3A": r[:, 3],
        "r_C4AF": r[:, 4],
    }
