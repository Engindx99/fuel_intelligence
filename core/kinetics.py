"""
Mass-conserving R1–R3 kinetics (18-STATE COMPATIBLE & STABILIZED)
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
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)

_cfg = load_config()

# Fiziksel Sabitler
R_gas   = float(_cfg["physics"]["r_gas"])
A_calc  = float(_cfg["physics"]["calcination"]["A"])
Ea_calc = float(_cfg["physics"]["calcination"]["Ea"])
A_c2s   = float(_cfg["physics"]["c2s_formation"]["A"])
Ea_c2s  = float(_cfg["physics"]["c2s_formation"]["Ea"])
A_c3s   = float(_cfg["physics"]["c3s_formation"]["A"])
Ea_c3s  = float(_cfg["physics"]["c3s_formation"]["Ea"])

# -------------------------------------------------
# MASS-BASED STOICHIOMETRY (NORMALIZE EDİLMİŞ)
# -------------------------------------------------
# Moleküler Ağırlıklar (kg/mol)
MW_CaCO3 = 0.10009
MW_CaO   = 0.05608
MW_SiO2  = 0.06008
MW_C2S   = 0.17224
MW_C3S   = 0.22832
MW_CO2   = 0.04401

# S matrisi: (Bileşen Sayısı, Reaksiyon Sayısı)
S = np.zeros((N_STATES, 3), dtype=np.float64)

# R1: Kalsinasyon (CaCO3 -> CaO + CO2)
# Kütle korunum oranları (1 kg CaCO3 başına)
S[StateIdx.CaCO3, 0] = -1.0
S[StateIdx.CaO, 0]   = MW_CaO / MW_CaCO3
S[StateIdx.CO2, 0]   = MW_CO2 / MW_CaCO3

# R2: Belit Oluşumu (2CaO + SiO2 -> C2S)
# Tüketilen toplam kütleye oranla normalize edildi
m_total_r2 = (2 * MW_CaO + MW_SiO2)
S[StateIdx.CaO, 1]   = -(2 * MW_CaO) / m_total_r2
S[StateIdx.SiO2, 1]  = -MW_SiO2 / m_total_r2
S[StateIdx.C2S, 1]   = 1.0

# R3: Alit Oluşumu (C2S + CaO -> C3S)
m_total_r3 = (MW_C2S + MW_CaO)
S[StateIdx.C2S, 2]   = -MW_C2S / m_total_r3
S[StateIdx.CaO, 2]   = -MW_CaO / m_total_r3
S[StateIdx.C3S, 2]   = 1.0

# -------------------------------------------------
# REACTION RATES
# -------------------------------------------------
def safe_exp(x):
    """Exponential fonksiyonu için nümerik koruma."""
    return np.exp(np.clip(x, -100.0, 50.0), dtype=np.float64)

def compute_reaction_rates(X, T=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if T is None:
        T = X[:, StateIdx.T_S]
    else:
        T = np.asarray(T, dtype=np.float64).reshape(-1)

    # Sıcaklık ve Kütle Korunumu (Clipping)
    T = np.clip(T, 300.0, 2500.0)
    invT = 1.0 / T
    Xp = np.maximum(X, 0.0)

    # Bileşenler (Kütle Oranları)
    CaCO3 = Xp[:, StateIdx.CaCO3]
    CaO   = Xp[:, StateIdx.CaO]
    SiO2  = Xp[:, StateIdx.SiO2]
    C2S   = Xp[:, StateIdx.C2S]

    # Hız Sabitleri (k)
    k1 = A_calc * safe_exp(-Ea_calc * invT / R_gas)
    k2 = A_c2s  * safe_exp(-Ea_c2s  * invT / R_gas)
    k3 = A_c3s  * safe_exp(-Ea_c3s  * invT / R_gas)

    # Reaksiyon Hızları (r)
    r1 = k1 * CaCO3
    r2 = k2 * CaO * SiO2
    
    # R3 için Yumuşak Geçiş (Sayısal süreklilik sağlar)
    # 1200K civarında aniden başlamak yerine yumuşak bir eğri çizer.
    switch_r3 = 1.0 / (1.0 + np.exp(-0.05 * (T - 1250.0)))
    r3 = k3 * np.minimum(C2S, CaO) * switch_r3

    return np.stack([r1, r2, r3], axis=1)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def dX_kin(r):
    """
    (N, 3) @ (3, 18) -> (N, 18) kütle değişim hızı.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        r = r.reshape(1, -1)
    return r @ S.T 

def compute_reaction_rates_vec(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    r = compute_reaction_rates(X)
    return {
        "r_calcination": r[:, 0],
        "r_C2S": r[:, 1],
        "r_C3S": r[:, 2]
    }