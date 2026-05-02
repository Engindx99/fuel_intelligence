import os
import yaml
import numpy as np


def _load_cfg():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "configs", "model_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def total_energy_J(X, L: float, D: float, Tw=None) -> float:
    """
    Crude total sensible energy in domain (gas + solid + optional wall).
    Uses rho*cp from config, assumes per-cell volume = A_cs * dz.
    """
    cfg = _load_cfg()
    th = (cfg.get("thermal") or {})
    rho_s = float(th.get("rho_solid", 1600.0))
    cp_s = float(th.get("cp_solid", 780.0))
    rho_g = float(th.get("rho_gas", 1.2))
    cp_g = float(th.get("cp_gas", 1050.0))
    wall_rho_cp = float(th.get("wall_rho_cp", 0.0))

    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    dz = float(L) / float(N)
    A_cs = np.pi * (float(D) / 2.0) ** 2
    V = A_cs * dz

    Ts = X[:, 0]  # IDX_T_S
    Tg = X[:, 1]  # IDX_T_G
    E = float(np.sum((rho_s * cp_s * Ts + rho_g * cp_g * Tg) * V))
    if Tw is not None and wall_rho_cp > 0.0:
        Tw = np.asarray(Tw, dtype=float).reshape(-1)
        E += float(np.sum((wall_rho_cp * Tw) * V))
    return E

