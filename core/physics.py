import numpy as np
import yaml
import os

from core.state import StateIdx, N_STATES, SOLID_SPECIES, GAS_SPECIES
from core.kinetics import compute_reaction_rates, dX_kin

# -------------------------------------------------
# CONFIG LOAD
# -------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = os.path.join(_ROOT, "configs", "model_config.yaml")

def load_config():
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

# -------------------------------------------------
# PARAMETERS (Constants)
# -------------------------------------------------
_r_gas = cfg["physics"]["r_gas"]
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
_L = float(_geom["length"])

_rho_cp_s = _rho_s * _cp_s
_rho_cp_g = _rho_g * _cp_g
_kiln_vol = _A_cs * _L

# -------------------------------------------------
# ENERGY DYNAMICS
# -------------------------------------------------
def energy_terms_vec(X, u):
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    dz = _L / N
    
    # Kararlılık için clipping
    T_s = np.clip(X[:, StateIdx.T_S], 250.0, 2500.0)
    T_g = np.clip(X[:, StateIdx.T_G], 250.0, 2800.0)
    phi = np.maximum(X[:, StateIdx.PHI], 1e-4)

    # Reaksiyon Isıları
    r = compute_reaction_rates(X, T_s)
    r_calc = r[:, 0]
    r_c3s = r[:, 2]

    MW_CaCO3 = 0.10009
    MW_C3S = 0.22832
    Q_rxn = (_dH_calc / MW_CaCO3) * r_calc - (_dH_c3s / MW_C3S) * r_c3s

    # Isı Transferi
    h_gs = _h_base * (1.0 + 0.5 * phi)
    Q_conv = h_gs * (T_g - T_s)
    Q_rad = _sigma * _eps_rad * (T_g**4 - T_s**4)
    Q_xfer = Q_conv + Q_rad

    # Brülör Alev Dağılımı (Z=1.0'dan giriş yapar, yayılarak ilerler)
    z = np.linspace(0, 1, N)
    fuel_rate = u[0] # IDX_FUEL
    q_dist = np.exp(-(1.0 - z)**2 / 0.08) # Daha geniş yayılım (0.02 -> 0.08)
    q_dist /= (np.sum(q_dist) / N + 1e-12)
    Q_fuel = (fuel_rate * _lhv) * q_dist / _kiln_vol

    # --- TAŞINIM (ADVECTION) TERİMLERİ ---
    # Katı soldan sağa (0 -> L), Gaz sağdan sola (L -> 0) akar
    v_s = calculate_vs(u[3]) # RPM'den hız hesabı
    v_g = 5.0 # Ortalama gaz hızı (m/s)

    # Katı Isı Taşınımı (Upwind)
    dT_s_adv = np.zeros(N)
    dT_s_adv[1:] = -v_s * (T_s[1:] - T_s[:-1]) / dz
    
    # Gaz Isı Taşınımı (Downwind - Çünkü gaz ters akar)
    dT_g_adv = np.zeros(N)
    dT_g_adv[:-1] = v_g * (T_g[1:] - T_g[:-1]) / dz

    solid_dT = ((-Q_rxn + Q_xfer) / (_rho_cp_s + 1e-6)) + dT_s_adv
    gas_dT = ((Q_fuel - Q_xfer) / (_rho_cp_g + 1e-6)) + dT_g_adv

    return solid_dT, gas_dT

# -------------------------------------------------
# MASS DYNAMICS
# -------------------------------------------------
def mass_terms_vec(X, u):
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    dz = _L / N
    v_s = calculate_vs(u[3]) # RPM'e bağlı hız

    # Kinetik reaksiyon hızları
    r = compute_reaction_rates(X, T=X[:, StateIdx.T_S])
    dX_reac = dX_kin(r) / (_rho_s + 1e-9)

    # --- KÜTLE TAŞINIMI (ADVECTION) ---
    # Upwind farkı: Maddeler bir sonraki hücreye itilir
    dX_adv = np.zeros_like(X)
    # Başlangıç hücresi (z=0) dışarıdan besleme alır (simülasyon motoru halleder)
    # Diğer hücreler bir önceki hücreden madde çeker
    dX_adv[1:, SOLID_SPECIES] = -v_s * (X[1:, SOLID_SPECIES] - X[:-1, SOLID_SPECIES]) / dz

    dC = dX_reac + dX_adv

    # Gözeneklilik değişimi
    dC[:, StateIdx.EPSILON] = 0.02 * r[:, 0]

    return dC

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def calculate_vs(rpm):
    """
    Fırın hızı hesabı (m/s). 
    3.5 RPM ve %3 eğim için yaklaşık 0.015 - 0.03 m/s üretir.
    """
    slope = 0.03 # %3 eğim
    return (rpm * _D * slope) / (0.19 * _L + 1e-6)

def energy_terms(x, u):
    s_dt, g_dt = energy_terms_vec(x, u)
    return {"solid_energy": float(s_dt[0]), "gas_energy": float(g_dt[0])}

def mass_terms(x, u):
    dC = mass_terms_vec(x, u)
    if dC.ndim > 1: dC = dC[0]
    res = {i: float(dC[i]) for i in range(len(dC))}
    for idx in StateIdx:
        res[idx.name] = float(dC[idx])
    return res