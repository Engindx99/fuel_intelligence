import numpy as np
import yaml
import os

from core.state import StateIdx, N_STATES, SOLID_SPECIES, GAS_SPECIES, IDX_REACTOR
from core.kinetics import compute_reaction_rates, dX_kin

# -------------------------------------------------
# CONFIG LOAD
# -------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = os.path.join(_ROOT, "configs", "model_config.yaml")

def load_config():
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
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
_fuel_heat_mode = str(cfg["thermal"].get("fuel_heat_mode", "distributed")).lower()
# Burner efficiency is used to scale fuel heat release in distributed mode.
_burner_eff = float(cfg["thermal"].get("burner_efficiency", 1.0))
_T_amb = float(cfg["thermal"].get("t_amb", 300.0))
_ua_g_amb = float(cfg["thermal"].get("ua_g_amb", 0.0))
_ua_s_amb = float(cfg["thermal"].get("ua_s_amb", 0.0))
# Above-threshold ridge losses (extra W/m³ per K overshoot vs threshold) tame runaway.
_ua_g_sup = float(cfg["thermal"].get("ua_g_super_linear", 0.0))
_ua_s_sup = float(cfg["thermal"].get("ua_s_super_linear", 0.0))
_thr_g_sup = float(cfg["thermal"].get("superheat_gas_linear_threshold_k", 1.0e6))
_thr_s_sup = float(cfg["thermal"].get("superheat_solid_linear_threshold_k", 1.0e6))
# Distributed fuel axial spread (fraction of kiln length toward hot end, ~σ of Gaussian).
_fuel_gaussian_sigma = float(cfg["thermal"].get("fuel_gaussian_sigma", 0.12))
_dH_c3a = float(cfg["thermal"].get("enthalpy_c3a", 2.8e5))
_dH_c4af = float(cfg["thermal"].get("enthalpy_c4af", 2.6e5))
_mass_r4 = 3.0 * 0.05608 + 0.10200  # 3CaO + Al2O3 -> C3A (reactant mass per kg C3A)
_mass_r5 = 4.0 * 0.05608 + 0.10200 + 0.16000  # 4CaO + Al2O3 + Fe2O3 -> C4AF
# Axial weakening of gas<->solid exchange near cold exit (counter-current kiln).
_h_gs_near_cold_mult = float(cfg["thermal"].get("h_gs_near_cold_mult", 1.0))
_h_gs_near_cold_width_frac = float(cfg["thermal"].get("h_gs_near_cold_width_frac", 0.0))
# Lumped volumetric surrogate: linearized radiation 4 eps sigma T^3 often >> h_gs at HT,
# collapsing Tg≈Ts each sub-step (unphysical for kiln flame vs bed ΔT ~ 150–450 K bands).
_rad_gs_scale = float(cfg["thermal"].get("radiation_gas_solid_scale", 1.0))
_interphase_exchange_scale = float(cfg["thermal"].get("interphase_exchange_scale", 1.0))

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
    
    # NOTE: Do not clip the *state* here. If you need bounds for numerical stability
    # in aux calculations, use local clipped copies only.
    T_s = X[:, StateIdx.T_S]
    T_g = X[:, StateIdx.T_G]
    phi = np.maximum(X[:, StateIdx.PHI], 1e-4)

    # Reaksiyon Isıları
    r = compute_reaction_rates(X, np.clip(T_s, 250.0, 2500.0))
    r_calc = r[:, 0]
    r_c3s = r[:, 2]
    r_c3a = r[:, 3]
    r_c4af = r[:, 4]

    MW_CaCO3 = 0.10009
    MW_C3S = 0.22832
    # Q_rxn > 0: net heat uptake by solids (calcination dominates when active).
    # Exothermic clinker-phase formation terms reduce Q_rxn (same convention as R3).
    Q_rxn = (_dH_calc / MW_CaCO3) * r_calc - (_dH_c3s / MW_C3S) * r_c3s
    Q_rxn += -(_dH_c3a / _mass_r4) * r_c3a - (_dH_c4af / _mass_r5) * r_c4af

    # Fuel heat release:
    # - boundary mode: handled as hot-end gas inlet enthalpy in simulation/engine.py
    # - distributed mode: legacy volumetric deposition near hot end (z ~ 1)
    if _fuel_heat_mode == "distributed":
        z = np.linspace(0.0, 1.0, N)
        fuel_rate = u[0]  # IDX_FUEL
        sig = max(_fuel_gaussian_sigma, 0.04)
        # Broader Gaussian → some heat deposited in calciner/precalc surrogate band (cold end uplift).
        q_dist = np.exp(-np.square((1.0 - z) / sig))
        q_dist /= (np.sum(q_dist) / N + 1e-12)
        Q_fuel = (_burner_eff * fuel_rate * _lhv) * q_dist / _kiln_vol
    else:
        Q_fuel = 0.0

    # Not: Bu fonksiyon yalnızca SOURCE TERMS üretmelidir.
    # Adveksiyon (uzaysal taşınım) `simulation/engine.py` içinde zaten uygulanıyor;
    # burada tekrar eklemek çift-sayım yaparak ısınmayı/taşınımı bozuyor.
    # IMPORTANT:
    # Keep this function to *source terms only* (reaction heats + fuel deposition).
    # Inter-phase heat exchange (gas <-> solid) is handled in the engine using
    # an energy-conserving semi-implicit update to avoid artificial clipping.
    # Ambient losses as source terms (W/m^3) -> K/s
    Ts_c = np.clip(T_s, 200.0, 4000.0)
    Tg_c = np.clip(T_g, 200.0, 4000.0)
    Q_loss_s = _ua_s_amb * (Ts_c - _T_amb)
    Q_loss_g = _ua_g_amb * (Tg_c - _T_amb)
    Q_loss_s = Q_loss_s + _ua_s_sup * np.maximum(Ts_c - _thr_s_sup, 0.0)
    Q_loss_g = Q_loss_g + _ua_g_sup * np.maximum(Tg_c - _thr_g_sup, 0.0)

    solid_dT = (-Q_rxn - Q_loss_s) / (_rho_cp_s + 1e-6)
    gas_dT = (Q_fuel - Q_loss_g) / (_rho_cp_g + 1e-6)

    return solid_dT, gas_dT


def heat_exchange_coeff_vec(X: np.ndarray) -> np.ndarray:
    """
    Effective volumetric heat exchange coefficient k_eff [W/m^3/K] so that
    Q_xfer ≈ k_eff * (Tg - Ts).

    Radiation is linearized about T_bar: (Tg^4 - Ts^4) ≈ 4*T_bar^3*(Tg - Ts),
    then scaled (`radiation_gas_solid_scale`, `interphase_exchange_scale`) so the lumped
    surrogate does not force Tg≈Ts at flame temperatures.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    dz = _L / N

    Ts = X[:, StateIdx.T_S]
    Tg = X[:, StateIdx.T_G]
    phi = np.maximum(X[:, StateIdx.PHI], 1e-4)

    # Convective part
    h_gs = _h_base * (1.0 + 0.5 * phi)

    # Radiation linearization (use clipped aux temps for stability)
    Ts_c = np.clip(Ts, 200.0, 4000.0)
    Tg_c = np.clip(Tg, 200.0, 4000.0)
    Tbar = 0.5 * (Ts_c + Tg_c)
    k_rad = _rad_gs_scale * (4.0 * _sigma * _eps_rad * (Tbar ** 3))

    k_eff = _interphase_exchange_scale * (h_gs + k_rad)

    # Axial tapering near cold end (same factor as previously applied to Q_xfer)
    z_centers = (np.arange(N, dtype=np.float64) + 0.5) * dz
    z_norm = np.clip(z_centers / (_L + 1e-12), 0.0, 1.0)
    if _h_gs_near_cold_width_frac > 1e-9 and (_h_gs_near_cold_mult + 1e-12) < 1.0:
        ramp = np.clip(z_norm / max(_h_gs_near_cold_width_frac, 1e-9), 0.0, 1.0)
        blend = 0.5 * (1.0 - np.cos(np.pi * ramp))
        fac = _h_gs_near_cold_mult + (1.0 - _h_gs_near_cold_mult) * blend
        k_eff = fac * k_eff

    return k_eff

# -------------------------------------------------
# MASS DYNAMICS
# -------------------------------------------------
def mass_terms_vec(X, u):
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    # Kinetik reaksiyon hızları
    r = compute_reaction_rates(X, T=X[:, StateIdx.T_S])
    dX_reac = dX_kin(r) / (_rho_s + 1e-9)
    # Not: Adveksiyon `simulation/engine.py` içinde uygulanıyor; burada sadece reaksiyon
    # ve yerel kaynaklar (örn. porozite) dönülür.
    dC = dX_reac

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