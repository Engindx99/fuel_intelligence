import numpy as np
import os
import yaml
from core.state import *
from core.physics import energy_terms_vec, mass_terms_vec, heat_exchange_coeff_vec
from core.flow import compute_velocities
from core.correction import mass_drift_correction

class KilnSimulation:
    def __init__(self, L, N_cells, dt):
        self.L = L
        self.N = N_cells
        self.dt = dt
        self.dz = L / N_cells
        self.t = 0.0
        self.X = np.zeros((N_cells, N_STATES), dtype=np.float64)
        # Config-driven boundary conditions (UTF-8 safe on Windows)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(root, "configs", "model_config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        thermal = (cfg.get("thermal") or {})
        self._fuel_heat_mode = str(thermal.get("fuel_heat_mode", "distributed")).lower()
        self._T_s_in = float(thermal.get("t_solid_inlet", 300.0))
        self._solid_inlet_soft = bool(thermal.get("t_solid_inlet_soft_bc", False))
        # Burner-driven hot-end gas inlet temperature (no prescribed ramp).
        self._T_air_in = float(thermal.get("t_air_in", 300.0))
        self._lhv = float(thermal.get("lhv_fuel", 3.2e7))
        self._burner_eff = float(thermal.get("burner_efficiency", 0.55))
        self._burner_tau = float(thermal.get("burner_tau_s", 7200.0))
        self._Tg_min = float(thermal.get("burner_t_gas_min", self._T_air_in))
        self._Tg_max_burner = float(thermal.get("burner_t_gas_max", 2600.0))
        self._mdot_g_base = float(thermal.get("mdot_gas_base", 2.0))
        self._mdot_g_per_fan = float(thermal.get("mdot_gas_per_fan", 0.002))
        self._T_s_min = float(thermal.get("t_solid_min", 300.0))
        self._T_s_max = float(thermal.get("t_solid_max", 9800.0))
        self._T_g_max = float(thermal.get("t_gas_max", 9800.0))
        self._enforce_thermal_caps = bool(thermal.get("enforce_yaml_thermal_caps", True))
        # Weak axial turbulent-like mixing along Tg smooths axial gradients (helps exit-vs-hot-end coherence).
        self._gas_axial_mix_m2_s = float(thermal.get("gas_axial_thermal_mix_m2_s", 0.0))
        # Volumetric heat capacities (for energy terms expressed per volume)
        rho_s = float(thermal.get("rho_solid", 1600.0))
        cp_s = float(thermal.get("cp_solid", 780.0))
        rho_g = float(thermal.get("rho_gas", 1.2))
        cp_g = float(thermal.get("cp_gas", 1050.0))
        self._rho_cp_s = rho_s * cp_s
        self._rho_cp_g = rho_g * cp_g
        self._cp_g = cp_g
        # Thermodynamically consistent wall / refractory thermal inertia (optional).
        self._wall_on = bool(thermal.get("wall_enabled", False))
        self._wall_rho_cp = float(thermal.get("wall_rho_cp", 1.0e6))
        self._h_gw = float(thermal.get("h_gw", 0.0))
        self._h_ws = float(thermal.get("h_ws", 0.0))
        self._wall_loss = float(thermal.get("wall_loss", 0.0))
        self._T_amb = float(thermal.get("t_amb", 300.0))
        wall_init = float(thermal.get("wall_t_init", self._T_s_in))
        self.Tw = np.full((N_cells,), wall_init, dtype=np.float64)
        # Burner state (hot-end gas inlet temperature)
        self.Tg_in = float(np.clip(wall_init, self._Tg_min, self._Tg_max_burner))
        # Inlet composition snapshot (set in set_initial_condition)
        self._solid_inlet = None
        self._eps_inlet = None
        self._gas_inlet = None
        self._phi_inlet = None

        # Advection index groups (do NOT advect the full state with both velocities)
        # Solid flows from z=0 -> L; Gas flows counter-current from z=L -> 0
        self._solid_adv_idx = np.array([IDX_T_S, *SOLID_SPECIES, IDX_EPSILON], dtype=int)
        self._gas_adv_idx = np.array([IDX_T_G, *GAS_SPECIES, IDX_PHI], dtype=int)
        
        # -----------------------------
        # HIZLANDIRMA: Sabitleri Önden Tanımla
        # -----------------------------
        self.mw = np.zeros(N_STATES)
        self.mw[IDX_CaCO3] = 0.10009
        self.mw[IDX_CaO]   = 0.05608
        self.mw[IDX_SiO2]  = 0.06008
        self.mw[IDX_C2S]   = 0.17224
        self.mw[IDX_C3S]   = 0.22832
        self.mw[IDX_CO2]   = 0.04401
        self.mw[IDX_C3A]   = 0.27000
        self.mw[IDX_C4AF]  = 0.48600
        self.mw[IDX_Al2O3] = 0.10200
        self.mw[IDX_Fe2O3] = 0.16000

    def set_initial_condition(self, x0):
        x0 = np.asarray(x0, dtype=np.float64)
        self.X[:] = x0
        # Solid feed snapshot at z=0; gas composition/phi snapshot at z=L (hot-end inlet).
        if x0.ndim == 2:
            row_s = x0[0]
            row_g = x0[-1]
        else:
            row_s = row_g = x0
        self._solid_inlet = row_s[SOLID_SPECIES].copy()
        self._eps_inlet = float(row_s[IDX_EPSILON])
        self._gas_inlet = row_g[GAS_SPECIES].copy()
        self._phi_inlet = float(row_g[IDX_PHI])

    def _mdot_gas(self, u) -> float:
        md = self._mdot_g_base + self._mdot_g_per_fan * float(u[IDX_FAN])
        return float(max(1e-6, md))

    def _burner_equilibrium_Tg(self, u) -> float:
        """Hot-end gas equilibrium temperature from fuel+air enthalpy balance."""
        # If fuel heat is deposited as a distributed volumetric source (core/physics),
        # keep the hot-end inlet at the air temperature to avoid double-counting.
        if self._fuel_heat_mode == "distributed":
            return float(np.clip(self._T_air_in, self._Tg_min, self._Tg_max_burner))
        mdot_g = self._mdot_gas(u)
        mfuel = float(max(0.0, u[IDX_FUEL]))
        Tg_eq = self._T_air_in + (self._burner_eff * self._lhv * mfuel) / (mdot_g * (self._cp_g + 1e-12))
        return float(np.clip(Tg_eq, self._Tg_min, self._Tg_max_burner))

    def step(self, u):
        # Yerel değişkenlere atama (hız için)
        X = self.X
        N = self.N
        dz = self.dz
        dt = self.dt

        # -----------------------------
        # 1. VELOCITIES & DYNAMIC CFL
        # -----------------------------
        # Orta hücre yerine giriş/çıkış kontrolü daha güvenlidir
        v_s, v_g = compute_velocities(X[N // 2], u)
        
        max_v = max(abs(v_s), abs(v_g))
        cfl = max_v * dt / dz
        
        if cfl > 0.8:
            dt = 0.8 * dz / (max_v + 1e-12)
            self.dt = dt

        # Burner update (hot-end gas inlet temperature)
        Tg_eq = self._burner_equilibrium_Tg(u)
        # First-order lag
        tau = max(1e-6, self._burner_tau)
        self.Tg_in += (dt / tau) * (Tg_eq - self.Tg_in)
        self.Tg_in = float(np.clip(self.Tg_in, self._Tg_min, self._Tg_max_burner))

        # -----------------------------
        # 2. ADVECTION (Upwind) - split by phase
        # -----------------------------
        dX = np.zeros_like(X)

        # Solid-like states (advect with v_s)
        s_idx = self._solid_adv_idx
        diff_s = (X[1:, s_idx] - X[:-1, s_idx]) / dz
        if v_s >= 0:
            dX[1:, s_idx] -= v_s * diff_s
        else:
            dX[:-1, s_idx] -= v_s * diff_s

        # Gas-like states (advect with v_g)
        g_idx = self._gas_adv_idx
        diff_g = (X[1:, g_idx] - X[:-1, g_idx]) / dz
        if v_g <= 0:
            dX[:-1, g_idx] -= v_g * diff_g
        else:
            dX[1:, g_idx] -= v_g * diff_g

        # -----------------------------
        # 3. SOURCE TERMS (Vektörize)
        # -----------------------------
        dE_s, dE_g = energy_terms_vec(X, u)
        dM = mass_terms_vec(X, u)

        # Source-only temperature terms (reaction heats + fuel deposition).
        dX[:, IDX_T_S] += dE_s
        dX[:, IDX_T_G] += dE_g
        dX += dM

        # Solid feed inlet (z=0): missing ghost-cell upwind ⇒ add convective enthalpy
        # influx so hot meal (~t_solid_inlet) is physically carried into the kiln.
        # Without this, cell 0 is advection-starved and equilibrates with cold gas
        # (dead zone + collapsing exitTs / exitTg paradox in counter-current plots).
        if v_s >= 0:
            dX[0, IDX_T_S] += v_s * (self._T_s_in - X[0, IDX_T_S]) / dz

        # -----------------------------
        # 4. EULER UPDATE
        # -----------------------------
        X_new = X + dt * dX

        # -----------------------------
        # 4a-lite. Explicit axial smoothing on gas temperature only (thermal diffusion surrogate).
        # -----------------------------
        chi = float(self._gas_axial_mix_m2_s)
        if chi > 1e-18:
            Tgx = X_new[:, IDX_T_G]
            lap = np.zeros_like(Tgx)
            # Interior Laplacian; boundaries unchanged → near Neumann flavour.
            if N > 2:
                lap[1:-1] = (Tgx[:-2] - 2.0 * Tgx[1:-1] + Tgx[2:]) / (dz * dz)
                X_new[1:-1, IDX_T_G] += chi * dt * lap[1:-1]

        # -----------------------------
        # 4a. ENERGY-CONSERVING GAS <-> SOLID HEAT EXCHANGE (semi-implicit)
        # -----------------------------
        # Use k_eff such that Q_xfer ≈ k_eff*(Tg - Ts). Solve the coupled exchange
        # implicitly to avoid artificial clipping while conserving sensible energy.
        k_eff = heat_exchange_coeff_vec(X_new)  # (N,)
        Ts0 = X_new[:, IDX_T_S].astype(np.float64, copy=False)
        Tg0 = X_new[:, IDX_T_G].astype(np.float64, copy=False)

        a = (dt * k_eff) / (self._rho_cp_s + 1e-12)
        b = (dt * k_eff) / (self._rho_cp_g + 1e-12)

        # Solve:
        # (1+a)Ts - a Tg = Ts0
        # -b Ts + (1+b)Tg = Tg0
        denom = (1.0 + a) * (1.0 + b) - (a * b)
        Ts = ((1.0 + b) * Ts0 + a * Tg0) / (denom + 1e-18)
        Tg = (b * Ts0 + (1.0 + a) * Tg0) / (denom + 1e-18)

        X_new[:, IDX_T_S] = Ts
        X_new[:, IDX_T_G] = Tg

        # -----------------------------
        # 4b. WALL THERMAL INERTIA (energy exchange gas <-> wall <-> solid)
        # -----------------------------
        if self._wall_on and self._wall_rho_cp > 0.0 and (self._h_gw > 0.0 or self._h_ws > 0.0):
            Ts = X_new[:, IDX_T_S]
            Tg = X_new[:, IDX_T_G]
            Tw = self.Tw

            # Effective volumetric heat exchange (W/m^3). Signs chosen so:
            # - Gas loses when Tg > Tw
            # - Solid gains when Tw > Ts
            Q_gw = self._h_gw * (Tg - Tw)
            Q_ws = self._h_ws * (Tw - Ts)
            Q_loss = self._wall_loss * (Tw - self._T_amb)

            # Convert to temperature rates and apply Euler step
            X_new[:, IDX_T_G] += dt * ((-Q_gw) / (self._rho_cp_g + 1e-12))
            X_new[:, IDX_T_S] += dt * ((Q_ws) / (self._rho_cp_s + 1e-12))
            Tw_new = Tw + dt * ((Q_gw - Q_ws - Q_loss) / (self._wall_rho_cp + 1e-12))
            self.Tw = Tw_new

        # -----------------------------
        # 5. MASS DRIFT CORRECTION (Önceden hesaplanmış mw ile)
        # -----------------------------
        X_new = mass_drift_correction(
            X_new,
            mw=self.mw,
            strength=0.7
        )

        # -----------------------------
        # 6. BOUNDARY CONDITIONS (Phase-consistent)
        # -----------------------------
        # Solid inlet at z=0 (hard Ts BC can create a thin shock + long low-gradient “plateau”).
        if not self._solid_inlet_soft:
            X_new[0, IDX_T_S] = self._T_s_in
        if self._solid_inlet is not None:
            X_new[0, SOLID_SPECIES] = self._solid_inlet
        if self._eps_inlet is not None:
            X_new[0, IDX_EPSILON] = self._eps_inlet

        # Gas inlet at z=L (gas flows from L -> 0)
        # In boundary-heating mode, hot-end inlet temperature is prescribed by the burner model.
        # In distributed-heating mode, Tg evolves with volumetric heat release near the hot end,
        # so we only enforce composition/phi and leave Tg free (outflow-type boundary).
        if self._fuel_heat_mode != "distributed":
            X_new[-1, IDX_T_G] = self.Tg_in
        if self._gas_inlet is not None:
            X_new[-1, GAS_SPECIES] = self._gas_inlet
        if self._phi_inlet is not None:
            X_new[-1, IDX_PHI] = self._phi_inlet

        # -----------------------------
        # 7. STABILIZATION (In-place clip kullanarak hızlanma)
        # -----------------------------
        # Negatif kütle oluşumunu önle
        np.clip(X_new[:, SOLID_SPECIES], 0.0, None, out=X_new[:, SOLID_SPECIES])
        np.clip(X_new[:, GAS_SPECIES], 0.0, None, out=X_new[:, GAS_SPECIES])

        # Thermal: enforce YAML sensible bounds (equipment / bed limit surrogate). Only NaN guards otherwise.
        if self._enforce_thermal_caps:
            np.clip(X_new[:, IDX_T_S], self._T_s_min, self._T_s_max, out=X_new[:, IDX_T_S])
            np.clip(X_new[:, IDX_T_G], self._Tg_min, self._T_g_max, out=X_new[:, IDX_T_G])

        badT = ~np.isfinite(X_new[:, IDX_T_S]) | ~np.isfinite(X_new[:, IDX_T_G])
        if np.any(badT):
            X_new[badT, IDX_T_S] = X[badT, IDX_T_S]
            X_new[badT, IDX_T_G] = X[badT, IDX_T_G]
        if self._wall_on:
            badW = ~np.isfinite(self.Tw)
            if np.any(badW):
                self.Tw[badW] = wall_init if "wall_init" in locals() else self._T_s_in

        # -----------------------------
        # 8. STATE UPDATE
        # -----------------------------
        self.X = X_new
        self.t += dt

        return self.X