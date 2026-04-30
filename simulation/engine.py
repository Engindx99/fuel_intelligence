import numpy as np

from core.state import *
from core.physics import energy_terms_vec, mass_terms_vec
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

    def set_initial_condition(self, x0):
        self.X[:] = np.asarray(x0, dtype=np.float64)

    def step(self, u):

        X = self.X
        N = self.N
        dz = self.dz
        dt = self.dt

        # -----------------------------
        # 1. VELOCITIES (CFL CHECK)
        # -----------------------------
        v_s, v_g = compute_velocities(X[N // 2], u)

        cfl = max(abs(v_s), abs(v_g)) * dt / dz
        if cfl > 0.8:
            dt = 0.8 * dz / (max(abs(v_s), abs(v_g)) + 1e-12)
            self.dt = dt

        # -----------------------------
        # 2. ADVECTION (UPWIND)
        # -----------------------------
        dX = np.zeros_like(X)

        if v_s >= 0:
            dX[1:, :] -= v_s * (X[1:, :] - X[:-1, :]) / dz
        else:
            dX[:-1, :] -= v_s * (X[1:, :] - X[:-1, :]) / dz

        if v_g <= 0:
            dX[:-1, :] -= v_g * (X[1:, :] - X[:-1, :]) / dz
        else:
            dX[1:, :] -= v_g * (X[1:, :] - X[:-1, :]) / dz

        # -----------------------------
        # 3. SOURCE TERMS
        # -----------------------------
        dE_s, dE_g = energy_terms_vec(X, u)
        dM = mass_terms_vec(X, u)

        dX[:, IDX_T_S] += dE_s
        dX[:, IDX_T_G] += dE_g
        dX += dM

        # -----------------------------
        # 4. EULER UPDATE
        # -----------------------------
        X_new = X + dt * dX

        # -----------------------------
        # 🔥 5. MASS DRIFT CORRECTION (CRITICAL)
        # -----------------------------
        mw = np.zeros(N_STATES)

        mw[IDX_CaCO3] = 0.10009
        mw[IDX_CaO]   = 0.05608
        mw[IDX_SiO2]  = 0.06008
        mw[IDX_C2S]   = 0.17224
        mw[IDX_C3S]   = 0.22832
        mw[IDX_CO2]   = 0.04401
        mw[IDX_C3A]   = 0.27000
        mw[IDX_C4AF]  = 0.48600
        mw[IDX_Al2O3] = 0.10200
        mw[IDX_Fe2O3] = 0.16000

        X_new = mass_drift_correction(
            X_new,
            mw=mw,
            strength=0.7
        )

        # -----------------------------
        # 6. BOUNDARY CONDITIONS
        # -----------------------------
        X_new[0, IDX_T_S] = 300.0
        X_new[-1, IDX_T_G] = 1200.0

        X_new[0, SOLID_SPECIES] = X[0, SOLID_SPECIES]

        # -----------------------------
        # 7. STABILIZATION
        # -----------------------------
        X_new[:, SOLID_SPECIES] = np.maximum(X_new[:, SOLID_SPECIES], 0.0)
        X_new[:, GAS_SPECIES] = np.maximum(X_new[:, GAS_SPECIES], 0.0)

        X_new[:, THERMAL_STATES] = np.clip(
            X_new[:, THERMAL_STATES],
            250.0,
            3500.0
        )

        # -----------------------------
        # 8. STATE UPDATE
        # -----------------------------
        self.X = X_new
        self.t += dt

        return self.X