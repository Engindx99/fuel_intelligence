import numpy as np
from core.state import *
from core.physics import energy_terms_vec, mass_terms_vec
from core.flow import compute_velocities


class KilnSimulation:

    def __init__(self, L, N_cells, dt):
        self.L = L
        self.N = N_cells
        self.dt = dt
        self.dz = L / N_cells
        self.t = 0.0

        self.X = np.zeros((N_cells, N_STATES), dtype=np.float64)

    # -----------------------------
    def set_initial_condition(self, x0):
        self.X[:] = np.asarray(x0, dtype=np.float64)

    # -----------------------------
    def step(self, u):

        X = self.X
        N = self.N
        dz = self.dz
        dt = self.dt

        # -----------------------------
        # FLOW (single representative)
        # -----------------------------
        v_s, v_g = compute_velocities(X[N // 2], u)

        cfl = max(abs(v_s), abs(v_g)) * dt / dz
        if cfl > 0.8:
            dt = 0.5 * dz / (abs(v_s) + abs(v_g) + 1e-12)
            self.dt = dt

        # -----------------------------
        # ADVECTIVE FLUX (UPWIND CORRECTED)
        # -----------------------------
        dX = np.zeros_like(X)

        # solid: v_s > 0 → backward difference
        if v_s >= 0:
            dX[1:, :] -= v_s * (X[1:, :] - X[:-1, :]) / dz
        else:
            dX[:-1, :] -= v_s * (X[1:, :] - X[:-1, :]) / dz

        # gas: v_g < 0 → forward difference
        if v_g >= 0:
            dX[1:, :] -= v_g * (X[1:, :] - X[:-1, :]) / dz
        else:
            dX[:-1, :] -= v_g * (X[1:, :] - X[:-1, :]) / dz

        # -----------------------------
        # SOURCE TERMS (SEPARATED)
        # -----------------------------
        dE_s, dE_g = energy_terms_vec(X, u)
        dM = mass_terms_vec(X, u)

        dX[:, IDX_T_S] += dE_s
        dX[:, IDX_T_G] += dE_g

        dX += dM

        # -----------------------------
        # EXPLICIT STEP
        # -----------------------------
        X_new = X + dt * dX

        # -----------------------------
        # BOUNDARIES
        # -----------------------------
        X_new[0, IDX_T_S] = 300.0
        X_new[-1, IDX_T_G] = 1200.0

        # -----------------------------
        # STABILIZATION
        # -----------------------------
        X_new[:, THERMAL_STATES] = np.clip(X_new[:, THERMAL_STATES], 250.0, 3500.0)
        X_new[:, SOLID_SPECIES] = np.maximum(X_new[:, SOLID_SPECIES], 0.0)

        self.X = X_new
        self.t += dt

        return self.X