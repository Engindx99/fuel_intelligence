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
        self.X[:] = np.asarray(x0, dtype=np.float64)

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

        # -----------------------------
        # 2. OPTİMİZE ADVECTION (Daha Az Kopya)
        # -----------------------------
        dX = np.zeros_like(X)
        
        # Upwind şeması: Fiziksel yöne göre farkları hesapla
        diff = (X[1:, :] - X[:-1, :]) / dz
        
        if v_s >= 0:
            dX[1:, :] -= v_s * diff
        else:
            dX[:-1, :] -= v_s * diff

        if v_g <= 0:
            dX[:-1, :] -= v_g * diff
        else:
            dX[1:, :] -= v_g * diff

        # -----------------------------
        # 3. SOURCE TERMS (Vektörize)
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
        # 5. MASS DRIFT CORRECTION (Önceden hesaplanmış mw ile)
        # -----------------------------
        X_new = mass_drift_correction(
            X_new,
            mw=self.mw,
            strength=0.7
        )

        # -----------------------------
        # 6. BOUNDARY CONDITIONS (Fiziksel mühür)
        # -----------------------------
        X_new[0, IDX_T_S] = 300.0
        X_new[-1, IDX_T_G] = 1200.0 # Gaz giriş sıcaklığı
        
        # Besleme miktarını koru
        X_new[0, SOLID_SPECIES] = X[0, SOLID_SPECIES]

        # -----------------------------
        # 7. STABILIZATION (In-place clip kullanarak hızlanma)
        # -----------------------------
        # Negatif kütle oluşumunu önle
        np.clip(X_new[:, SOLID_SPECIES], 0.0, None, out=X_new[:, SOLID_SPECIES])
        np.clip(X_new[:, GAS_SPECIES], 0.0, None, out=X_new[:, GAS_SPECIES])
        
        # Termal patlamaları önle
        np.clip(X_new[:, THERMAL_STATES], 250.0, 3500.0, out=X_new[:, THERMAL_STATES])

        # -----------------------------
        # 8. STATE UPDATE
        # -----------------------------
        self.X = X_new
        self.t += dt

        return self.X