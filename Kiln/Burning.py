import math
import numpy as np


class KilnPDE:

    def __init__(self, N=80, L=60.0):

        self.N = N
        self.L = L
        self.dz = L / N

        # physical properties
        self.rho_g = 1.2
        self.rho_s = 1600.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0

        # velocities
        self.u_g = 2.5
        self.u_s = 0.02

        # heat transfer (W/m²K → volumetric scaling assumed via dz)
        self.h_gs = 40.0
        self.h_gw = 8.0
        self.h_ws = 12.0

        # -------------------------
        # FUEL PROPERTIES
        # -------------------------
        self.LHV_petcoke = 32000.0
        self.LHV_lignite = 18000.0
        self.LHV_alt = 25000.0

        # combustion
        self.O2_opt = 3.5
        self.O2_sigma2 = 25.0

        # effective cross section (normalized 1D assumption)
        self.A = 1.0

    # -------------------------
    # combustion efficiency
    # -------------------------
    def combustion_efficiency(self, O2):
        return np.exp(-((O2 - self.O2_opt) ** 2) / self.O2_sigma2)

    # -------------------------
    # UPWIND
    # -------------------------
    def d_dz(self, T):
        dT = np.zeros_like(T)
        dT[1:] = (T[1:] - T[:-1]) / self.dz
        dT[0] = dT[1]
        return dT

    # -------------------------
    # STEP
    # -------------------------
    def step(self, Tg, Ts, Tw, inputs, dt):

        Tg_n = Tg.copy()
        Ts_n = Ts.copy()
        Tw_n = Tw.copy()

        dTg_dz = self.d_dz(Tg)
        dTs_dz = self.d_dz(Ts)

        # ======================================================
        # FUEL MIX
        # ======================================================
        p = inputs.get("Petcoke", 0.6)
        a = inputs.get("Alternative_Fuel", 0.2)
        l = max(1.0 - p - a, 0.0)

        norm = p + a + l + 1e-12
        p, a, l = p / norm, a / norm, l / norm

        fuel_rate = inputs.get("Fuel_rate", 1.0)
        O2 = inputs.get("O2", 3.5)

        eta = self.combustion_efficiency(O2)

        LHV_mix = p * self.LHV_petcoke + l * self.LHV_lignite + a * self.LHV_alt

        # ======================================================
        # TOTAL FUEL POWER (kJ/s)
        # ======================================================
        Q_total = fuel_rate * LHV_mix * eta

        # distribute over kiln LENGTH (IMPORTANT FIX)
        q_fuel = Q_total / (self.N * self.dz * self.rho_s)  # kJ/m³·s

        for i in range(1, self.N):

            # -------------------------
            # HEAT TRANSFER (VOL. FORM)
            # -------------------------
            Q_gs = self.h_gs * (Ts[i] - Tg[i])
            Q_gw = self.h_gw * (Tw[i] - Tg[i])
            Q_ws = self.h_ws * (Tg[i] - Ts[i])

            # -------------------------
            # GAS PDE
            # -------------------------
            Tg_n[i] = Tg[i] + dt * (
                -self.u_g * dTg_dz[i]
                + (Q_gs + Q_gw + q_fuel) / (self.rho_g * self.Cp_g)
            )

            # -------------------------
            # SOLID PDE
            # -------------------------
            Ts_n[i] = Ts[i] + dt * (
                -self.u_s * dTs_dz[i] + (-Q_gs + Q_ws) / (self.rho_s * self.Cp_s)
            )

            # -------------------------
            # WALL DYNAMICS
            # -------------------------
            Tw_n[i] = Tw[i] + dt * ((Q_gw - Q_ws) / 2.5e6)

        return Tg_n, Ts_n, Tw_n


# ======================================================
# SIMULATION
# ======================================================
if __name__ == "__main__":

    model = KilnPDE(N=50)

    Ts = np.ones(50) * 1400.0
    Tg = np.ones(50) * 1500.0
    Tw = np.ones(50) * 1200.0

    inputs = {"Fuel_rate": 3.0, "Petcoke": 0.6, "Alternative_Fuel": 0.2, "O2": 3.5}

    dt = 0.1

    for i in range(10):

        Tg, Ts, Tw = model.step(Tg, Ts, Tw, inputs, dt)

        print(
            f"step={i} | "
            f"Ts_center={Ts[25]:.2f} | "
            f"Tg_center={Tg[25]:.2f} | "
            f"Tw_center={Tw[25]:.2f}"
        )
