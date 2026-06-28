import numpy as np


class KilnPDE:

    def __init__(self, N=80, L=60.0):

        self.N = N
        self.L = L
        self.dz = L / N

        # ================= GEOMETRY =================
        self.D = 4.2
        self.A_cross = np.pi * self.D**2 / 4.0
        self.V_total = self.A_cross * self.L
        self.V_cell = self.V_total / self.N

        # ================= PROPERTIES =================
        self.rho_g = 1.2
        self.rho_s = 1600.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ================= VELOCITIES =================
        self.u_g = 2.5
        self.u_s = 0.02

        # holdup transport
        self.u_phi = 0.02

        # ================= HEAT TRANSFER =================
        self.hv_gs = 900.0
        self.hv_gw = 250.0
        self.hv_ws = 300.0

        # ================= FUEL =================
        self.LHV_petcoke = 32e6
        self.LHV_lignite = 18e6
        self.LHV_alt = 25e6

        self.O2_opt = 3.5
        self.O2_sigma2 = 25.0

        self.eps = 1e-12

    # ======================================================
    def combustion_efficiency(self, O2):
        return np.exp(-((O2 - self.O2_opt) ** 2) / self.O2_sigma2)

    # ======================================================
    def d_dz(self, X):
        dX = np.zeros_like(X)
        dX[1:] = (X[1:] - X[:-1]) / self.dz
        dX[0] = dX[1]
        return dX

    # ======================================================
    def step(self, Tg, Ts, Tw, phi, inputs, dt):

        Tg_n = Tg.copy()
        Ts_n = Ts.copy()
        Tw_n = Tw.copy()
        phi_n = phi.copy()

        dTg_dz = self.d_dz(Tg)
        dTs_dz = self.d_dz(Ts)
        dphi_dz = self.d_dz(phi)

        # ================= FUEL =================
        p = inputs.get("Petcoke", 0.6)
        a = inputs.get("Alternative_Fuel", 0.2)
        l = max(1.0 - p - a, 0.0)

        norm = p + a + l + self.eps
        p, a, l = p / norm, a / norm, l / norm

        fuel_rate = inputs.get("Fuel_rate", 1.0)
        O2 = inputs.get("O2", 3.5)

        eta = self.combustion_efficiency(O2)

        LHV_mix = p * self.LHV_petcoke + l * self.LHV_lignite + a * self.LHV_alt

        # ================= ENERGY INPUT =================
        Q_in_total = fuel_rate * LHV_mix * eta
        q_base = Q_in_total / (self.V_total + self.eps)

        # ================= MAIN LOOP =================
        for i in range(1, self.N):

            # ---------------- HEAT TRANSFER ----------------
            q_gs = self.hv_gs * (Ts[i] - Tg[i])
            q_gw = self.hv_gw * (Tw[i] - Tg[i])
            q_ws = self.hv_ws * (Tg[i] - Ts[i])

            # ---------------- HOLDDUP ----------------
            phi_i = np.clip(phi[i], 0.05, 1.0)

            # ================= EFFECTIVE THERMAL MASS =================

            m_s = self.rho_s * self.V_cell * phi_i
            m_g = self.rho_g * self.V_cell

            inertia_g = m_g * self.Cp_g

            solid_gain = 1450 + 4000.0 * (1.0 - phi_i)
            Cp_s_eff = self.Cp_s * (1.0 + 2.5 * (1.0 - phi_i))
            inertia_s = (m_s * self.Cp_s) / solid_gain

            # ================= FUEL SENSITIVITY =================
            # transport + packing coupling
            transport_factor = 0.55 + 0.45 * phi_i

            q_f = q_base * transport_factor

            # extra solid coupling (IMPORTANT FIX)
            q_f_s = (2.0 + 3.0 * (1.0 - phi_i)) * q_f

            # ================= GAS =================
            Tg_n[i] = Tg[i] + dt * (
                -self.u_g * dTg_dz[i] + (q_gs + q_gw + q_f) / (inertia_g + self.eps)
            )

            # ================= SOLID =================
            Ts_n[i] = Ts[i] + dt * (
                -self.u_s * dTs_dz[i] + (-q_gs + q_ws + q_f_s) / (inertia_s + self.eps)
            )

            # ================= WALL =================
            Tw_n[i] = Tw[i] + dt * ((-q_gw - q_ws) / (self.rho_wall * self.Cp_wall))

            # ================= HOLDDUP DYNAMICS =================
            phi_n[i] = phi[i] + dt * (-self.u_phi * dphi_dz[i] + 0.01 * (1.0 - phi[i]))

            phi_n[i] = np.clip(phi_n[i], 0.05, 1.0)

        return Tg_n, Ts_n, Tw_n, phi_n


if __name__ == "__main__":

    model = KilnPDE(N=50)

    Tg = np.ones(50) * 1500.0
    Ts = np.ones(50) * 1400.0
    Tw = np.ones(50) * 1200.0

    phi = np.ones(50) * 0.7

    inputs = {
        "Fuel_rate": 5.0,
        "Petcoke": 0.6,
        "Alternative_Fuel": 0.2,
        "O2": 3.5,
    }

    dt = 0.01

    history_Tg = []
    history_Ts = []
    history_phi = []

    for i in range(50):

        Tg, Ts, Tw, phi = model.step(Tg, Ts, Tw, phi, inputs, dt)

        # store history (CRITICAL FIX)
        history_Tg.append(Tg[25])
        history_Ts.append(Ts[25])
        history_phi.append(phi[25])

        print(
            f"step={i} | "
            f"Ts_center={Ts[25]:.3f} | "
            f"Tg_center={Tg[25]:.3f} | "
            f"phi_center={phi[25]:.3f}"
        )
