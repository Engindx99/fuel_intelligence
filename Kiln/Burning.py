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

        # ================= INTERFACIAL AREA MODEL =================
        self.epsilon_bed = 0.35
        self.a_gs = 6.0 * (1.0 - self.epsilon_bed) / self.D
        self.a_gw = 2.0 * np.pi * self.D
        self.a_ws = self.a_gs * 0.6

        # ================= PROPERTIES =================
        self.rho_g = 4.2
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ================= VELOCITIES =================
        self.u_g = 2.5
        self.u_s = 0.02

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

        # ================= FEED DYNAMICS =================
        self.feed = 40000.0
        self.feed_target = 40000.0
        self.feed_tau = 7200.0

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
    def step(self, Tg, Ts, Tw, inputs, dt):

        Tg_n = Tg.copy()
        Ts_n = Ts.copy()
        Tw_n = Tw.copy()

        dTg_dz = self.d_dz(Tg)
        dTs_dz = self.d_dz(Ts)

        # ================= FEED =================
        self.feed_target = inputs.get("Feed_rate", self.feed_target)
        self.feed += (self.feed_target - self.feed) * dt / self.feed_tau

        # ================= FUEL =================
        p = inputs.get("Petcoke", 0.6)
        a = inputs.get("Alternative_Fuel", 0.2)
        l = max(1.0 - p - a, 0.0)

        norm = p + a + l + self.eps
        p /= norm
        a /= norm
        l /= norm

        fuel_rate = inputs.get("Fuel_rate", 1.0)
        O2 = inputs.get("O2", 3.5)

        eta = self.combustion_efficiency(O2)

        LHV_mix = p * self.LHV_petcoke + l * self.LHV_lignite + a * self.LHV_alt

        # ================= ENERGY INPUT =================
        Q_in = fuel_rate * LHV_mix * eta
        q_vol = Q_in / (self.V_total + self.eps)

        # ======================================================
        for i in range(1, self.N):

            # ================= HEAT TRANSFER =================
            q_gs = self.hv_gs * self.a_gs * (Tg[i] - Ts[i])
            q_gw = self.hv_gw * self.a_gw * (Tg[i] - Tw[i])
            q_ws = self.hv_ws * self.a_ws * (Ts[i] - Tw[i])

            # ================= MASS =================
            m_g = self.rho_g * self.V_cell
            m_s = self.rho_s * self.V_cell

            # ================= CAPACITIES =================
            C_g = m_g * self.Cp_g
            C_w = self.rho_wall * self.Cp_wall * self.V_cell

            # ================= SOLID EFFECTIVE CAPACITY (CALIBRATABLE PHYSICS)
            # ======================================================

            # thermal diffusivity
            alpha_s = self.hv_gs * self.a_gs / (self.rho_s * self.Cp_s + self.eps)

            # flow residence time
            tau_flow = self.dz / max(self.u_s, 1e-9)

            # thermal penetration depth
            delta_T = np.sqrt(alpha_s * tau_flow)

            # active volume (geometric limit)
            V_active = self.a_gs * delta_T

            # physical clamp
            V_cell_eff = min(self.V_cell, V_active)

            phi_coupling = 1e-3

            # effective solid capacity
            C_s = phi_coupling * self.rho_s * V_cell_eff * self.Cp_s

            # ======================================================
            # GAS ENERGY BALANCE
            # ======================================================
            Tg_n[i] = Tg[i] + dt * (
                -self.u_g * dTg_dz[i] + (q_vol - q_gs - q_gw) / (C_g + self.eps)
            )

            # ======================================================
            # SOLID ENERGY BALANCE
            # ======================================================
            Ts_n[i] = Ts[i] + dt * (
                -self.u_s * dTs_dz[i] + (q_gs - q_ws) / (C_s + self.eps)
            )

            # ======================================================
            # WALL ENERGY BALANCE
            # ======================================================
            h_ext, T_amb = 18.0, 300.0

            C_w = self.rho_wall * self.Cp_wall * self.a_gw * 0.1
            A_wall_cell = self.a_gw * self.dz

            Q_loss = h_ext * A_wall_cell * (Tw[i] - T_amb)

            Tw_n[i] = Tw[i] + dt * ((q_gw + q_ws - Q_loss) / (C_w + self.eps))

        return Tg_n, Ts_n, Tw_n


if __name__ == "__main__":

    model = KilnPDE(N=50)

    Tg = np.ones(50) * (1500.0 + 273.15)
    Ts = np.ones(50) * (1400.0 + 273.15)
    Tw = np.ones(50) * (1200.0 + 273.15)

    inputs = {
        "Fuel_rate": 5.8,
        "Petcoke": 0.6,
        "Alternative_Fuel": 0.2,
        "O2": 3.5,
    }

    # ======================================================
    # TIME SCALE = HOURS
    # ======================================================
    dt = 1.0 / 3600.0  # 1 second in hours
    t_end = 6.0  # simulation hour

    n_steps = int(t_end / dt)

    t = 0.0

    history_Tg = []
    history_Ts = []
    history_Tw = []

    for i in range(n_steps):

        Tg, Ts, Tw = model.step(Tg, Ts, Tw, inputs, dt)

        t += dt

        history_Tg.append(Tg[25])
        history_Ts.append(Ts[25])
        history_Tw.append(Tw[25])

        if i % 1000 == 0:
            print(
                f"step={i:06d} | "
                f"time={t:.4f} h | "
                f"Tg={Tg[25]-273.15:7.2f} °C | "
                f"Ts={Ts[25]-273.15:7.2f} °C | "
                f"Tw={Tw[25]-273.15:7.2f} °C"
            )
