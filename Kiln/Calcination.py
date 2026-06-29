import numpy as np
import matplotlib.pyplot as plt


class Calcination:

    def __init__(self, N=80, L=60.0):

        self.N = N
        self.L = L
        self.dz = L / N

        # ================= GEOMETRY =================
        self.D = 4.2
        self.A_cross = np.pi * self.D**2 / 4.0
        self.V_total = self.A_cross * self.L
        self.V_cell = self.V_total / self.N

        # ================= INTERFACIAL AREA =================
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

        # ================= FEED =================
        self.feed = 40000.0
        self.feed_target = 40000.0
        self.feed_tau = 7200.0

    # ======================================================
    def combustion_efficiency(self, O2):
        return np.exp(-((O2 - self.O2_opt) ** 2) / self.O2_sigma2)

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, inputs, dt):

        Tg_n = Tg.copy()
        Ts_n = Ts.copy()
        Tw_n = Tw.copy()

        # ================= GRADIENT =================
        dTg_dz = np.zeros_like(Tg)
        dTs_dz = np.zeros_like(Ts)

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]

        # ================= FEED (SMOOTH) =================
        self.feed_target = inputs.get("Feed_rate", self.feed_target)
        self.feed += (self.feed_target - self.feed) * dt / self.feed_tau

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

        Q_in = fuel_rate * LHV_mix * eta
        q_vol = Q_in / self.V_total

        # ================= HEAT TRANSFER =================
        q_gs = (self.hv_gs * self.a_gs * (Tg - Ts)) / self.V_cell
        q_gw = (self.hv_gw * self.a_gw * (Tg - Tw)) / self.V_cell
        q_ws = (self.hv_ws * self.a_ws * (Ts - Tw)) / self.V_cell

        # ================= GAS CAPACITY =================
        m_g = self.rho_g * self.V_cell
        C_g = m_g * self.Cp_g

        # ================= CONVECTION =================
        conv = -self.u_g * dTg_dz

        # ================= TRUE DIFFUSION =================
        d2Tg_dz2 = np.zeros_like(Tg)
        d2Tg_dz2[1:-1] = (Tg[2:] - 2 * Tg[1:-1] + Tg[:-2]) / (self.dz**2)

        # Neumann boundary
        d2Tg_dz2[0] = d2Tg_dz2[1]
        d2Tg_dz2[-1] = d2Tg_dz2[-2]

        Pe = 8.0
        D_axial = self.u_g * self.dz / Pe

        q_disp = (self.rho_g * self.Cp_g) * D_axial * d2Tg_dz2

        # ================= RADIATION (LINEARIZED STABLE) =================
        sigma = 5.67e-8
        eps_rad = 0.45
        F_rad = 0.06

        T_ref = 1500.0 + 273.15
        A_gw = self.a_gw * self.dz

        q_rad_vol = (
            F_rad * eps_rad * 4.0 * sigma * (T_ref**3) * (Tg - Tw) * A_gw / self.V_cell
        )

        # ================= GAS UPDATE =================
        Tg_n = Tg + dt * (
            conv
            + (q_vol - q_gs - q_gw) / (C_g + self.eps)
            - q_rad_vol / (C_g + self.eps)
            + q_disp / (C_g + self.eps)
        )

        # ================= SOLID (NO REACTION) =================
        alpha_s = self.hv_gs * self.a_gs / (self.rho_s * self.Cp_s + self.eps)
        tau_flow = self.dz / (self.u_s + self.eps)

        delta_T = np.sqrt(alpha_s * tau_flow + self.eps)
        V_active = self.a_gs * delta_T

        V_cell_eff = self.V_cell / (1.0 + (self.V_cell / (V_active + self.eps)))

        phi_coupling = 1e-1
        C_s = phi_coupling * self.rho_s * V_cell_eff * self.Cp_s

        Ts_n = Ts + dt * (-self.u_s * dTs_dz + (q_gs - q_ws) / (C_s + self.eps))

        # ================= WALL =================
        h_ext = 18.0
        T_amb = 300.0

        A_wall_cell = np.pi * self.D * self.dz

        q_in_wall = q_gw + q_ws

        m_w = self.rho_wall * self.V_cell
        C_w = m_w * self.Cp_wall

        tau_w = C_w / (h_ext * A_wall_cell + self.eps)

        Tw_eq = T_amb + q_in_wall * self.V_cell / (h_ext * A_wall_cell + self.eps)

        Tw_n = Tw + (dt / (tau_w + self.eps)) * (Tw_eq - Tw)

        return Tg_n, Ts_n, Tw_n


if __name__ == "__main__":

    model = Calcination(N=80, L=60.0)

    # ================= INITIAL CONDITIONS (CALCINATION ZONE) =================

    Tg = np.ones(model.N) * (1200.0 + 273.15)  # gas entering from burning zone
    Ts = np.ones(model.N) * (950.0 + 273.15)  # solid decomposition active region
    Tw = np.ones(model.N) * (500.0 + 273.15)  # wall cooler than bed

    # ================= INPUTS =================
    inputs = {
        "Fuel_rate": 5.0,
        "Petcoke": 0.6,
        "Alternative_Fuel": 0.2,
        "O2": 3.5,
        "Feed_rate": 40000.0,
    }

    # ================= TIME SETTINGS =================
    dt = 0.1  # stable CFL regime
    t_end = 1.0 * 3600  #  1 hours (debug shorter first)

    n_steps = int(t_end / dt)

    # ================= MONITOR POINT =================
    mid = model.N // 2

    Tg_hist = []
    Ts_hist = []
    Tw_hist = []
    time_hist = []

    t = 0.0

    # ================= SIMULATION LOOP =================
    for i in range(n_steps):

        Tg, Ts, Tw = model.thermal_step(Tg, Ts, Tw, inputs, dt)

        t += dt

        # ================= STORE =================
        Tg_hist.append(Tg[mid] - 273.15)
        Ts_hist.append(Ts[mid] - 273.15)
        Tw_hist.append(Tw[mid] - 273.15)
        time_hist.append(t / 3600)

        # ================= PRINT =================
        if i % 5000 == 0:
            print(
                f"step={i:06d} | "
                f"time={t/3600:.3f} h | "
                f"Tg={Tg[mid]-273.15:7.2f} °C | "
                f"Ts={Ts[mid]-273.15:7.2f} °C | "
                f"Tw={Tw[mid]-273.15:7.2f} °C"
            )
