import numpy as np


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

        # ================= SAFETY =================
        self.eps = 1e-9
        self.T_MIN = 200.0
        self.T_MAX = 2500.0

        # inlet smoothing (CRITICAL FOR STABILITY)
        self.alpha_in = 0.7
        self.Tg_in_mem = None
        self.Ts_in_mem = None

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, inputs, dt):

        Tg_n = Tg.copy()
        Ts_n = Ts.copy()
        Tw_n = Tw.copy()

        # ======================================================
        # INLET (SELF-SUSTAINED STABILITY)
        # ======================================================
        Tg_out = Tg[-1]
        Ts_out = Ts[-1]

        if self.Tg_in_mem is None:
            self.Tg_in_mem = Tg_out
            self.Ts_in_mem = Ts_out

        Tg_in = self.alpha_in * Tg_out + (1 - self.alpha_in) * self.Tg_in_mem
        Ts_in = self.alpha_in * Ts_out + (1 - self.alpha_in) * self.Ts_in_mem

        self.Tg_in_mem = Tg_in
        self.Ts_in_mem = Ts_in

        # ======================================================
        # GRADIENTS
        # ======================================================
        dTg_dz = np.zeros_like(Tg)
        dTs_dz = np.zeros_like(Ts)

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = (Tg[0] - Tg_in) / self.dz
        dTs_dz[0] = (Ts[0] - Ts_in) / self.dz

        dTg_dz = np.clip(dTg_dz, -150, 150)
        dTs_dz = np.clip(dTs_dz, -150, 150)

        # ======================================================
        # HEAT TRANSFER (SMOOTHED)
        # ======================================================
        dT_gs = np.clip(Tg - Ts, -600, 600)
        dT_gw = np.clip(Tg - Tw, -600, 600)
        dT_ws = np.clip(Ts - Tw, -600, 600)

        q_gs = 0.25 * self.hv_gs * self.a_gs * dT_gs
        q_gw = 0.25 * self.hv_gw * self.a_gw * dT_gw
        q_ws = 0.25 * self.hv_ws * self.a_ws * dT_ws

        # ======================================================
        # CAPACITIES
        # ======================================================
        C_g = max(self.rho_g * self.V_cell * self.Cp_g, self.eps)
        C_s = max(self.rho_s * self.V_cell * self.Cp_s, self.eps)
        C_w = max(self.rho_wall * self.V_cell * self.Cp_wall, self.eps)

        # ======================================================
        # GAS
        # ======================================================
        Tg_n = Tg + dt * (-self.u_g * dTg_dz + (q_gs - q_gw) / C_g)

        # ======================================================
        # SOLID (LOWER RESPONSE SPEED = REALISTIC)
        # ======================================================
        solid_gain = 0.35
        Ts_n = Ts + dt * (-self.u_s * dTs_dz + solid_gain * (q_gs - q_ws) / C_s)

        # ======================================================
        # WALL
        # ======================================================
        Tw_n = Tw + dt * ((q_gw + q_ws) / C_w)

        # ======================================================
        # CLAMP
        # ======================================================
        Tg_n = np.clip(Tg_n, self.T_MIN, self.T_MAX)
        Ts_n = np.clip(Ts_n, self.T_MIN, self.T_MAX)
        Tw_n = np.clip(Tw_n, self.T_MIN, 2000.0)

        # ======================================================
        # FEEDBACK ENERGY
        # ======================================================
        calcination_sink = np.mean(q_gs + q_gw)

        return Tg_n, Ts_n, Tw_n, calcination_sink


# ======================================================
# 🔥 STANDALONE DEBUG RUNNER
# ======================================================

if __name__ == "__main__":

    model = Calcination(N=80, L=60.0)

    # ================= INITIAL CONDITIONS =================
    Tg = np.ones(model.N) * (900 + 273.15)
    Ts = np.ones(model.N) * (850 + 273.15)
    Tw = np.ones(model.N) * (700 + 273.15)

    inputs = {}

    dt = 0.1
    t_end = 3600
    n_steps = int(t_end / dt)

    for i in range(n_steps):

        Tg, Ts, Tw, Qsink = model.thermal_step(Tg, Ts, Tw, inputs, dt)

        if i % 2000 == 0:
            print(
                f"step={i:06d} | "
                f"Tg_out={Tg[-1]-273.15:7.2f} °C | "
                f"Ts_mid={Ts[len(Ts)//2]-273.15:7.2f} °C | "
                f"Qsink={Qsink:10.2f}"
            )
