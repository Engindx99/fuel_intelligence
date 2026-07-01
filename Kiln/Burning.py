# ================================= TEMPERATURE =================================

import numpy as np


class Burning:

    def __init__(self, N=5, L=60.0):

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

        # ================= WALL =================
        self.A_wall = self.a_gw * self.L
        self.h_ext = 12.0

        self.T_amb = 300.0  # K

        self.V_wall = self.A_wall * 0.05

        # ================= PROPERTIES =================
        self.rho_g = 48.1
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
        self.LHV_RDF = 25e6

        self.O2_opt = 3.5
        self.O2_sigma2 = 25.0

        self.eps = 1e-9

    # ======================================================
    def combustion_efficiency(self, O2):
        return np.exp(-((O2 - self.O2_opt) ** 2) / self.O2_sigma2)

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, inputs, dt, calcination_sink=0.0):

        Tg_n = Tg
        Ts_n = Ts
        Tw_n = Tw

        # ================= GRADIENTS =================
        dTg_dz = np.zeros_like(Tg)
        dTs_dz = np.zeros_like(Ts)

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]

        # ================= FUEL MIX =================
        p = inputs.get("Petcoke", 0.6)
        a = inputs.get("RDF_Fuel", 0.2)
        l = max(1.0 - p - a, 0.0)

        norm = p + a + l + self.eps
        p, a, l = p / norm, a / norm, l / norm

        fuel_rate = inputs.get("Fuel_rate", 1.0)
        O2 = inputs.get("O2", 3.5)

        eta = self.combustion_efficiency(O2)

        LHV_mix = p * self.LHV_petcoke + l * self.LHV_lignite + a * self.LHV_RDF

        # ================= HEAT SOURCE =================
        Q_in = fuel_rate * LHV_mix * eta
        q_vol = Q_in / (self.V_total + self.eps)

        # ================= CALCINATION COUPLING =================
        sink_density = calcination_sink / (self.V_total + self.eps)
        q_vol = q_vol - 0.05 * sink_density

        # ================= HEAT TRANSFER =================
        q_gs = (self.hv_gs * self.a_gs * (Tg - Ts)) / self.V_cell
        q_gw = (self.hv_gw * self.a_gw * (Tg - Tw)) / self.V_cell
        q_ws = (self.hv_ws * self.a_ws * (Ts - Tw)) / self.V_cell

        # ================= SOLID EFFECTIVE CAPACITY =================
        alpha_s = self.hv_gs * self.a_gs / (self.rho_s * self.Cp_s + self.eps)
        tau_flow = self.dz / max(self.u_s, self.eps)

        delta_T = np.sqrt(max(alpha_s * tau_flow, 0.0))
        V_active = self.a_gs * delta_T

        V_cell_eff = min(self.V_cell, max(V_active, 0.01 * self.V_cell))

        phi_coupling = 1e-1
        C_s = phi_coupling * self.rho_s * V_cell_eff * self.Cp_s
        C_s = max(C_s, self.eps)

        # ================= GAS CAPACITY =================
        C_g = max(self.rho_g * self.V_cell * self.Cp_g, self.eps)

        # ================= WALL CAPACITY =================
        m_w = self.rho_wall * self.V_wall
        C_w = max(m_w * self.Cp_wall, self.eps)

        q_loss = (self.h_ext * self.A_wall * (Tw - self.T_amb)) / (
            self.V_cell + self.eps
        )

        # ================= DYNAMICS =================
        Tg_n = Tg + dt * (-self.u_g * dTg_dz + (q_vol - q_gs - q_gw) / C_g)

        Ts_n = Ts + dt * (-self.u_s * dTs_dz + (q_gs - q_ws) / C_s)

        Tw_n = Tw + dt * ((q_gw + q_ws - q_loss) / C_w)

        return Tg_n, Ts_n, Tw_n

    # ======================================================
    def apply(self, state, inputs, dt):

        Tg, Ts, Tw = self.thermal_step(
            state.Tg_burning,
            state.Ts_burning,
            state.Tw_burning,
            inputs,
            dt,
            calcination_sink=getattr(state, "Calcination_Q_sink", 0.0),
        )

        state.Tg_burning = Tg
        state.Ts_burning = Ts
        state.Tw_burning = Tw

        return state


# ======================================================
# TEST RUN
# ======================================================
if __name__ == "__main__":

    model = Burning(N=5)

    Tg = np.ones(5) * (1773.15)  # K
    Ts = np.ones(5) * (1673.15)  # K
    Tw = np.ones(5) * (873.15)  # K

    inputs = {
        "Fuel_rate": 5.5,
        "Petcoke": 0.6,
        "RDF_Fuel": 0.2,
        "O2": 3.5,
    }

    dt = 0.05
    t_end = 1 * 3600
    n_steps = int(t_end / dt)

    t = 0.0
    idx = 5 // 2

    for i in range(n_steps):

        Tg, Ts, Tw = model.thermal_step(Tg, Ts, Tw, inputs, dt)

        t += dt

        if i % 1000 == 0:
            print(
                f"step={i:06d} | "
                f"time={t/3600:.4f} h | "
                f"Tg={Tg[idx]:7.2f} °C | "
                f"Ts={Ts[idx]:7.2f} °C | "
                f"Tw={Tw[idx]:7.2f} °C"
            )
