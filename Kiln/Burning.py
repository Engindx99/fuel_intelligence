# ================================= TEMPERATURE =================================
import numpy as np

print("BURNING FILE:", __file__)


class Burning:

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

        # ================= WALL =================
        self.A_wall = self.a_gw * self.L
        self.h_ext = 3.0
        self.T_amb = 300.0
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

        # ================= RADIATION =================
        self.sigma_rad = 5.67e-8
        self.epsilon_rad = 0.85

        self.eps = 1e-9

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, inputs, dt):

        Tg_n = Tg.copy()
        Ts_n = Ts.copy()
        Tw_n = Tw.copy()

        # ================= GRADIENTS =================
        dTg_dz = np.zeros_like(Tg)
        dTs_dz = np.zeros_like(Ts)

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]

        # ======================================================
        # ENERGY INPUT (ONLY SOURCE)
        # ======================================================
        Q_burning = inputs.get("Q_burning", 0.0)

        q_gas = 8.5e4

        # ✔ physical injection ONLY to gas phase

        # ======================================================
        # HEAT TRANSFER
        # ======================================================
        q_gs = (self.hv_gs * self.a_gs * (Tg - Ts)) / self.V_cell
        q_gw = (self.hv_gw * self.a_gw * (Tg - Tw)) / self.V_cell
        q_ws = (self.hv_ws * self.a_ws * (Ts - Tw)) / self.V_cell

        # ================= RADIATION =================
        q_rad = self.epsilon_rad * self.sigma_rad * (Tg**4 - Ts**4) / self.V_cell

        # ======================================================
        # EFFECTIVE SOLID CAPACITY
        # ======================================================

        alpha_s = self.hv_gs * self.a_gs / (self.rho_s * self.Cp_s + self.eps)
        tau_flow = self.dz / max(self.u_s, self.eps)

        delta_T = np.sqrt(max(alpha_s * tau_flow, 0.0))
        V_active = self.a_gs * delta_T

        V_cell_eff = min(self.V_cell, V_active)

        phi_coupling = 0.1
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

        # ======================================================
        # GAS BALANCE
        # ======================================================
        Tg_n = Tg + dt * (-self.u_g * dTg_dz + (q_gas - q_gs - q_gw - q_rad) / C_g)

        # ======================================================
        # SOLID BALANCE
        # ======================================================
        Ts_n = Ts + dt * (-self.u_s * dTs_dz + (q_gs + q_rad - q_ws) / C_s)

        # ======================================================
        # WALL BALANCE
        # ======================================================
        Tw_n = Tw + dt * ((q_gw + q_ws - q_loss) / C_w)

        return Tg_n, Ts_n, Tw_n

    # ======================================================
    def apply(self, state, inputs, dt):

        if inputs is None:
            inputs = {}

        Tg, Ts, Tw = self.thermal_step(
            state.Tg_burning,
            state.Ts_burning,
            state.Tw_burning,
            inputs,
            dt,
        )

        state.Tg_burning = Tg
        state.Ts_burning = Ts
        state.Tw_burning = Tw

        return state


# ======================================================
# TEST RUN
# ======================================================
if __name__ == "__main__":

    model = Burning(N=50)

    Tg = np.ones(50) * (1450.0 + 273.15)
    Ts = np.ones(50) * (1400.0 + 273.15)
    Tw = np.ones(50) * (1200.0 + 273.15)

    inputs = {
        "Fuel_rate": 5.0,
        "Petcoke": 0.6,
        "RDF_Fuel": 0.2,
        "O2": 3.5,
    }

    dt = 0.1
    t_end = 3600
    n_steps = int(t_end / dt)

    t = 0.0

    for i in range(n_steps):

        Tg, Ts, Tw = model.thermal_step(Tg, Ts, Tw, inputs, dt)

        t += dt

        if i % 1000 == 0:
            print(
                f"step={i:06d} | "
                f"time={t/3600:.4f} h | "
                f"Tg={Tg[25]-273.15:7.2f} °C | "
                f"Ts={Ts[25]-273.15:7.2f} °C | "
                f"Tw={Tw[25]-273.15:7.2f} °C"
            )


# ================================= REACTIONS =================================
