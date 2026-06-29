# ================================= TEMPERATURE =================================

import numpy as np


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
        self.LHV_RDF = 25e6

        self.O2_opt = 3.5
        self.O2_sigma2 = 25.0

        #  CRITICAL STABILITY PARAMETER
        self.eps = 1e-9  # 1e-12 çok küçük → blow-up yapar

    # ======================================================
    def combustion_efficiency(self, O2):
        return np.exp(-((O2 - self.O2_opt) ** 2) / self.O2_sigma2)

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, inputs, dt, calcination_sink=0.0):

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

        # ================= ENERGY SOURCE =================
        Q_in = fuel_rate * LHV_mix * eta

        #  IMPORTANT NORMALIZATION
        q_vol = Q_in / (self.V_total + self.eps)

        # ================= CALCINATION SINK =================
        # IMPORTANT: clamp negative energy removal
        q_sink = max(calcination_sink, 0.0)

        q_vol -= q_sink / (self.V_total + self.eps)

        # ================= HEAT TRANSFER =================
        q_gs = (self.hv_gs * self.a_gs * (Tg - Ts)) / self.V_cell
        q_gw = (self.hv_gw * self.a_gw * (Tg - Tw)) / self.V_cell
        q_ws = (self.hv_ws * self.a_ws * (Ts - Tw)) / self.V_cell

        # ================= CAPACITY =================
        m_g = self.rho_g * self.V_cell
        C_g = m_g * self.Cp_g

        # ================= NUMERICAL SAFETY =================
        C_g = max(C_g, self.eps)

        # ================= GAS UPDATE =================
        Tg_n = Tg + dt * (-self.u_g * dTg_dz + (q_vol - q_gs - q_gw) / C_g)

        # ================= SOLID UPDATE =================
        Ts_n = Ts + dt * (
            -self.u_s * dTs_dz + (q_gs - q_ws) / (self.rho_s * self.Cp_s + self.eps)
        )

        # ================= WALL =================
        Tw_n = Tw + dt * (q_gw + q_ws)

        # ================= CLAMP (CRITICAL) =================
        Tg_n = np.clip(Tg_n, 200.0, 2500.0)
        Ts_n = np.clip(Ts_n, 200.0, 2500.0)
        Tw_n = np.clip(Tw_n, 200.0, 2000.0)

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
    # TWIN COMPATIBILITY WRAPPER (CRITICAL)
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


if __name__ == "__main__":

    model = Burning(N=50)

    Tg = np.ones(50) * (1500.0 + 273.15)
    Ts = np.ones(50) * (1400.0 + 273.15)
    Tw = np.ones(50) * (1200.0 + 273.15)

    inputs = {
        "Fuel_rate": 5.0,
        "Petcoke": 0.6,
        "RDF_Fuel": 0.2,
        "O2": 3.5,
    }

    # ================= SI TIME =================
    dt = 0.1  #  second
    t_end = 1.0 * 3600  # second

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
