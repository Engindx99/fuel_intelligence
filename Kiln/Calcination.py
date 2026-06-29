import numpy as np
import matplotlib.pyplot as plt
from Kiln.Burning import Burning


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

        # ================= NUMERICAL SAFETY =================
        self.eps = 1e-9

        # ================= INPUT HEAT COUPLING =================
        self.k_fuel = 2.5e6  # effective scaling (tunable)

        # ================= INLET MEMORY =================
        self.alpha_in = 0.7
        self.Tg_in_mem = None
        self.Ts_in_mem = None

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, inputs, dt, burning_state=None):

        Tg_n = Tg.copy()
        Ts_n = Ts.copy()
        Tw_n = Tw.copy()

        # ======================================================
        # BURNING COUPLING
        # ======================================================
        if burning_state is not None:
            Tg_in_raw = burning_state["Tg"][-1]
            Ts_in_raw = burning_state["Ts"][-1]
        else:
            Tg_in_raw = Tg[0]
            Ts_in_raw = Ts[0]

        # smoothing inlet (stability)
        if self.Tg_in_mem is None:
            self.Tg_in_mem = Tg_in_raw
            self.Ts_in_mem = Ts_in_raw

        Tg_in = self.alpha_in * Tg_in_raw + (1 - self.alpha_in) * self.Tg_in_mem
        Ts_in = self.alpha_in * Ts_in_raw + (1 - self.alpha_in) * self.Ts_in_mem

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

        # soft limiter
        dTg_dz = np.tanh(dTg_dz / 120.0) * 120.0
        dTs_dz = np.tanh(dTs_dz / 120.0) * 120.0

        # ======================================================
        # HEAT TRANSFER (PHYSICAL SIGN FIXED)
        # ======================================================
        dT_gs = Tg - Ts
        dT_gw = Tg - Tw
        dT_ws = Ts - Tw

        # nonlinear damping
        dT_gs = dT_gs / (1.0 + np.abs(dT_gs) / 500.0)
        dT_gw = dT_gw / (1.0 + np.abs(dT_gw) / 500.0)
        dT_ws = dT_ws / (1.0 + np.abs(dT_ws) / 500.0)

        q_gs = self.hv_gs * self.a_gs * dT_gs
        q_gw = self.hv_gw * self.a_gw * dT_gw
        q_ws = self.hv_ws * self.a_ws * dT_ws

        # ======================================================
        # CAPACITIES
        # ======================================================
        C_g = self.rho_g * self.V_cell * self.Cp_g + self.eps
        C_s = self.rho_s * self.V_cell * self.Cp_s + self.eps
        C_w = self.rho_wall * self.V_cell * self.Cp_wall + self.eps

        # ======================================================
        # INPUT HEAT SOURCE (THIS WAS MISSING BEFORE)
        # ======================================================
        fuel_power = (
            inputs.get("Fuel_rate", 0.0)
            * self.k_fuel
            * (0.5 + inputs.get("O2", 0.0) / 10.0)
        )

        Q_in = fuel_power / self.N

        # ======================================================
        # GAS (loses + gains physically)
        # ======================================================
        Tg_n = Tg + dt * (-self.u_g * dTg_dz - (q_gs + q_gw) / C_g + Q_in / C_g)

        # ======================================================
        # SOLID
        # ======================================================
        Ts_n = Ts + dt * (-self.u_s * dTs_dz + (q_gs - q_ws) / C_s)

        # ======================================================
        # WALL
        # ======================================================
        Tw_n = Tw + dt * ((q_gw + q_ws) / C_w)

        # ======================================================
        # SINK (physically consistent energy loss)
        # ======================================================
        # energy removed per unit volume [W/m3]
        q_vol_sink = q_gs + q_gw

        # total power removed from gas phase
        calcination_sink = np.sum(q_vol_sink) * self.V_cell

        # stability constraint
        calcination_sink = max(calcination_sink, 0.0)

        return Tg_n, Ts_n, Tw_n, calcination_sink

    # ======================================================
    def apply(self, state, inputs, dt):

        burning_state = {
            "Tg": state.Tg_burning,
            "Ts": state.Ts_burning,
        }

        Tg, Ts, Tw, Q_sink = self.thermal_step(
            state.Tg_calcination,
            state.Ts_calcination,
            state.Tw_calcination,
            inputs,
            dt,
            burning_state=burning_state,
        )

        state.Tg_calcination = Tg
        state.Ts_calcination = Ts
        state.Tw_calcination = Tw
        state.Calcination_Q_sink = Q_sink

        return state


# ======================================================
# SIMULATION LOOP
# ======================================================

if __name__ == "__main__":

    burning_model = Burning(N=80, L=60.0)
    calcination_model = Calcination(N=80, L=60.0)

    inputs = {
        "Fuel_rate": 5.0,
        "Petcoke": 0.6,
        "RDF_Fuel": 0.2,
        "O2": 3.5,
        "Feed_rate": 40000.0,
    }

    dt = 0.1
    t_end = 3600.0
    n_steps = int(t_end / dt)

    Tg_b = np.ones(burning_model.N) * (1500.0 + 273.15)
    Ts_b = np.ones(burning_model.N) * (1100.0 + 273.15)
    Tw_b = np.ones(burning_model.N) * (600.0 + 273.15)

    Tg_c = None
    Ts_c = None
    Tw_c = None

    t = 0.0

    for i in range(n_steps):

        Tg_b, Ts_b, Tw_b = burning_model.thermal_step(Tg_b, Ts_b, Tw_b, inputs, dt)

        Tg_in = Tg_b[-1]
        Ts_in = Ts_b[-1]

        inputs["Tg_in"] = Tg_in
        inputs["Ts_in"] = Ts_in

        if Tg_c is None:
            Tg_c = np.ones(calcination_model.N) * Tg_in
            Ts_c = np.ones(calcination_model.N) * Ts_in
            Tw_c = np.ones(calcination_model.N) * Ts_in

        Tg_c, Ts_c, Tw_c = calcination_model.thermal_step(
            Tg_c, Ts_c, Tw_c, inputs, dt, burning_state={"Tg": Tg_b, "Ts": Ts_b}
        )

        t += dt

        if i % 5000 == 0:
            print(
                f"step={i:06d} | "
                f"time={t/3600:.3f} h | "
                f"Tg_b_out={Tg_b[-1]-273.15:7.2f} °C | "
                f"Tg_c_mid={Tg_c[len(Tg_c)//2]-273.15:7.2f} °C | "
                f"Qsink={Q_sink:10.2f}"
            )
