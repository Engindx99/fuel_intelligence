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
        self.LHV_RDF = 25e6

        self.eps = 1e-12

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, inputs, dt, burning_state=None):

        Tg_n = Tg.copy()
        Ts_n = Ts.copy()
        Tw_n = Tw.copy()

        # ======================================================
        # BURNING COUPLING
        # ======================================================
        if burning_state is not None:
            Tg_in = burning_state["Tg"][-1]
            Ts_in = burning_state["Ts"][-1]
        else:
            Tg_in = Tg[0]
            Ts_in = Ts[0]

        # ======================================================
        # TEMPERATURE GRADIENTS
        # ======================================================
        dTg_dz = np.zeros_like(Tg)
        dTs_dz = np.zeros_like(Ts)

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = (Tg[0] - Tg_in) / self.dz
        dTs_dz[0] = (Ts[0] - Ts_in) / self.dz

        # ======================================================
        # HEAT TRANSFER
        # ======================================================
        q_gs = (self.hv_gs * self.a_gs * (Tg - Ts)) / self.V_cell

        q_gw = (self.hv_gw * self.a_gw * (Tg - Tw)) / self.V_cell

        q_ws = (self.hv_ws * self.a_ws * (Ts - Tw)) / self.V_cell

        # ======================================================
        # GAS CAPACITY
        # ======================================================
        m_g = self.rho_g * self.V_cell
        C_g = m_g * self.Cp_g

        # ======================================================
        # GAS
        # ======================================================
        Tg_n = Tg + dt * (-self.u_g * dTg_dz + (q_gs - q_gw) / (C_g + self.eps))

        # ======================================================
        # SOLID
        # ======================================================
        Ts_n = Ts + dt * (-self.u_s * dTs_dz + (q_gs - q_ws))

        # ======================================================
        # WALL
        # ======================================================
        Tw_n = Tw + dt * (q_gw + q_ws)

        # ======================================================
        # ENERGY REMOVED FROM BURNING ZONE
        # ======================================================
        calcination_sink = np.mean(q_gs + q_gw) * self.V_cell

        return Tg_n, Ts_n, Tw_n, calcination_sink

    # ======================================================
    # TWIN INTERFACE
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

        # Burning zone tarafından bir sonraki adımda kullanılacak
        state.Calcination_Q_sink = Q_sink

        return state


# ======================================================
#  SIMULATION LOOP
# ======================================================

from Kiln.Burning import Burning

if __name__ == "__main__":

    # ================= MODELS =================
    burning_model = Burning(N=80, L=60.0)
    calcination_model = Calcination(N=80, L=60.0)

    # ================= INPUTS =================
    inputs = {
        "Fuel_rate": 5.0,
        "Petcoke": 0.6,
        "RDF_Fuel": 0.2,
        "O2": 3.5,
        "Feed_rate": 40000.0,
    }

    dt = 0.1
    t_end = 3600.0  # second
    n_steps = int(t_end / dt)

    # ======================================================
    #  INITIAL RUN OF BURNING (to generate first inlet)
    # ======================================================
    Tg_b = np.ones(burning_model.N) * (1500.0 + 273.15)
    Ts_b = np.ones(burning_model.N) * (1100.0 + 273.15)
    Tw_b = np.ones(burning_model.N) * (600.0 + 273.15)

    Tg_c = None
    Ts_c = None
    Tw_c = None

    t = 0.0

    for i in range(n_steps):

        # ======================================================
        #  BURNING STEP
        # ======================================================
        Tg_b, Ts_b, Tw_b = burning_model.thermal_step(Tg_b, Ts_b, Tw_b, inputs, dt)

        # ======================================================
        #  COUPLING (REAL OUTLET → INLET)
        # ======================================================
        Tg_in = Tg_b[-1]
        Ts_in = Ts_b[-1]
        Tw_in = Tw_b[-1]

        inputs["Tg_in"] = Tg_in
        inputs["Ts_in"] = Ts_in

        # ======================================================
        # ❄️ CALCINATION INITIALIZATION (ONLY ON FIRST STEP)
        # ======================================================
        if Tg_c is None:

            Tg_c = np.ones(calcination_model.N) * Tg_in
            Ts_c = np.ones(calcination_model.N) * Ts_in
            Tw_c = np.ones(calcination_model.N) * Tw_in

        # ======================================================
        # ❄️ CALCINATION STEP (OWN DYNAMICS)
        # ======================================================
        Tg_c, Ts_c, Tw_c = calcination_model.thermal_step(Tg_c, Ts_c, Tw_c, inputs, dt)

        t += dt

        # ======================================================
        # PRINT
        # ======================================================
        if i % 5000 == 0:
            print(
                f"step={i:06d} | "
                f"time={t/3600:.3f} h | "
                f"Tg_b_out={Tg_b[-1]-273.15:7.2f} °C | "
                f"Tg_c_mid={Tg_c[calcination_model.N//2]-273.15:7.2f} °C"
            )
