import numpy as np
from Kiln.Burning import Burning


class Calcination:

    def __init__(self, N=80, L=60.0):

        self.N = N
        self.L = L

        # ================= ENERGY STORAGE =================
        self.rho_g = 4.2
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # effective heat capacity (lumped system)
        self.C_total = (
            self.rho_g * self.Cp_g
            + self.rho_s * self.Cp_s
            + self.rho_wall * self.Cp_wall
        )

        # ================= COUPLING =================
        self.alpha_in = 0.7
        self.T_in_mem = None

        # ================= STABILITY =================
        self.eps = 1e-9

    # ======================================================
    # ENERGY <-> TEMPERATURE (GLOBAL)
    # ======================================================
    def to_T(self, E):
        return E / (self.C_total + self.eps)

    def to_E(self, T):
        return T * self.C_total

    # ======================================================
    # MAIN STEP (SIMPLE ENERGY BALANCE)
    # ======================================================

    def thermal_step(self, Eg, Es, Ew, dt, burning_state=None, inputs=None):

        # ================= SAFETY =================
        if inputs is None:
            inputs = {}

        Q_in = inputs.get("Q_in", 0.0)

        Q_burn_used = inputs.get("Q_burning_consumed", 0.0)

        Q_net = Q_in - Q_burn_used

        Q_net = np.clip(Q_net, -5e7, 5e7)

        E_new = Q_net * dt

        return E_new, E_new, E_new

    def apply(self, state, inputs, dt):

        if inputs is None:
            inputs = {}

        Eg, Es, Ew = self.thermal_step(
            state.Tg_calcination,
            state.Ts_calcination,
            state.Tw_calcination,
            dt,
            burning_state=None,
            inputs=inputs,
        )

        state.E_calcination = Eg

        state.T_calcination = self.to_T(Eg)

        return state


if __name__ == "__main__":

    burning_model = Burning(N=80, L=60.0)
    calcination_model = Calcination(N=80, L=60.0)

    dt = 0.1
    t_end = 3600.0
    n_steps = int(t_end / dt)

    Tg_b = np.ones(burning_model.N) * (1500.0 + 273.15)
    Ts_b = np.ones(burning_model.N) * (1100.0 + 273.15)
    Tw_b = np.ones(burning_model.N) * (600.0 + 273.15)

    # ================= ENERGY STATE =================
    Eg_c = np.zeros(calcination_model.N)
    Es_c = np.zeros(calcination_model.N)
    Ew_c = np.zeros(calcination_model.N)

    # init temperature buffers (IMPORTANT)
    Tg_c, Ts_c, Tw_c = calcination_model.to_T(Eg_c, Es_c, Ew_c)

    t = 0.0

    for i in range(n_steps):

        # ================= BURNING =================
        Tg_b, Ts_b, Tw_b = burning_model.thermal_step(
            Tg_b,
            Ts_b,
            Tw_b,
            {"Fuel_rate": 5.0, "Petcoke": 0.6, "RDF_Fuel": 0.2, "O2": 3.5},
            dt,
        )

        # ================= ONLY ENERGY TRANSFER =================
        Tg_out = Tg_b[-1]

        # single source term (NO double integration)
        Q_in = max(Tg_out - 800.0, 0.0) * 1e4

        # distribute energy (stable normalization)
        Eg_c += (Q_in * dt) / calcination_model.N

        # ================= CALCINATION =================
        Eg_c, Es_c, Ew_c = calcination_model.thermal_step(
            Eg_c,
            Es_c,
            Ew_c,
            dt,
            burning_state={"Tg": Tg_b, "Ts": Ts_b},
        )

        # ================= POST PROCESS =================
        Tg_c, Ts_c, Tw_c = calcination_model.to_T(Eg_c, Es_c, Ew_c)

        t += dt

        if i % 5000 == 0:
            print(
                f"step={i:06d} | "
                f"time={t/3600:.3f} h | "
                f"Tg_b_out={Tg_b[-1]-273.15:7.2f} °C | "
                f"Tg_c_mid={Tg_c[len(Tg_c)//2]-273.15:7.2f} °C | "
            )
