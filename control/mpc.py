import casadi as ca
import numpy as np
from Kiln.Burning import Burning


class MasterMPC:

    def __init__(self):

        self.Np = 20
        self.Nc = 5
        self.dt = 0.05

        self.Tg_ref = 1823.15  # K
        self.Ts_ref = 1723.15  # K

        self.model = Burning(N=5)

        self.opti = ca.Opti()
        self.prev_sol = None

        # ================= LOG CONTROL =================
        self.log_interval = 60.0  # 1 min
        self.last_log_time = 0.0

        self._build()

    # ======================================================
    def _project_state(self, x, N_target):

        x = np.asarray(x).flatten()
        N_src = x.shape[0]

        if N_src == N_target:
            return x

        src = np.linspace(0, 1, N_src)
        tgt = np.linspace(0, 1, N_target)

        return np.interp(tgt, src, x)

    # ======================================================
    def _log(self, t, state, fuel):

        if t - self.last_log_time < self.log_interval:
            return

        self.last_log_time = t

        idx = len(state.Tg_burning) // 2

        print(
            f"[MPC] t={t/60:.2f} min | "
            f"Fuel={fuel:.3f} | "
            f"Tg={state.Tg_burning[idx]:.2f} K | "
            f"Ts={state.Ts_burning[idx]:.2f} K"
        )

    # ======================================================
    def _build(self):

        N = self.model.N
        eps = self.model.eps

        self.Tg = self.opti.variable(N, self.Np + 1)
        self.Ts = self.opti.variable(N, self.Np + 1)
        self.Tw = self.opti.variable(N, self.Np + 1)

        self.Fuel = self.opti.variable(self.Nc)

        self.Tg0 = self.opti.parameter(N)
        self.Ts0 = self.opti.parameter(N)
        self.Tw0 = self.opti.parameter(N)

        self.opti.subject_to(self.Tg[:, 0] == self.Tg0)
        self.opti.subject_to(self.Ts[:, 0] == self.Ts0)
        self.opti.subject_to(self.Tw[:, 0] == self.Tw0)

        self.opti.set_initial(self.Fuel, 4.0)

        cost = 0

        def u(k):
            return min(k, self.Nc - 1)

        # ================= MIX =================
        p0, a0, l0 = 0.6, 0.2, 0.2
        norm = p0 + a0 + l0 + eps
        p0, a0, l0 = p0 / norm, a0 / norm, l0 / norm

        O2 = 3.5
        eta = ca.exp(-((O2 - self.model.O2_opt) ** 2) / self.model.O2_sigma2)

        LHV_mix = (
            p0 * self.model.LHV_petcoke
            + l0 * self.model.LHV_lignite
            + a0 * self.model.LHV_RDF
        )

        # ======================================================
        for k in range(self.Np):

            uk = u(k)
            Fuel_k = self.Fuel[uk]
            Fuel_k = ca.fmax(2.0, ca.fmin(6.0, Fuel_k))

            Q_in = Fuel_k * LHV_mix * eta
            q_vol = Q_in / (self.model.V_total + eps)

            dTg_dz = ca.vertcat(
                self.Tg[0, k], (self.Tg[1:, k] - self.Tg[:-1, k]) / self.model.dz
            )

            dTs_dz = ca.vertcat(
                self.Ts[0, k], (self.Ts[1:, k] - self.Ts[:-1, k]) / self.model.dz
            )

            q_gs = (
                self.model.hv_gs * self.model.a_gs * (self.Tg[:, k] - self.Ts[:, k])
            ) / self.model.V_cell

            q_gw = (
                self.model.hv_gw * self.model.a_gw * (self.Tg[:, k] - self.Tw[:, k])
            ) / self.model.V_cell

            q_ws = (
                self.model.hv_ws * self.model.a_ws * (self.Ts[:, k] - self.Tw[:, k])
            ) / self.model.V_cell

            C_g = self.model.rho_g * self.model.V_cell * self.model.Cp_g
            C_s = self.model.rho_s * self.model.V_cell * self.model.Cp_s
            C_w = self.model.rho_wall * self.model.V_wall * self.model.Cp_wall

            q_loss = (
                self.model.h_ext
                * self.model.A_wall
                * (self.Tw[:, k] - self.model.T_amb)
            ) / (self.model.V_cell + eps)

            Tg_next = self.Tg[:, k] + self.dt * (
                -self.model.u_g * dTg_dz + (q_vol - q_gs - q_gw) / C_g
            )

            Ts_next = self.Ts[:, k] + self.dt * (
                -self.model.u_s * dTs_dz + (q_gs - q_ws) / C_s
            )

            Tw_next = self.Tw[:, k] + self.dt * ((q_gw + q_ws - q_loss) / C_w)

            self.opti.subject_to(self.Tg[:, k + 1] == Tg_next)
            self.opti.subject_to(self.Ts[:, k + 1] == Ts_next)
            self.opti.subject_to(self.Tw[:, k + 1] == Tw_next)

            Tg_mean = ca.sum1(self.Tg[:, k]) / N
            Ts_mean = ca.sum1(self.Ts[:, k]) / N

            cost += 1e-3 * (Tg_mean - self.Tg_ref) ** 2
            cost += 1e-3 * (Ts_mean - self.Ts_ref) ** 2
            cost += 1e-2 * Fuel_k**2

        self.opti.minimize(cost)

        self.opti.subject_to(self.Fuel >= 2.0)
        self.opti.subject_to(self.Fuel <= 6.0)

        self.opti.solver(
            "ipopt",
            {
                "ipopt.max_iter": 80,
                "ipopt.tol": 1e-3,
                "print_time": 0,
                "ipopt.print_level": 0,
            },
        )

    # ======================================================
    def compute_control(self, state, t=0.0):

        Tg0 = self._project_state(state.Tg_burning, self.model.N)
        Ts0 = self._project_state(state.Ts_burning, self.model.N)
        Tw0 = self._project_state(state.Tw_burning, self.model.N)

        self.opti.set_value(self.Tg0, Tg0)
        self.opti.set_value(self.Ts0, Ts0)
        self.opti.set_value(self.Tw0, Tw0)

        if self.prev_sol is not None:
            try:
                for k in range(self.Nc):
                    self.opti.set_initial(
                        self.Fuel[k], self.prev_sol.value(self.Fuel[k])
                    )
            except:
                pass

        try:
            sol = self.opti.solve()
            self.prev_sol = sol
            fuel = float(sol.value(self.Fuel[0]))

        except Exception as e:
            print("MPC FAILED → fallback:", repr(e))
            fuel = 5.0

        self._log(t, state, fuel)

        return {"Fuel_rate": fuel}


if __name__ == "__main__":

    import numpy as np
    import os
    import json
    import pickle
    from Kiln.Burning import Burning

    # ======================================================
    # CHECKPOINT HELPERS
    # ======================================================
    def save_ckpt(path, state):
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_ckpt(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ======================================================
    # MODELS
    # ======================================================
    plant_model = Burning(N=5)
    mpc_controller = MasterMPC()

    # ======================================================
    # INITIAL INPUTS
    # ======================================================
    plant_inputs = {
        "Petcoke": 0.6,
        "RDF_Fuel": 0.2,
        "O2": 3.5,
        "Fuel_rate": 5.5,
    }

    dt = 0.05

    # ======================================================
    # TIME STRUCTURE
    # ======================================================
    chunk_hours = 1.0
    chunk_time = chunk_hours * 3600
    steps_per_chunk = int(chunk_time / dt)

    total_hours = 6.0
    n_chunks = int(total_hours / chunk_hours)

    ckpt_path = "control/mpc_ckpt.pkl"
    log_path = "control/mpc_status.jsonl"

    # ======================================================
    # INIT STATE
    # ======================================================
    if os.path.exists(ckpt_path):
        st = load_ckpt(ckpt_path)
        Tg, Ts, Tw = st["Tg"], st["Ts"], st["Tw"]
        start_chunk = int(st["t"] // chunk_time)
        print(f"[RESUME] chunk={start_chunk}")
    else:
        Tg = np.ones(5) * 1773.15
        Ts = np.ones(5) * 1673.15
        Tw = np.ones(5) * 873.15
        start_chunk = 0

    outlet = -1

    # ======================================================
    # MAIN LOOP
    # ======================================================
    for chunk in range(start_chunk, n_chunks):

        t_local = 0.0
        next_log = 0.0

        for i in range(steps_per_chunk):

            # ================= MPC =================
            state_obj = type("S", (), {})()
            state_obj.Tg_burning = Tg
            state_obj.Ts_burning = Ts
            state_obj.Tw_burning = Tw

            ctrl = mpc_controller.compute_control(
                state_obj, t=chunk * chunk_time + t_local
            )

            plant_inputs["Fuel_rate"] = ctrl["Fuel_rate"]

            # ================= PLANT =================
            Tg, Ts, Tw = plant_model.thermal_step(Tg, Ts, Tw, plant_inputs, dt)

            t_local += dt

            # ================= LOG (10 min) =================
            if t_local >= next_log:

                record = {
                    "chunk": chunk,
                    "time_h": f"{(chunk * chunk_time + t_local)/3600.0:.4f}",
                    "Tg": float(Tg[outlet]),
                    "Ts": float(Ts[outlet]),
                    "Tw": float(Tw[outlet]),
                    "Fuel": float(plant_inputs["Fuel_rate"]),
                }

                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                next_log += 600.0

        # ================= CHECKPOINT =================
        global_t = (chunk + 1) * chunk_time

        save_ckpt(ckpt_path, {"Tg": Tg, "Ts": Ts, "Tw": Tw, "t": global_t})

        print(
            f"[CHUNK {chunk}] "
            f"t={global_t/3600:.2f} h | "
            f"Tg={Tg[outlet]:.2f} | "
            f"Ts={Ts[outlet]:.2f} | "
            f"Tw={Tw[outlet]:.2f}"
        )
