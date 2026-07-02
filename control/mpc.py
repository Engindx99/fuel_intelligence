from tracemalloc import start

import casadi as ca
import numpy as np
from datetime import datetime
from Kiln.Burning import Burning
import os
import json
import pickle
import yaml


class MasterMPC:

    def __init__(self, cfg):

        self.cfg = cfg

        self.Np = int(cfg["mpc"]["prediction_horizon"])
        self.Nc = int(cfg["mpc"]["control_horizon"])
        self.dt_model = float(cfg["mpc"]["dt_model"])
        self.dt_prediction = float(cfg["mpc"]["dt_prediction"])
        self.dt_control = float(cfg["mpc"]["dt_control"])

        self.n_substeps = int(self.dt_prediction / self.dt_model)

        self.last_control_time = -1e30
        self.last_control = 5.0

        self.fuel_min = float(cfg["mpc"]["fuel_min"])
        self.fuel_max = float(cfg["mpc"]["fuel_max"])

        self.Tg_ref = float(cfg["mpc"]["Tg_setpoint"])
        self.Ts_ref = float(cfg["mpc"]["Ts_setpoint"])

        self.w_T = float(cfg["mpc"].get("w_T", 1.0))
        self.w_F = float(cfg["mpc"].get("w_F", 0.01))
        self.w_ramp = float(cfg["mpc"].get("w_ramp", 0.1))

        self.model = Burning(N=5)

        self.opti = ca.Opti()
        self.prev_sol = None

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
    def _build(self):

        N = self.model.N
        eps = float(self.model.eps)

        # ================= STATES =================
        self.Tg = self.opti.variable(N, self.Np + 1)
        self.Ts = self.opti.variable(N, self.Np + 1)
        self.Tw = self.opti.variable(N, self.Np + 1)

        # ================= CONTROL =================
        self.Fuel = self.opti.variable(self.Nc)

        # ================= INITIAL CONDITIONS =================
        self.Tg0 = self.opti.parameter(N)
        self.Ts0 = self.opti.parameter(N)
        self.Tw0 = self.opti.parameter(N)

        self.opti.subject_to(
            ca.sumsqr(self.Tg[:, 0] - self.Tg0)
            + ca.sumsqr(self.Ts[:, 0] - self.Ts0)
            + ca.sumsqr(self.Tw[:, 0] - self.Tw0)
            <= 1e-2
        )

        # ================= INITIAL GUESS =================
        self.opti.set_initial(self.Tg, 1800)
        self.opti.set_initial(self.Ts, 1700)
        self.opti.set_initial(self.Tw, 900)
        self.opti.set_initial(self.Fuel, 4.0)

        # ================= HARD CONSTRAINTS =================
        self.opti.subject_to(self.Fuel >= self.fuel_min)
        self.opti.subject_to(self.Fuel <= self.fuel_max)

        # ================= MIX =================
        p0, a0, l0 = 0.6, 0.2, 0.2
        norm = p0 + a0 + l0 + eps
        p0, a0, l0 = p0 / norm, a0 / norm, l0 / norm

        O2 = 3.0
        eta = ca.exp(-((O2 - self.model.O2_opt) ** 2) / self.model.O2_sigma2)

        LHV_mix = (
            p0 * self.model.LHV_petcoke
            + l0 * self.model.LHV_lignite
            + a0 * self.model.LHV_RDF
        )

        # ================= COST =================
        cost = 0.0

        def u(k):
            return min(k, self.Nc - 1)

        # ================= PREDICTION LOOP =================
        for k in range(self.Np):

            Fuel_k = self.Fuel[u(k)]

            Tg_pred = self.Tg[:, k]
            Ts_pred = self.Ts[:, k]
            Tw_pred = self.Tw[:, k]

        # ================= PREDICTION LOOP =================
        for k in range(self.Np):

            Fuel_k = self.Fuel[u(k)]

            # Prediction starts from current optimization state
            Tg_pred = self.Tg[:, k]
            Ts_pred = self.Ts[:, k]
            Tw_pred = self.Tw[:, k]

            # ======================================================
            # Integrate physical plant model over one prediction step
            # ======================================================
            for _ in range(self.n_substeps):

                # ---------------- Fuel ----------------
                fuel_rate_kg_s = Fuel_k * 1000.0 / 3600.0

                Q_in = fuel_rate_kg_s * LHV_mix * eta
                q_vol = Q_in / (self.model.V_total + eps)

                # ---------------- Gradients ----------------
                dTg = (Tg_pred[1:] - Tg_pred[:-1]) / self.model.dz
                dTg_dz = ca.vertcat(dTg[0], dTg)

                dTs = (Ts_pred[1:] - Ts_pred[:-1]) / self.model.dz
                dTs_dz = ca.vertcat(dTs[0], dTs)

                # ---------------- Heat Transfer ----------------
                q_gs = (
                    self.model.hv_gs * self.model.a_gs * (Tg_pred - Ts_pred)
                ) / self.model.V_cell

                q_gw = (
                    self.model.hv_gw * self.model.a_gw * (Tg_pred - Tw_pred)
                ) / self.model.V_cell

                q_ws = (
                    self.model.hv_ws * self.model.a_ws * (Ts_pred - Tw_pred)
                ) / self.model.V_cell

                # ---------------- Capacities ----------------
                C_g = self.model.rho_g * self.model.V_cell * self.model.Cp_g

                effective = 0.01

                C_s = effective * self.model.rho_s * self.model.Cp_s

                C_w = self.model.rho_wall * self.model.V_wall * self.model.Cp_wall

                # ---------------- Wall Loss ----------------
                q_loss = (
                    self.model.h_ext * self.model.A_wall * (Tw_pred - self.model.T_amb)
                ) / (self.model.V_cell + eps)

                # ---------------- Euler Integration ----------------
                Tg_pred = Tg_pred + self.dt_model * (
                    -self.model.u_g * dTg_dz + (q_vol - q_gs - q_gw) / C_g
                )

                Ts_pred = Ts_pred + self.dt_model * (
                    -self.model.u_s * dTs_dz + (q_gs - q_ws) / C_s
                )

                Tw_pred = Tw_pred + self.dt_model * ((q_gw + q_ws - q_loss) / C_w)

            # ======================================================
            # End of one prediction interval (5 s)
            # ======================================================

            self.opti.subject_to(self.Tg[:, k + 1] == Tg_pred)
            self.opti.subject_to(self.Ts[:, k + 1] == Ts_pred)
            self.opti.subject_to(self.Tw[:, k + 1] == Tw_pred)

            # ---------------- Tracking Cost ----------------
            Tg_out = Tg_pred[-1]
            Ts_out = Ts_pred[-1]

            cost += self.w_T * (Tg_out - self.Tg_ref) ** 2
            cost += self.w_T * (Ts_out - self.Ts_ref) ** 2

            # ---------------- Fuel Cost ----------------
            cost += self.w_F * Fuel_k**2

        # ======================================================
        # RAMP COST
        # ======================================================
        cost += self.w_ramp * ca.sumsqr(self.Fuel[1:] - self.Fuel[:-1])

        # ======================================================
        # OBJECTIVE
        # ======================================================
        self.opti.minimize(cost)

        # ======================================================
        # SOLVER
        # ======================================================
        self.opti.solver(
            "ipopt",
            {
                "ipopt.max_iter": int(self.cfg["solver"]["max_iter"]),
                "ipopt.tol": float(self.cfg["solver"]["tolerance"]),
                "print_time": 0,
                "ipopt.print_level": 0,
            },
        )

    # ======================================================
    def compute_control(self, state, t=0.0):

        if t - self.last_control_time < self.dt_control:
            return {"Fuel_rate": self.last_control}

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
                        self.Fuel[k],
                        self.prev_sol.value(self.Fuel[k]),
                    )
            except Exception:
                pass

        start = datetime.now()
        print(f"[{start.strftime('%H:%M:%S')}] MPC Solve started at t={t:.1f} s")

        try:
            sol = self.opti.solve()

            end = datetime.now()
            print(
                f"[{end.strftime('%H:%M:%S')}] MPC Solve finished "
                f"({(end-start).total_seconds():.2f} s)"
            )

            self.prev_sol = sol
            fuel = float(sol.value(self.Fuel[0]))

            self.last_control = fuel
            self.last_control_time = t

        except Exception as e:
            print("MPC FAILED → fallback:", repr(e))
            fuel = self.last_control

        return {"Fuel_rate": fuel}


# ======================================================
# REPORTING
# ======================================================
def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    # ================= CONFIGS =================
    twin_cfg = load_cfg("configs/twin_cfg.yaml")
    mpc_cfg = load_cfg("configs/mpc_cfg.yaml")

    # ================= HELPERS =================
    def save_ckpt(path, state):
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_ckpt(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ================= MODELS =================
    plant_model = Burning(N=twin_cfg["plant"]["N"])
    mpc_controller = MasterMPC(mpc_cfg)

    # ================= SIM PARAMS =================
    dt = twin_cfg["simulation"]["dt"]
    chunk_hours = twin_cfg["simulation"]["chunk_hours"]

    chunk_time = chunk_hours * 3600.0
    steps_per_chunk = int(chunk_time / dt)

    total_hours = twin_cfg["simulation"]["total_hours"]
    n_chunks = int(total_hours / chunk_hours)

    # ================= LOGGING =================
    log_interval = twin_cfg["logging"]["interval_sec"]
    next_log = 0.0

    # ================= CHECKPOINT =================
    ckpt_path = twin_cfg["checkpoint"]["file"]
    ckpt_enabled = twin_cfg["checkpoint"]["enabled"]

    log_path = "control/mpc_status.jsonl"

    # ================= INIT STATE =================
    if os.path.exists(ckpt_path):
        st = load_ckpt(ckpt_path)
        Tg, Ts, Tw = st["Tg"], st["Ts"], st["Tw"]
        start_chunk = int(st["t"] // chunk_time)
        print(f"[RESUME] chunk={start_chunk}")
    else:
        Tg = np.ones(twin_cfg["plant"]["N"]) * 1773.15
        Ts = np.ones(twin_cfg["plant"]["N"]) * 1673.15
        Tw = np.ones(twin_cfg["plant"]["N"]) * 873.15
        start_chunk = 0

    outlet = -1

    # ================= PLANT INPUTS =================
    plant_inputs = {
        "Petcoke": twin_cfg["fuel"]["petcoke_fraction"],
        "RDF_Fuel": twin_cfg["fuel"]["rdf_fraction"],
        "O2": twin_cfg["fuel"]["oxygen"],
        "Fuel_rate": twin_cfg["fuel"]["fuel_rate"],
    }

    # ======================================================
    # MAIN LOOP
    # ======================================================
    for chunk in range(start_chunk, n_chunks):

        t_local = 0.0

        for i in range(steps_per_chunk):

            global_time = chunk * chunk_time + t_local

            # ================= STATE OBJECT =================
            state_obj = type("S", (), {})()
            state_obj.Tg_burning = Tg
            state_obj.Ts_burning = Ts
            state_obj.Tw_burning = Tw

            # ================= MPC =================
            ctrl = mpc_controller.compute_control(
                state_obj,
                t=global_time,
            )

            plant_inputs["Fuel_rate"] = ctrl["Fuel_rate"]

            # ================= PLANT =================
            Tg, Ts, Tw = plant_model.thermal_step(
                Tg,
                Ts,
                Tw,
                plant_inputs,
                dt,
            )

            t_local += dt
            global_time = chunk * chunk_time + t_local

            # ================= LOGGING =================
            if global_time >= next_log:

                record = {
                    "chunk": chunk,
                    "time_h": f"{global_time/3600.0:.4f}",
                    "time_s": round(global_time, 2),
                    "Tg": float(Tg[outlet]),
                    "Ts": float(Ts[outlet]),
                    "Tw": float(Tw[outlet]),
                    "Fuel": float(plant_inputs["Fuel_rate"]),
                }

                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                next_log += log_interval

        # ================= CHECKPOINT =================
        global_t = (chunk + 1) * chunk_time

        if ckpt_enabled:
            save_ckpt(
                ckpt_path,
                {
                    "Tg": Tg,
                    "Ts": Ts,
                    "Tw": Tw,
                    "t": global_t,
                },
            )

        print(
            f"[CHUNK {chunk}] "
            f"t={global_t/3600:.2f} h | "
            f"Tg={Tg[outlet]:.2f} | "
            f"Ts={Ts[outlet]:.2f} | "
            f"Tw={Tw[outlet]:.2f}"
        )
