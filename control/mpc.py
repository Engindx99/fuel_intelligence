import os
import json
import pickle
import numpy as np
import yaml

from datetime import datetime

import casadi as ca

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
class BurningMPCModel:

    def __init__(self, burning_model):
        self.m = burning_model
        self.N = burning_model.N

    def dynamics(self, x, u):

        N = self.N
        m = self.m
        eps = m.eps

        Tg = x[0:N]
        Ts = x[N:2*N]
        Tw = x[2*N:3*N]

        fuel_rate = u * 1000.0 / 3600.0
        Q_in = fuel_rate * m.LHV_petcoke  # baseline
        q_vol = Q_in / (m.V_total + eps)

        dTg = ca.vertcat((Tg[1:] - Tg[:-1]) / m.dz, 0)
        dTs = ca.vertcat((Ts[1:] - Ts[:-1]) / m.dz, 0)

        q_gs = (m.hv_gs * m.a_gs * (Tg - Ts)) / m.V_cell
        q_gw = (m.hv_gw * m.a_gw * (Tg - Tw)) / m.V_cell
        q_ws = (m.hv_ws * m.a_ws * (Ts - Tw)) / m.V_cell

        C_g = m.rho_g * m.V_cell * m.Cp_g
        C_s = 0.01 * m.rho_s * m.Cp_s
        C_w = m.rho_wall * m.V_wall * m.Cp_wall

        q_loss = (m.h_ext * m.A_wall * (Tw - m.T_amb)) / (m.V_cell + eps)

        Tg_dot = -m.u_g * dTg + (q_vol - q_gs - q_gw) / C_g
        Ts_dot = -m.u_s * dTs + (q_gs - q_ws) / C_s
        Tw_dot = (q_gw + q_ws - q_loss) / C_w

        return ca.vertcat(Tg_dot, Ts_dot, Tw_dot)
    


        from acados_template import (
            AcadosOcp,
            AcadosOcpSolver,
            AcadosModel,
        )

        # ==========================================================
        # 1. CASADI DYNAMICS WRAPPER (Burning Model)
        # ==========================================================

        def build_dynamics(model):

            N = model.N
            dz = model.dz

            Tg = ca.SX.sym("Tg", N)
            Ts = ca.SX.sym("Ts", N)
            Tw = ca.SX.sym("Tw", N)

            u = ca.SX.sym("Fuel")
            eta = ca.SX.sym("eta")

            # parameters (mix simplified)
            LHV = ca.SX.sym("LHV")

            eps = model.eps

            fuel_rate = u * 1000.0 / 3600.0
            Q_in = fuel_rate * LHV * eta
            q_vol = Q_in / (model.V_total + eps)

            # gradients
            dTg = (Tg[1:] - Tg[:-1]) / dz
            dTs = (Ts[1:] - Ts[:-1]) / dz

            dTg_dz = ca.vertcat(dTg[0], dTg)
            dTs_dz = ca.vertcat(dTs[0], dTs)

            # heat transfer
            q_gs = (model.hv_gs * model.a_gs * (Tg - Ts)) / model.V_cell
            q_gw = (model.hv_gw * model.a_gw * (Tg - Tw)) / model.V_cell
            q_ws = (model.hv_ws * model.a_ws * (Ts - Tw)) / model.V_cell

            # capacities
            C_g = model.rho_g * model.V_cell * model.Cp_g
            C_s = 0.01 * model.rho_s * model.Cp_s
            C_w = model.rho_wall * model.V_wall * model.Cp_wall

            q_loss = (model.h_ext * model.A_wall * (Tw - model.T_amb)) / (model.V_cell + eps)

            # dynamics
            Tg_dot = -model.u_g * dTg_dz + (q_vol - q_gs - q_gw) / C_g
            Ts_dot = -model.u_s * dTs_dz + (q_gs - q_ws) / C_s
            Tw_dot = (q_gw + q_ws - q_loss) / C_w

            xdot = ca.vertcat(Tg_dot, Ts_dot, Tw_dot)

            x = ca.vertcat(Tg, Ts, Tw)

            return x, xdot, u, eta, LHV


        # ==========================================================
        # 2. ACADOS OCP BUILDER
        # ==========================================================

        def build_acados_ocp(model, cfg):

            ocp = AcadosOcp()

            # ------------------------------------------------------
            # MODEL
            # ------------------------------------------------------
            x, xdot, u, eta, LHV = build_dynamics(model)

            nx = x.size1()
            nu = u.size1()

            acados_model = AcadosModel()
            acados_model.x = x
            acados_model.xdot = xdot
            acados_model.u = u

            acados_model.name = "burning_ocp"

            ocp.model = acados_model

            # ------------------------------------------------------
            # TIME SETTINGS
            # ------------------------------------------------------
            ocp.dims.N = cfg["mpc"]["prediction_horizon"]
            ocp.solver_options.tf = cfg["mpc"]["dt_prediction"] * ocp.dims.N

            # ------------------------------------------------------
            # COST
            # ------------------------------------------------------
            Tg_ref = cfg["mpc"]["Tg_setpoint"]
            Ts_ref = cfg["mpc"]["Ts_setpoint"]

            Q_T = cfg["mpc"]["w_T"]
            R_F = cfg["mpc"]["w_F"]

            cost_expr = (
                Q_T * ca.sumsqr(x[:model.N] - Tg_ref)
                + Q_T * ca.sumsqr(x[model.N:2*model.N] - Ts_ref)
                + R_F * u**2
            )

            ocp.cost.cost_type = "EXTERNAL"
            ocp.model.cost_expr_ext_cost = cost_expr

            # ------------------------------------------------------
            # CONSTRAINTS
            # ------------------------------------------------------
            ocp.constraints.lbu = np.array([cfg["mpc"]["fuel_min"]])
            ocp.constraints.ubu = np.array([cfg["mpc"]["fuel_max"]])
            ocp.constraints.idxbu = np.array([0])

            # ------------------------------------------------------
            # INITIAL CONDITION PLACEHOLDER
            # ------------------------------------------------------
            ocp.constraints.x0 = np.zeros(nx)

            # ------------------------------------------------------
            # SOLVER OPTIONS
            # ------------------------------------------------------
            ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
            ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
            ocp.solver_options.integrator_type = "IRK"
            ocp.solver_options.nlp_solver_type = "SQP_RTI"

            ocp.solver_options.print_level = 0

            return ocp


        # ==========================================================
        # 3. SOLVER WRAPPER (RL READY)
        # ==========================================================

from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver

class ACADOSMPC:

    def __init__(self, burning_model, cfg):

        self.m = burning_model
        self.cfg = cfg
        self.model = BurningMPCModel(burning_model)

        self.N = burning_model.N

        self.ocp = AcadosOcp()
        self.build()

        self.solver = AcadosOcpSolver(self.ocp)

    def build(self):

        N = self.N

        x = ca.SX.sym("x", 3*N)
        u = ca.SX.sym("u")

        xdot = self.model.dynamics(x, u)

        model = AcadosModel()
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = "burning"

        self.ocp.model = model

        # -------------------------
        # TIME
        # -------------------------
        self.ocp.dims.N = self.cfg["mpc"]["prediction_horizon"]
        self.ocp.solver_options.tf = self.cfg["mpc"]["dt_prediction"] * self.ocp.dims.N

        # -------------------------
        # COST
        # -------------------------
        Tg_ref = self.cfg["mpc"]["Tg_setpoint"]
        Ts_ref = self.cfg["mpc"]["Ts_setpoint"]

        Q = self.cfg["mpc"]["w_T"]
        R = self.cfg["mpc"]["w_F"]

        Tg = x[0:N]
        Ts = x[N:2*N]

        cost = Q * ca.sumsqr(Tg - Tg_ref) + Q * ca.sumsqr(Ts - Ts_ref) + R * u**2

        self.ocp.cost.cost_type = "EXTERNAL"
        self.ocp.model.cost_expr_ext_cost = cost

        # -------------------------
        # INPUT CONSTRAINT
        # -------------------------
        self.ocp.constraints.lbu = np.array([self.cfg["mpc"]["fuel_min"]])
        self.ocp.constraints.ubu = np.array([self.cfg["mpc"]["fuel_max"]])
        self.ocp.constraints.idxbu = np.array([0])

        # -------------------------
        # INITIAL STATE
        # -------------------------
        self.ocp.constraints.x0 = np.zeros(3*N)

        # -------------------------
        # SOLVER
        # -------------------------
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.integrator_type = "IRK"
        self.ocp.solver_options.print_level = 0

    # ==================================================
    def build(self):

        m = self.m
        N = self.N

        x = ca.SX.sym("x", 3*N)
        u = ca.SX.sym("u")

        eta = ca.SX.sym("eta")
        LHV = ca.SX.sym("LHV")

        xdot = self.model.dynamics(x, u, eta, LHV)

        model = AcadosModel()
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = "burning"

        self.ocp.model = model

        # TIME
        self.ocp.dims.N = self.cfg["mpc"]["prediction_horizon"]
        self.ocp.solver_options.tf = self.cfg["mpc"]["dt_prediction"] * self.ocp.dims.N

        # COST (REAL ACADOS STYLE)
        Tg_ref = self.cfg["mpc"]["Tg_setpoint"]
        Ts_ref = self.cfg["mpc"]["Ts_setpoint"]

        Q = self.cfg["mpc"]["w_T"]
        R = self.cfg["mpc"]["w_F"]

        cost = 0

        Tg = x[0:N]
        Ts = x[N:2*N]

        cost += Q * ca.sumsqr(Tg - Tg_ref)
        cost += Q * ca.sumsqr(Ts - Ts_ref)
        cost += R * u**2

        self.ocp.cost.cost_type = "EXTERNAL"
        self.ocp.model.cost_expr_ext_cost = cost

        # CONSTRAINTS
        self.ocp.constraints.lbu = np.array([self.cfg["mpc"]["fuel_min"]])
        self.ocp.constraints.ubu = np.array([self.cfg["mpc"]["fuel_max"]])
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.constraints.x0 = np.zeros(3*N)

        # SOLVER
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.integrator_type = "IRK"
    
    
class MasterMPC:

    def __init__(self, cfg, burning_model):

        self.cfg = cfg
        self.mpc = ACADOSMPC(burning_model, cfg)

        self.last_u = cfg["mpc"]["fuel_min"]
        self.last_t = -1e9
        self.dt_control = cfg["mpc"]["dt_control"]

    def compute_control(self, state, t=0.0):

        if t - self.last_t < self.dt_control:
            return {"Fuel_rate": self.last_u}

        x0 = np.concatenate([
            state.Tg_burning,
            state.Ts_burning,
            state.Tw_burning
        ])

        self.mpc.solver.set(0, "lbx", x0)
        self.mpc.solver.set(0, "ubx", x0)

        self.mpc.solver.solve()

        u = self.mpc.solver.get(0, "u")[0]

        self.last_u = float(u)
        self.last_t = t

        return {"Fuel_rate": self.last_u}
    
    
    if __name__ == "__main__":

    twin_cfg = load_cfg("configs/twin_cfg.yaml")
    mpc_cfg = load_cfg("configs/mpc_cfg.yaml")

    def save_ckpt(path, state):
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_ckpt(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    plant = Burning(N=twin_cfg["plant"]["N"])
    mpc = MasterMPC(mpc_cfg, plant)

    dt = twin_cfg["simulation"]["dt"]
    chunk_hours = twin_cfg["simulation"]["chunk_hours"]

    chunk_time = chunk_hours * 3600.0
    steps_per_chunk = int(chunk_time / dt)

    total_hours = twin_cfg["simulation"]["total_hours"]
    n_chunks = int(total_hours / chunk_hours)

    log_interval = twin_cfg["logging"]["interval_sec"]
    next_log = 0.0

    log_path = "control/mpc_status.jsonl"

    Tg = np.ones(plant.N) * 1773
    Ts = np.ones(plant.N) * 1673
    Tw = np.ones(plant.N) * 873

    outlet = -1

    plant_inputs = {
        "Petcoke": 0.6,
        "RDF_Fuel": 0.2,
        "O2": 3.5,
        "Fuel_rate": 4.0,
    }

    for chunk in range(n_chunks):

        t_local = 0.0

        for i in range(steps_per_chunk):

            t = chunk * chunk_time + t_local

            state = type("S", (), {})()
            state.Tg_burning = Tg
            state.Ts_burning = Ts
            state.Tw_burning = Tw

            ctrl = mpc.compute_control(state, t)

            plant_inputs["Fuel_rate"] = ctrl["Fuel_rate"]

            Tg, Ts, Tw = plant.thermal_step(
                Tg, Ts, Tw, plant_inputs, dt
            )

            t_local += dt

            if t >= next_log:

                record = {
                    "t_h": t / 3600.0,
                    "Tg": float(Tg[outlet]),
                    "Ts": float(Ts[outlet]),
                    "Tw": float(Tw[outlet]),
                    "Fuel": float(plant_inputs["Fuel_rate"]),
                }

                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                next_log += log_interval

        print(f"[CHUNK {chunk}] Tg={Tg[outlet]:.2f} Ts={Ts[outlet]:.2f}")