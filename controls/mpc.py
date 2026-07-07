import numpy as np
import casadi as ca
import yaml
from kiln.burning import Burning
from kiln.globalstate import GlobalState
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
from enum import IntEnum
from controls.mpc_parameters import MPCParameter


class BurningMPCModel:

    def __init__(self, burning_model):

        self.np = len(MPCParameter)

        # ======================================================
        # REFERENCE TO PHYSICAL MODEL
        # ======================================================
        self.m = burning_model

        # ======================================================
        # DISCRETIZATION
        # ======================================================
        self.N = burning_model.N

        # ======================================================
        # STATE DIMENSIONS
        # ======================================================
        self.n_Tg = self.N
        self.n_Ts = self.N
        self.n_Tw = self.N

        self.nx = self.n_Tg + self.n_Ts + self.n_Tw
        self.nu = 1  # Fuel_rate

    # ======================================================
    # STATE VECTOR
    # ======================================================
    def build_state(self):

        Tg = ca.SX.sym("Tg", self.N)
        Ts = ca.SX.sym("Ts", self.N)
        Tw = ca.SX.sym("Tw", self.N)

        x = ca.vertcat(Tg, Ts, Tw)

        return x, Tg, Ts, Tw

    # ======================================================
    # CONTROL VECTOR
    # ======================================================
    def build_control(self):

        Fuel_rate = ca.SX.sym("Fuel_rate")
        u = ca.vertcat(Fuel_rate)

        return u, Fuel_rate

    # ======================================================
    # PARAMETERS
    # ======================================================
    def build_parameters(self):

        O2 = ca.SX.sym("O2")

        Petcoke_ratio = ca.SX.sym("Petcoke_ratio")
        Coal_ratio = ca.SX.sym("Coal_ratio")
        RDF_ratio = ca.SX.sym("RDF_ratio")
        H2_ratio = ca.SX.sym("H2_ratio")



        Fuel_prev = ca.SX.sym("Fuel_prev")

        p = ca.vertcat(
            O2,
            Petcoke_ratio,
            Coal_ratio,
            RDF_ratio,
            H2_ratio,
            Fuel_prev,
        )

        return (
            p,
            O2,
            Petcoke_ratio,
            Coal_ratio,
            RDF_ratio,
            H2_ratio,
            Fuel_prev,
        )

    # ======================================================
    # COMBUSTION EFFICIENCY
    # ======================================================
    def combustion_efficiency(self, O2):

        return ca.exp(-((O2 - self.m.O2_opt) ** 2) / self.m.O2_sigma2)

    # ======================================================
    # HEAT SOURCE
    # ======================================================
    def build_heat_source(
        self,
        Fuel_rate,
        O2,
        Petcoke_ratio,
        Coal_ratio,
        RDF_ratio,
        H2_ratio,
    ):

        # --------------------------------------------------
        # Fuel fractions
        # --------------------------------------------------
        p = Petcoke_ratio
        c = Coal_ratio
        r = RDF_ratio
        h = H2_ratio

        norm = p + c + r + h + self.m.eps

        p /= norm
        c /= norm
        r /= norm
        h /= norm

        # --------------------------------------------------
        # Combustion efficiency
        # --------------------------------------------------
        eta = self.combustion_efficiency(O2)

        # --------------------------------------------------
        # Mixed fuel heating value
        # --------------------------------------------------
        LHV_mix = (
            p * self.m.LHV_petcoke
            + c * self.m.LHV_coal
            + r * self.m.LHV_RDF
            + h * self.m.LHV_H2
        )

        # --------------------------------------------------
        # Fuel conversion
        # --------------------------------------------------
        fuel_rate_kg_s = Fuel_rate * 1000.0 / 3600.0

        # --------------------------------------------------
        # Heat input
        # --------------------------------------------------
        Q_in = fuel_rate_kg_s * LHV_mix * eta
        q_vol = Q_in / (self.m.V_total + self.m.eps)

        # --------------------------------------------------
        # Transition sink
        # --------------------------------------------------
        sink_density = Transition_Q_sink / (self.m.V_total + self.m.eps)
        q_vol -= 0.05 * sink_density

        return q_vol

    # ======================================================
    # HEAT TRANSFER
    # ======================================================
    def build_heat_transfer(self, Tg, Ts, Tw):

        q_gs = self.m.hv_gs * self.m.a_gs * (Tg - Ts)
        q_gw = self.m.hv_gw * self.m.a_gw * (Tg - Tw)
        q_ws = self.m.hv_ws * self.m.a_ws * (Ts - Tw)

        return q_gs, q_gw, q_ws

    # ======================================================
    # SPATIAL GRADIENTS
    # ======================================================
    def build_gradients(self, Tg, Ts):

        dTg = (Tg[1:] - Tg[:-1]) / self.m.dz
        dTs = (Ts[1:] - Ts[:-1]) / self.m.dz

        dTg_dz = ca.vertcat(dTg[0], dTg)
        dTs_dz = ca.vertcat(dTs[0], dTs)

        return dTg_dz, dTs_dz

    # ======================================================
    # THERMAL CAPACITIES
    # ======================================================
    def build_capacities(self):

        effective = 0.1

        C_s = self.m.rho_s * self.m.Cp_s
        effective_C_s = effective * C_s

        C_g = self.m.rho_g * self.m.V_cell * self.m.Cp_g
        C_w = self.m.rho_wall * self.m.V_wall * self.m.Cp_wall

        return C_g, effective_C_s, C_w

    # ======================================================
    # WALL HEAT LOSSES
    # ======================================================
    def build_wall_losses(self, Tw):

        q_loss = (
            self.h_ext
            * self.A_wall
            * (Tw - self.T_amb)
        ) / (self.V_cell + self.eps)

        state.Wall_loss_transition = np.sum(q_loss * self.V_cell)

        return q_loss

    # ======================================================
    # DYNAMICS
    # ======================================================
    def build_dynamics(self):

        x, Tg, Ts, Tw = self.build_state()
        u, Fuel_rate = self.build_control()

        (
            p,
            O2,
            Petcoke_ratio,
            Coal_ratio,
            RDF_ratio,
            H2_ratio,
            Fuel_prev,
        ) = self.build_parameters()

        dTg_dz, dTs_dz = self.build_gradients(Tg, Ts)

        q_vol = self.build_heat_source(
            Fuel_rate,
            O2,
            Petcoke_ratio,
            Coal_ratio,
            RDF_ratio,
            H2_ratio,
        )

        q_gs, q_gw, q_ws = self.build_heat_transfer(Tg, Ts, Tw)

        C_g, C_s, C_w = self.build_capacities()

        q_loss = self.build_wall_losses(Tw)

        Tg_dot = -self.m.u_g * dTg_dz + (q_vol - q_gs - q_gw) / C_g
        Ts_dot = -self.m.u_s * dTs_dz + (q_gs - q_ws) / C_s
        Tw_dot = (q_gw + q_ws - q_loss) / C_w

        xdot = ca.vertcat(Tg_dot, Ts_dot, Tw_dot)

        return x, u, p, xdot
    
class ACADOSMPC:

    def __init__(self, cfg, burning_model):

        # ======================================================
        # CONFIG
        # ======================================================
        self.cfg = cfg

        # ======================================================
        # PHYSICAL MODEL
        # ======================================================
        self.burning_model = burning_model

        # ======================================================
        # CASADI MODEL
        # ======================================================
        self.casadi_model = BurningMPCModel(burning_model)

        # ======================================================
        # BUILD MODEL
        # ======================================================
        self.model = self.build_model()

        # ======================================================
        # BUILD OCP
        # ======================================================
        self.ocp = self.build_ocp()
        self.ocp = self.build_cost(self.ocp)
        self.ocp = self.build_constraints(self.ocp)
        self.ocp = self.build_solver_options(self.ocp)

        # ======================================================
        # SOLVER
        # ======================================================
        self.solver = self.build_solver()

    # ======================================================
    # BUILD ACADOS MODEL
    # ======================================================
    def build_model(self):

        x, u, p, xdot = self.casadi_model.build_dynamics()

        model = AcadosModel()
        model.name = "burning_zone"

        model.x = x
        model.u = u
        model.p = p

        model.xdot = ca.SX.sym("xdot", self.casadi_model.nx)

        model.f_expl_expr = xdot
        model.f_impl_expr = model.xdot - xdot

        return model

    # ======================================================
    # BUILD OCP
    # ======================================================
    def build_ocp(self):

        ocp = AcadosOcp()

        ocp.model = self.model

        ocp.parameter_values = np.zeros(self.casadi_model.np)

        ocp.dims.N = self.cfg["mpc"]["prediction_horizon"]

        ocp.solver_options.tf = (
            self.cfg["mpc"]["prediction_horizon"]
            * self.cfg["mpc"]["dt_prediction"]
        )

        return ocp

    # ======================================================
    # STAGE COST
    # ======================================================
    def build_stage_cost(self):

        N = self.casadi_model.N

        x = self.model.x
        u = self.model.u
        p = self.model.p

        Tg = x[0:N]
        Ts = x[N:2 * N]

        Fuel_rate = u[0]

        Fuel_prev = p[MPCParameter.FUEL_PREV]

        Tg_ref = self.cfg["mpc"]["Tg_setpoint"]
        Ts_ref = self.cfg["mpc"]["Ts_setpoint"]

        w_T = float(self.cfg["mpc"]["w_T"])
        w_F = float(self.cfg["mpc"]["w_F"])
        w_ramp = float(self.cfg["mpc"]["w_ramp"])

        tracking_cost = (
            w_T * ca.sumsqr(Tg - Tg_ref)
            + w_T * ca.sumsqr(Ts - Ts_ref)
        )

        fuel_cost = w_F * Fuel_rate**2

        fuel_delta = Fuel_rate - Fuel_prev
        ramp_cost = w_ramp * fuel_delta**2

        return tracking_cost + fuel_cost + ramp_cost

    # ======================================================
    # TERMINAL COST
    # ======================================================
    def build_terminal_cost(self):

        N = self.casadi_model.N

        x = self.model.x

        Tg = x[0:N]
        Ts = x[N:2 * N]

        Tg_ref = self.cfg["mpc"]["Tg_setpoint"]
        Ts_ref = self.cfg["mpc"]["Ts_setpoint"]

        w_T = float(self.cfg["mpc"]["w_T"])

        return (
            w_T * ca.sumsqr(Tg - Tg_ref)
            + w_T * ca.sumsqr(Ts - Ts_ref)
        )

    # ======================================================
    # COST
    # ======================================================
    def build_cost(self, ocp):

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.model.cost_expr_ext_cost = self.build_stage_cost()
        ocp.model.cost_expr_ext_cost_e = self.build_terminal_cost()

        return ocp

    # ======================================================
    # CONSTRAINTS
    # ======================================================
    def build_constraints(self, ocp):

        # --------------------------------------------------
        # Fuel bounds
        # --------------------------------------------------
        ocp.constraints.lbu = np.array([self.cfg["mpc"]["fuel_min"]])
        ocp.constraints.ubu = np.array([self.cfg["mpc"]["fuel_max"]])
        ocp.constraints.idxbu = np.array([0])

        # --------------------------------------------------
        # Ramp constraint (ΔFuel hard constraint)
        # --------------------------------------------------
        Fuel = self.model.u[0]
        Fuel_prev = self.model.p[MPCParameter.FUEL_PREV]

        fuel_delta = Fuel - Fuel_prev

        delta_max = float(self.cfg["mpc"]["max_fuel_delta"])

        ocp.model.con_h_expr = ca.vertcat(fuel_delta)
        ocp.constraints.lh = np.array([-delta_max])
        ocp.constraints.uh = np.array([delta_max])

        # --------------------------------------------------
        # Initial state
        # --------------------------------------------------
        ocp.constraints.x0 = np.zeros(self.casadi_model.nx)

        return ocp

    # ======================================================
    # SOLVER OPTIONS
    # ======================================================
    def build_solver_options(self, ocp):

        solver_cfg = self.cfg["solver"]

        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "IRK"

        ocp.solver_options.nlp_solver_max_iter = solver_cfg["max_iter"]

        tol = float(solver_cfg["tolerance"])

        ocp.solver_options.nlp_solver_tol_stat = tol
        ocp.solver_options.nlp_solver_tol_eq = tol
        ocp.solver_options.nlp_solver_tol_ineq = tol
        ocp.solver_options.nlp_solver_tol_comp = tol

        ocp.solver_options.print_level = 0

        return ocp

    # ======================================================
    # SOLVER
    # ======================================================
    def build_solver(self):

        return AcadosOcpSolver(
            self.ocp,
            json_file="acados_ocp.json",
        )

    # ======================================================
    # INITIAL STATE
    # ======================================================
    def set_initial_state(self, Tg, Ts, Tw):

        x0 = np.concatenate([Tg, Ts, Tw])

        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

    # ======================================================
    # PARAMETERS
    # ======================================================
    def set_parameters(self, parameters):

        parameters = np.asarray(parameters, dtype=float)

        if len(parameters) != self.casadi_model.np:
            raise ValueError(
                f"Expected {self.casadi_model.np} MPC parameters, "
                f"got {len(parameters)}."
            )

        for k in range(self.ocp.dims.N + 1):
            self.solver.set(k, "p", parameters)

    # ======================================================
    # SOLVE
    # ======================================================
    def solve(self):

        status = self.solver.solve()

        if status != 0:
            raise RuntimeError(f"ACADOS solver failed with status {status}")

        return float(self.solver.get(0, "u")[0])
        


class MasterMPC:

    def __init__(self, cfg, burning_model):

        self.cfg = cfg
        self.mpc = ACADOSMPC(cfg, burning_model)

        # ======================================================
        # TIMING
        # ======================================================
        self.dt_mpc = cfg["mpc"]["mpc_update_sec"]          # 5 s
        self.dt_act = cfg["mpc"]["actuator_update_sec"]     # 60 s

        self.last_mpc_time = -1e30
        self.last_act_time = -1e30

        # ======================================================
        # CONTROL MEMORY
        # ======================================================
        self.last_control = cfg["mpc"]["fuel_min"]

        self._current_plan = None
        self._plan_index = 0

    # ======================================================
    # STATE UPDATE
    # ======================================================
    def update_state(self, state):

        self.mpc.set_initial_state(
            state.Tg_burning,
            state.Ts_burning,
            state.Tw_burning,
        )

    # ======================================================
    # PARAMETER UPDATE
    # ======================================================
    def update_parameters(self, inputs, state):

        p = np.zeros(len(MPCParameter))

        p[MPCParameter.O2] = inputs["O2"]

        p[MPCParameter.PETCOKE_RATIO] = inputs["Petcoke_ratio"]
        p[MPCParameter.COAL_RATIO]    = inputs["Coal_ratio"]
        p[MPCParameter.RDF_RATIO]     = inputs["RDF_ratio"]
        p[MPCParameter.H2_RATIO]      = inputs["H2_ratio"]

        p[MPCParameter.CALCINATION] = getattr(
            state,
            "Calcination_Q_sink",
            0.0,
        )

        p[MPCParameter.FUEL_PREV] = (
            self._current_plan[0]
            if self._current_plan
            else self.last_control
        )

        self.mpc.set_parameters(p)

    # ======================================================
    # SOLVE MPC
    # ======================================================
    def solve(self):

        status = self.mpc.solver.solve()

        if status != 0:
            raise RuntimeError(f"ACADOS failed: {status}")

        N = self.mpc.cfg["mpc"]["prediction_horizon"]

        U = []

        for k in range(N):
            U.append(float(self.mpc.solver.get(k, "u")[0]))

        self._current_plan = U
        self._plan_index = 0

        return U

    # ======================================================
    # CONTROL LOOP
    # ======================================================
    def compute_control(self, state, inputs, t):

        # ======================================================
        # MPC UPDATE
        # ======================================================
        if t - self.last_mpc_time >= self.dt_mpc:

            self.update_state(state)
            self.update_parameters(inputs, state)

            self.solve()

            self.last_mpc_time = t

        # ======================================================
        # ACTUATOR UPDATE
        # ======================================================
        if t - self.last_act_time >= self.dt_act:

            if self._current_plan is not None:

                if self._plan_index < len(self._current_plan):
                    self.last_control = self._current_plan[self._plan_index]
                    self._plan_index += 1
                else:
                    self.last_control = self._current_plan[-1]

            self.last_act_time = t

        # ======================================================
        # ZERO-ORDER HOLD OUTPUT
        # ======================================================
        return {
            "Fuel_rate_total": self.last_control,
        }
           