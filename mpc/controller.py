import casadi as ca
import numpy as np
import os
import yaml
from core.state import *
from mpc.constraints import get_control_constraints, get_state_constraints


class MPCController:
    def __init__(self, N_horizon=15, dt_mpc=5.0):
        self.N = N_horizon
        self.dt = dt_mpc

        self.u_min, self.u_max = get_control_constraints()
        self.x_min, self.x_max = get_state_constraints()

        with open(os.path.join("configs", "model_config.yaml"), "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.solver = self._build_nlp_solver()

    # -------------------------------------------------
    # NLP SETUP
    # -------------------------------------------------

    def _build_nlp_solver(self):

        U = ca.MX.sym('U', N_CONTROLS, self.N)
        X = ca.MX.sym('X', N_STATES, self.N + 1)
        P = ca.MX.sym('P', N_STATES + N_STATES)

        x_init = P[:N_STATES]
        x_target = P[N_STATES:]

        obj = 0
        g = []

        g.append(X[:, 0] - x_init)

        Q_T = 10000.0
        Q_U = 0.001
        Q_DU = 1.0

        for k in range(self.N):

            st = X[:, k]
            con = U[:, k]

            # -------------------------
            # TRACKING & QUALITY COST
            # -------------------------
            obj += Q_T * ca.sumsqr(st[IDX_T_S] - x_target[IDX_T_S])
            
            # Klinker üretimini teşvik et (negatif maliyet = ödül)
            obj -= 5000.0 * st[IDX_C3S] 

            # fuel penalty
            obj += Q_U * ca.sumsqr(con[IDX_FUEL])

            # smooth control
            if k > 0:
                du = U[:, k] - U[:, k-1]
                obj += Q_DU * ca.sumsqr(du)

            # -------------------------
            # DYNAMICS
            # -------------------------
            f_rhs = self._predictive_model(st, con)
            st_next = st + self.dt * f_rhs

            g.append(X[:, k+1] - st_next)

        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1)
        )

        nlp_prob = {
            'f': obj,
            'x': opt_vars,
            'g': ca.vertcat(*g),
            'p': P
        }

        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 80,
            'ipopt.tol': 1e-5,
            'ipopt.linear_solver': 'mumps',
            'print_time': 0
        }

        return ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # -------------------------------------------------
    # PREDICTIVE MODEL (CASADI SAFE)
    # -------------------------------------------------

    def _predictive_model(self, x, u):
        # MPC uses a simplified lumped-parameter model for speed
        A_calc = self.cfg['physics']['calcination']['A']
        Ea_calc = self.cfg['physics']['calcination']['Ea']
        A_c3s = self.cfg['physics']['c3s_formation']['A']
        Ea_c3s = self.cfg['physics']['c3s_formation']['Ea']
        R = self.cfg['physics']['r_gas']
        LHV = self.cfg['thermal']['lhv_fuel']
        CP_S = self.cfg['thermal']['cp_solid']
        RHO_S = self.cfg['thermal']['rho_solid']

        T_s = ca.fmax(x[IDX_T_S], 300.0)
        T_g = ca.fmax(x[IDX_T_G], 500.0)

        # Kinetics
        k_calc = A_calc * ca.exp(-Ea_calc / (R * T_s))
        r_calc = k_calc * ca.fmax(x[IDX_CaCO3], 0.0)

        k_c3s = A_c3s * ca.exp(-Ea_c3s / (R * T_s))
        r_c3s = k_c3s * ca.fmax(x[IDX_CaO], 0.0) * ca.fmax(x[IDX_C2S], 0.0)

        rhs = ca.MX.zeros(N_STATES)

        # -------------------------
        # ENERGY (Lumped Approximation)
        # -------------------------
        # Match simulation scaling
        RHO_G = self.cfg['thermal']['rho_gas']
        CP_G = self.cfg['thermal']['cp_gas']
        RHO_CP_G = RHO_G * CP_G
        RHO_CP_S = RHO_S * CP_S

        V_KILN = self.cfg['kiln_geometry']['length'] * self.cfg['thermal']['kiln_cross_section']
        Q_fuel = u[IDX_FUEL] * LHV * 0.9 / V_KILN # scaled per unit vol
        
        # Heat transfer approximation (Convection + Radiation)
        h_gs = self.cfg['thermal']['h_gs_base']
        Q_conv = h_gs * (T_g - T_s)
        Q_rad = 5.67e-8 * 0.85 * (T_g**4 - T_s**4)
        Q_xfer = Q_conv + Q_rad

        # Advection (Fan effect): 
        # m_dot_gas * cp_g * (T_in - T_out) / Vol
        # Approximate m_dot_gas proportional to fan_rpm
        m_dot_gas = u[IDX_FAN] * 0.05 
        T_g_inlet = 500.0
        Q_adv = m_dot_gas * CP_G * (T_g_inlet - T_g) / V_KILN

        # Solid Temperature
        # Enthalpy: Calcination (+) cools, C3S formation (-) heats
        rhs[IDX_T_S] = (Q_xfer - 178000 * r_calc + 13000 * r_c3s) / RHO_CP_S
        
        # Gas Temperature
        rhs[IDX_T_G] = (Q_fuel - Q_xfer + Q_adv) / RHO_CP_G

        # -------------------------
        # MASS
        # -------------------------
        rhs[IDX_CaCO3] = -r_calc
        rhs[IDX_CaO]   =  r_calc - r_c3s
        rhs[IDX_C2S]   = -r_c3s
        rhs[IDX_C3S]   =  r_c3s

        return rhs

    # -------------------------------------------------
    # SOLVE
    # -------------------------------------------------

    def compute_action(self, x_measured, x_target, u_last):

        x0_guess = np.tile(x_measured, self.N + 1)
        u0_guess = np.tile(u_last, self.N)

        init_guess = np.concatenate([x0_guess, u0_guess])
        p_val = np.concatenate([x_measured, x_target])

        # Bounds
        lbx = np.concatenate([np.tile(self.x_min, self.N + 1), np.tile(self.u_min, self.N)])
        ubx = np.concatenate([np.tile(self.x_max, self.N + 1), np.tile(self.u_max, self.N)])

        sol = self.solver(x0=init_guess, p=p_val, lbx=lbx, ubx=ubx)

        u_opt = sol['x'][(self.N + 1) * N_STATES:]
        u_opt = np.array(u_opt).reshape(self.N, N_CONTROLS)

        # hard safety clamp
        u_opt[0] = np.clip(u_opt[0], self.u_min, self.u_max)

        return u_opt[0]