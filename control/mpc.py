import casadi as ca
import numpy as np
from Kiln.Burning import Burning


class MasterMPC:

    def __init__(self):

        self.Np = 60
        self.Nc = 10
        self.dt = 0.05

        self.Tg_ref = 1550.0
        self.Ts_ref = 1450.0

        self.model = Burning(N=5)

        self.opti = ca.Opti()
        self.prev_sol = None

        self._build()

    # ======================================================
    # 🔥 STATE PROJECTION (SAFE + ROBUST)
    # ======================================================
    def _project_state(self, x, N_target):

        x = np.asarray(x).flatten()
        N_src = x.shape[0]

        if N_src == N_target:
            return x

        # interpolation (better than indexing)
        src_idx = np.linspace(0, 1, N_src)
        tgt_idx = np.linspace(0, 1, N_target)

        return np.interp(tgt_idx, src_idx, x)

    # ======================================================
    # 🧾 CLEAN LOGGING (PHYSICS STYLE)
    # ======================================================
    def _log(self, t, sol, state, fuel):

        idx = len(state.Tg_burning) // 2

        Tg = state.Tg_burning[idx] - 273.15
        Ts = state.Ts_burning[idx] - 273.15
        Tw = state.Tw_burning[idx] - 273.15

        print(
            f"t={t:7.1f}s | "
            f"Fuel={fuel:4.2f} | "
            f"Tg[{idx}]={Tg:7.2f}°C | "
            f"Ts[{idx}]={Ts:7.2f}°C | "
            f"Tw[{idx}]={Tw:7.2f}°C"
        )

    # ======================================================
    def _build(self):

        N = self.model.N
        eps = self.model.eps

        # ================= STATES =================
        self.Tg = self.opti.variable(N, self.Np + 1)
        self.Ts = self.opti.variable(N, self.Np + 1)
        self.Tw = self.opti.variable(N, self.Np + 1)

        # ================= CONTROL =================
        self.Fuel = self.opti.variable(self.Nc)

        # ================= PARAMETERS =================
        self.Tg0 = self.opti.parameter(N)
        self.Ts0 = self.opti.parameter(N)
        self.Tw0 = self.opti.parameter(N)

        self.opti.subject_to(self.Tg[:, 0] == self.Tg0)
        self.opti.subject_to(self.Ts[:, 0] == self.Ts0)
        self.opti.subject_to(self.Tw[:, 0] == self.Tw0)

        # ================= INIT GUESS =================
        self.opti.set_initial(self.Tg, 1500.0)
        self.opti.set_initial(self.Ts, 1400.0)
        self.opti.set_initial(self.Tw, 800.0)
        self.opti.set_initial(self.Fuel, 4.0)

        cost = 0

        def u(k):
            return min(k, self.Nc - 1)

        # ================= FUEL MIX =================
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

            # 🔥 HARD SAFETY CLAMP
            Fuel_k = ca.fmax(2.0, ca.fmin(6.0, Fuel_k))

            # ================= HEAT SOURCE =================
            Q_in = Fuel_k * LHV_mix * eta
            q_vol = Q_in / (self.model.V_total + eps)

            # ================= GRADIENTS =================
            dTg_dz = ca.vertcat(
                self.Tg[0, k], (self.Tg[1:, k] - self.Tg[:-1, k]) / self.model.dz
            )

            dTs_dz = ca.vertcat(
                self.Ts[0, k], (self.Ts[1:, k] - self.Ts[:-1, k]) / self.model.dz
            )

            # ================= HEAT TRANSFER =================
            q_gs = (
                self.model.hv_gs * self.model.a_gs * (self.Tg[:, k] - self.Ts[:, k])
            ) / self.model.V_cell

            q_gw = (
                self.model.hv_gw * self.model.a_gw * (self.Tg[:, k] - self.Tw[:, k])
            ) / self.model.V_cell

            q_ws = (
                self.model.hv_ws * self.model.a_ws * (self.Ts[:, k] - self.Tw[:, k])
            ) / self.model.V_cell

            # ================= CAPACITIES =================
            C_g = self.model.rho_g * self.model.V_cell * self.model.Cp_g
            C_s = self.model.rho_s * self.model.V_cell * self.model.Cp_s
            C_w = self.model.rho_wall * self.model.V_wall * self.model.Cp_wall

            q_loss = (
                self.model.h_ext
                * self.model.A_wall
                * (self.Tw[:, k] - self.model.T_amb)
            ) / (self.model.V_cell + eps)

            # ================= DYNAMICS =================
            Tg_next = self.Tg[:, k] + self.dt * (
                -self.model.u_g * dTg_dz + (q_vol - q_gs - q_gw) / C_g
            )

            Ts_next = self.Ts[:, k] + self.dt * (
                -self.model.u_s * dTg_dz + (q_gs - q_ws) / C_s
            )

            Tw_next = self.Tw[:, k] + self.dt * ((q_gw + q_ws - q_loss) / C_w)

            self.opti.subject_to(self.Tg[:, k + 1] == Tg_next)
            self.opti.subject_to(self.Ts[:, k + 1] == Ts_next)
            self.opti.subject_to(self.Tw[:, k + 1] == Tw_next)

            # ================= COST =================
            Tg_mean = ca.sum1(self.Tg[:, k]) / N
            Ts_mean = ca.sum1(self.Ts[:, k]) / N

            cost += 1e-4 * (Tg_mean - self.Tg_ref) ** 2
            cost += 1e-4 * (Ts_mean - self.Ts_ref) ** 2
            cost += 1e-3 * Fuel_k**2

            if k > 0:
                cost += 5e-2 * (Fuel_k - self.Fuel[u(k - 1)]) ** 2

        # ================= TERMINAL =================
        Tg_end = ca.sum1(self.Tg[:, self.Np]) / N
        cost += 10.0 * (Tg_end - self.Tg_ref) ** 2

        self.opti.minimize(cost)

        # ================= BOUNDS =================
        self.opti.subject_to(self.Fuel >= 2.0)
        self.opti.subject_to(self.Fuel <= 6.0)

        # ================= SOLVER =================
        self.opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": 80,
                "ipopt.tol": 1e-4,
                "ipopt.mu_strategy": "adaptive",
                "ipopt.linear_solver": "mumps",
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

        # ================= WARM START =================
        if self.prev_sol is not None:
            try:
                for k in range(self.Nc):
                    self.opti.set_initial(
                        self.Fuel[k], self.prev_sol.value(self.Fuel[k])
                    )
            except:
                pass

        sol = self.opti.solve()
        self.prev_sol = sol

        fuel = float(sol.value(self.Fuel[0]))

        # ================= CLEAN LOG =================
        self._log(t, sol, state, fuel)

        return {"Fuel_rate": fuel}
