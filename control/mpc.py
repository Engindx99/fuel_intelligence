import casadi as ca
import numpy as np


class MasterMPC:

    def __init__(self):

        # =========================
        # MPC SETTINGS
        # =========================
        self.N = 10
        self.dt = 0.5
        self.T_target = 1450 + 273.15

        # =========================
        # SCALING
        # =========================
        self.T_scale = 2000.0
        self.F_scale = 8.0
        self.O2_scale = 6.0

        # =========================
        # FUEL LHV
        # =========================
        self.LHV = {"Petcoke": 32e6, "Lignite": 18e6, "RDF_Fuel": 25e6}

        # radiation coefficient (NEW FIX)
        self.rad_k = 1.2e-12

        self.opti = None
        self.prev_sol = None
        self._build_problem()

    # ======================================================
    # BUILD OPTIMIZATION GRAPH
    # ======================================================
    def _build_problem(self):

        self.opti = ca.Opti()

        # STATES
        self.Tg = self.opti.variable(self.N + 1)
        self.Ts = self.opti.variable(self.N + 1)
        self.Tw = self.opti.variable(self.N + 1)

        # CONTROLS
        self.Fuel = self.opti.variable(self.N)
        self.p = self.opti.variable(self.N)
        self.l = self.opti.variable(self.N)
        self.a = self.opti.variable(self.N)
        self.O2 = self.opti.variable(self.N)

        cost = 0

        for k in range(self.N):

            # =========================
            # MIXTURE
            # =========================
            mix_sum = self.p[k] + self.l[k] + self.a[k] + 1e-6

            p = self.p[k] / mix_sum
            l = self.l[k] / mix_sum
            a = self.a[k] / mix_sum

            LHV_mix = (
                p * self.LHV["Petcoke"]
                + l * self.LHV["Lignite"]
                + a * self.LHV["RDF_Fuel"]
            )

            # =========================
            # O2 EFFECT
            # =========================
            eta = ca.exp(-0.08 * (self.O2[k] - 3.5) ** 2)

            # =========================
            # HEAT INPUT
            # =========================
            Q_in = (self.Fuel[k] * self.F_scale) * LHV_mix * eta

            # =========================
            # DYNAMICS (FIXED BALANCE)
            # =========================
            loss_conv = 0.006 * (self.Tg[k] - self.Ts[k])
            loss_rad = self.rad_k * (self.Tg[k] ** 4 - (300 + 273.15) ** 4)

            Tg_next = self.Tg[k] + self.dt * (
                2.5e-8 * Q_in / 1e7 - loss_conv - loss_rad
            )

            Ts_next = self.Ts[k] + self.dt * (
                0.008 * (self.Tg[k] - self.Ts[k])  # stronger coupling
            )

            Tw_next = self.Tw[k] + self.dt * (0.0015 * (self.Tg[k] - self.Tw[k]))

            self.opti.subject_to(self.Tg[k + 1] == Tg_next)
            self.opti.subject_to(self.Ts[k + 1] == Ts_next)
            self.opti.subject_to(self.Tw[k + 1] == Tw_next)

            # =========================
            # COST
            # =========================
            err = self.Tg[k] - self.T_target

            cost += 3e-4 * err**2  #  stronger tracking

            cost += 1e-4 * self.Fuel[k] ** 2
            cost += 1e-2 * (self.O2[k] - 3.5) ** 2

            # smoothness
            if k > 0:
                cost += 5e-2 * (self.Fuel[k] - self.Fuel[k - 1]) ** 2
                cost += 1e-2 * (self.O2[k] - self.O2[k - 1]) ** 2

        # terminal cost (stronger)
        cost += 80 * (self.Tg[self.N] - self.T_target) ** 2

        self.opti.minimize(cost)

        # =========================
        # BOUNDS
        # =========================
        self.opti.subject_to(self.Fuel >= 0.5)
        self.opti.subject_to(self.Fuel <= 8.0)

        self.opti.subject_to(self.p >= 0.01)
        self.opti.subject_to(self.l >= 0.01)
        self.opti.subject_to(self.a >= 0.01)

        self.opti.subject_to(self.O2 >= 2.2)
        self.opti.subject_to(self.O2 <= 5.8)

        # =========================
        # INITIAL CONDITIONS
        # =========================
        self.Tg0 = self.opti.parameter()
        self.Ts0 = self.opti.parameter()
        self.Tw0 = self.opti.parameter()

        self.opti.subject_to(self.Tg[0] == self.Tg0)
        self.opti.subject_to(self.Ts[0] == self.Ts0)
        self.opti.subject_to(self.Tw[0] == self.Tw0)

        # =========================
        # SOLVER
        # =========================
        self.opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.max_iter": 120,
                "ipopt.tol": 1e-4,
                "ipopt.linear_solver": "mumps",
                "ipopt.mu_strategy": "adaptive",
            },
        )

    # ======================================================
    def _reset(self):
        self._build_problem()

    # ======================================================
    def _warm_start(self):
        if self.prev_sol is None:
            return

        try:
            sol = self.prev_sol

            for k in range(self.N):
                self.opti.set_initial(self.Fuel[k], sol.value(self.Fuel[k]))
                self.opti.set_initial(self.O2[k], sol.value(self.O2[k]))
                self.opti.set_initial(self.p[k], sol.value(self.p[k]))
                self.opti.set_initial(self.l[k], sol.value(self.l[k]))
                self.opti.set_initial(self.a[k], sol.value(self.a[k]))
        except:
            pass

    # ======================================================
    def compute_control(self, state):

        try:
            Tg0 = float(np.mean(state.Tg_burning))
            Ts0 = float(np.mean(state.Ts_burning))
            Tw0 = float(np.mean(state.Tw_burning))

            self.opti.set_value(self.Tg0, Tg0)
            self.opti.set_value(self.Ts0, Ts0)
            self.opti.set_value(self.Tw0, Tw0)

            self._warm_start()

            sol = self.opti.solve()
            self.prev_sol = sol

            return {
                "Fuel_rate": float(sol.value(self.Fuel[0])),
                "Petcoke": float(sol.value(self.p[0])),
                "Lignite": float(sol.value(self.l[0])),
                "RDF_Fuel": float(sol.value(self.a[0])),
                "O2": float(sol.value(self.O2[0])),
            }

        except Exception as e:
            print("MPC FAILED → RESETTING OPTIMIZER:", str(e))
            self._reset()

            return {
                "Fuel_rate": 3.5,
                "Petcoke": 0.33,
                "Lignite": 0.33,
                "RDF_Fuel": 0.33,
                "O2": 3.5,
            }
