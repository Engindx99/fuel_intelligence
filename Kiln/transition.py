import numpy as np
from physics.physics import solid_mass_flow
from physics.physics import fuel_heat_release
from physics.physics import cp_gas, h_gas
from physics.physics import wall_thermal_resistance
from physics.physics import residence_time
from physics.physics import gas_axial_velocity
from physics.physics import heat_transfer
from physics.physics import radiation
from physics.physics import interfacial_areas
from physics.physics import kiln_geometry
from physics.physics import solid_axial_velocity
from physics.physics import thermal_capacities
from physics.physics import wall_geometry
from physics.physics import wall_losses
from physics.physics import gas_mass_balance
from physics.physics import ZONE_HT_CONFIG

class Transition:

    def __init__(self, N=5, L=25.0):

        self.N = N
        self.L = L
        self.dz = L / N

        # ================= ZONE =================
        self.zone = "transition"

        # ================= GEOMETRY =================
        self.D = 4.2
        self.A_cross = np.pi * self.D**2 / 4.0
        self.V_total = self.A_cross * self.L
        self.V_cell = self.V_total / self.N

        # ================= INTERFACIAL AREA =================
        self.epsilon_bed = 0.35
        self.k_interfacial = 1.0

        a_gs_base = 6.0 * (1.0 - self.epsilon_bed) / self.D

        self.a_gs = self.k_interfacial * a_gs_base
        self.a_ws = 0.6 * self.a_gs

        # ================= WALL GEOMETRY =================
        self.wall_perimeter = np.pi * self.D
        self.A_wall = self.wall_perimeter * self.L
        self.A_wall_cell = self.A_wall / self.N
        self.a_gw = self.A_wall_cell / self.V_cell

        # ================= REFRACTORY =================
        self.refractory_thickness = 0.05
        self.refractory_conductivity = 1.8

        self.V_wall = self.A_wall * self.refractory_thickness
        self.V_wall_cell = self.V_wall / self.N

        # ================= EXTERNAL WALL =================
        self.h_ext = 12.0
        self.T_ref = 298.15
        self.T_amb = 300.0

        # ================= PROPERTIES =================
        self.rho_g = 0.30
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ================= BED =================
        self.fill_fraction = 0.10

        # ================= FLOW =================
        self.u_g = 0.0
        self.u_s = 0.0

        # ================= HEAT TRANSFER =================
        cfg = ZONE_HT_CONFIG[self.zone]

        self.hv_gs = cfg["hv_gs"]
        self.hv_gw = cfg["hv_gw"]
        self.hv_ws = cfg["hv_ws"]

        # ================= NUMERICAL =================
        self.eps = 1e-9

        # ================= BUFFERS =================
        self._dTg_dz = np.zeros(N)
        self._dTs_dz = np.zeros(N)

        # ================= CACHE =================
        self._rho_g_Vcell_Cp_g = self.rho_g * self.V_cell * self.Cp_g
        self._rho_s_Vcell_Cp_s = self.rho_s * self.V_cell * self.Cp_s
        self._rho_wall_Vwall_cell_Cp = self.rho_wall * self.V_wall_cell * self.Cp_wall
        
        # ======================================================
    def thermal_step(self, Tg, Ts, Tw, state, dt):

        # ======================================================
        # FLOW / MASS FLOW
        # ======================================================

        u_g = state.u_g
        u_s = state.u_s

        m_dot_g = state.m_dot_g
        m_dot_s = state.m_dot_s

        # ======================================================
        # SOLID THERMAL CAPACITY
        # ======================================================

        Cp_s = self.Cp_s

        Cs = (
            m_dot_s
            * Cp_s
        )

        # ======================================================
        # INLET TEMPERATURES FROM ENTHALPY
        #
        # Burning -> Transition
        #
        # Gas inlet:
        #   i = N-1
        #
        # Solid inlet:
        #   i = 0
        # ======================================================

        Tg_in = self.gas_inlet_temperature_from_enthalpy(
            state.Hgas_transition_in,
            state,
        )

        Ts_in = self.solid_inlet_temperature_from_enthalpy(
            state.Hsolid_transition_in,
            state,
        )

        # ======================================================
        # GEOMETRY
        # ======================================================

        N = len(Tg)

        V_cell = self.V_cell

        # ======================================================
        # HEAT TRANSFER COEFFICIENTS
        # ======================================================

        hv_gs = self.hv_gs
        hv_gw = self.hv_gw
        hv_ws = self.hv_ws

        a_gs = self.a_gs
        a_gw = self.a_gw
        a_ws = self.a_ws

        K_gs = hv_gs * a_gs
        K_gw = hv_gw * a_gw
        K_ws = hv_ws * a_ws

        # ======================================================
        # WALL THERMAL RESISTANCE
        # ======================================================

        R_ref, R_conv, R_total = wall_thermal_resistance(
            refractory_thickness=self.refractory_thickness,
            refractory_conductivity=self.refractory_conductivity,
            h_ext=self.h_ext,
            A_wall_cell=self.A_wall_cell,
        )

        # ======================================================
        # STEADY-STATE ITERATION
        # ======================================================

        max_iter = 100
        tol = 1.0e-6
        relaxation = 0.5

        Tg_iter = np.asarray(Tg, dtype=float).copy()
        Ts_iter = np.asarray(Ts, dtype=float).copy()
        Tw_iter = np.asarray(Tw, dtype=float).copy()

        converged = False
        error = np.inf

        # ======================================================
        # PICARD ITERATION
        # ======================================================

        for iteration in range(max_iter):

            # ==================================================
            # RADIATION
            # ==================================================

            q_gs_rad = radiation(
                Tg_iter,
                Ts_iter,
                zone=self.zone,
                area=a_gs,
            )

            q_gw_rad = radiation(
                Tg_iter,
                Tw_iter,
                zone=self.zone,
                area=a_gw,
            )

            q_ws_rad = radiation(
                Ts_iter,
                Tw_iter,
                zone=self.zone,
                area=a_ws,
            )

            # ==================================================
            # LINEAR SYSTEM
            # ==================================================

            n_unknowns = 3 * N

            A = np.zeros(
                (n_unknowns, n_unknowns)
            )

            b = np.zeros(n_unknowns)

            row = 0

            # ==================================================
            # CELL EQUATIONS
            # ==================================================

            for i in range(N):

                Tg_i = i
                Ts_i = N + i
                Tw_i = 2 * N + i

                # ==================================================
                # GAS LOCAL THERMODYNAMIC PROPERTIES
                # ==================================================

                Cp_g_i = cp_gas(
                    Tg_iter[i]
                )

                Cg_i = (
                    m_dot_g
                    * Cp_g_i
                )

                # ==================================================
                # GAS ENERGY BALANCE
                #
                # Gas flow:
                #
                # N-1 -> N-2 -> ... -> 1 -> 0
                #
                # Inlet  = N-1
                # Outlet = 0
                #
                # Enthalpy formulation:
                #
                # m_dot_g [h(Tg_i) - h(Tg_up)]
                # + Q_gs
                # + Q_gw
                # = 0
                # ==================================================

                h_i_iter = h_gas(
                    Tg_iter[i],
                    self.T_ref
                )

                h_linear_const_i = (
                    h_i_iter
                    - Cp_g_i * Tg_iter[i]
                )

                # --------------------------------------------------
                # GAS -> SOLID
                # GAS -> WALL
                # --------------------------------------------------

                A[row, Tg_i] += (
                    Cg_i
                    + V_cell * K_gs
                    + V_cell * K_gw
                )

                A[row, Ts_i] += (
                    -V_cell * K_gs
                )

                A[row, Tw_i] += (
                    -V_cell * K_gw
                )

                # --------------------------------------------------
                # RADIATION GAS SINK
                # --------------------------------------------------

                radiation_gas_sink = (
                    V_cell
                    * (
                        q_gs_rad[i]
                        + q_gw_rad[i]
                    )
                )

                # --------------------------------------------------
                # GAS INLET / UPSTREAM CELL
                # --------------------------------------------------

                if i == N - 1:

                    # ----------------------------------------------
                    # GAS INLET FROM BURNING
                    # ----------------------------------------------

                    h_in = h_gas(
                        Tg_in,
                        self.T_ref
                    )

                    b[row] = (
                        -radiation_gas_sink
                        + m_dot_g * h_in
                        - m_dot_g * h_linear_const_i
                    )

                else:

                    # ----------------------------------------------
                    # UPSTREAM GAS CELL
                    #
                    # Gas moves:
                    #
                    # i+1 -> i
                    # ----------------------------------------------

                    Tg_up_i = i + 1

                    Cp_g_up = cp_gas(
                        Tg_iter[i + 1]
                    )

                    h_up_iter = h_gas(
                        Tg_iter[i + 1],
                        self.T_ref
                    )

                    h_linear_const_up = (
                        h_up_iter
                        - Cp_g_up * Tg_iter[i + 1]
                    )

                    A[row, Tg_up_i] += (
                        -m_dot_g * Cp_g_up
                    )

                    b[row] = (
                        -radiation_gas_sink
                        - m_dot_g * h_linear_const_i
                        + m_dot_g * h_linear_const_up
                    )

                row += 1

                # ==================================================
                # SOLID ENERGY BALANCE
                #
                # Solid flow:
                #
                # 0 -> 1 -> ... -> N-2 -> N-1
                #
                # Inlet  = 0
                # Outlet = N-1
                # ==================================================

                A[row, Ts_i] += (
                    Cs
                    + V_cell * K_gs
                    + V_cell * K_ws
                )

                A[row, Tg_i] += (
                    -V_cell * K_gs
                )

                A[row, Tw_i] += (
                    -V_cell * K_ws
                )

                # --------------------------------------------------
                # RADIATION SOLID SOURCE
                # --------------------------------------------------

                radiation_solid_source = (
                    V_cell
                    * (
                        q_gs_rad[i]
                        - q_ws_rad[i]
                    )
                )

                # --------------------------------------------------
                # SOLID INLET / UPSTREAM CELL
                # --------------------------------------------------

                if i == 0:

                    # ----------------------------------------------
                    # SOLID INLET FROM BURNING
                    # ----------------------------------------------

                    b[row] = (
                        Cs * Ts_in
                        + radiation_solid_source
                    )

                else:

                    # ----------------------------------------------
                    # UPSTREAM SOLID CELL
                    #
                    # Solid moves:
                    #
                    # i-1 -> i
                    # ----------------------------------------------

                    Ts_up_i = N + i - 1

                    A[row, Ts_up_i] += (
                        -Cs
                    )

                    b[row] = (
                        radiation_solid_source
                    )

                row += 1

                # ==================================================
                # WALL ENERGY BALANCE
                # ==================================================

                A[row, Tg_i] += (
                    V_cell * K_gw
                )

                A[row, Ts_i] += (
                    V_cell * K_ws
                )

                A[row, Tw_i] += (
                    -V_cell * K_gw
                    -V_cell * K_ws
                    -1.0 / R_total
                )

                # --------------------------------------------------
                # WALL RADIATION SOURCE
                # --------------------------------------------------

                radiation_wall_source = (
                    V_cell
                    * (
                        q_gw_rad[i]
                        + q_ws_rad[i]
                    )
                )

                b[row] = (
                    -self.T_amb / R_total
                    - radiation_wall_source
                )

                row += 1

            # ======================================================
            # SOLVE LINEAR SYSTEM
            # ======================================================

            x_solution = np.linalg.solve(
                A,
                b,
            )

            Tg_new = x_solution[:N]

            Ts_new = x_solution[
                N:2 * N
            ]

            Tw_new = x_solution[
                2 * N:3 * N
            ]

            # ======================================================
            # CONVERGENCE
            # ======================================================

            error = max(
                np.max(
                    np.abs(
                        Tg_new - Tg_iter
                    )
                ),

                np.max(
                    np.abs(
                        Ts_new - Ts_iter
                    )
                ),

                np.max(
                    np.abs(
                        Tw_new - Tw_iter
                    )
                ),
            )

            # ==================================================
            # RELAXATION
            # ==================================================

            Tg_iter = (
                relaxation * Tg_new
                + (1.0 - relaxation)
                * Tg_iter
            )

            Ts_iter = (
                relaxation * Ts_new
                + (1.0 - relaxation)
                * Ts_iter
            )

            Tw_iter = (
                relaxation * Tw_new
                + (1.0 - relaxation)
                * Tw_iter
            )

            # ==================================================
            # CONVERGED
            # ==================================================

            if error < tol:

                converged = True

                break

        # ======================================================
        # CONVERGENCE WARNING
        # ======================================================

        if not converged:

            print(
                "WARNING: Transition radiation iteration "
                f"did not converge after {max_iter} iterations. "
                f"Final error = {error:.6e} K"
            )

        # ======================================================
        # FINAL STEADY-STATE TEMPERATURES
        # ======================================================

        Tg_ss = Tg_iter
        Ts_ss = Ts_iter
        Tw_ss = Tw_iter

        # ======================================================
        # FINAL RADIATION
        # ======================================================

        q_gs_rad = radiation(
            Tg_ss,
            Ts_ss,
            zone=self.zone,
            area=a_gs,
        )

        q_gw_rad = radiation(
            Tg_ss,
            Tw_ss,
            zone=self.zone,
            area=a_gw,
        )

        q_ws_rad = radiation(
            Ts_ss,
            Tw_ss,
            zone=self.zone,
            area=a_ws,
        )

        # ======================================================
        # FINAL CONVECTION
        # ======================================================

        q_gs_conv = (
            K_gs
            * (
                Tg_ss
                - Ts_ss
            )
        )

        q_gw_conv = (
            K_gw
            * (
                Tg_ss
                - Tw_ss
            )
        )

        q_ws_conv = (
            K_ws
            * (
                Ts_ss
                - Tw_ss
            )
        )

        # ======================================================
        # TOTAL CELL HEAT TRANSFER
        # ======================================================

        q_gs_cell = (
            q_gs_conv
            + q_gs_rad
        )

        q_gw_cell = (
            q_gw_conv
            + q_gw_rad
        )

        q_ws_cell = (
            q_ws_conv
            + q_ws_rad
        )

        # ======================================================
        # WALL LOSS
        # ======================================================

        Q_loss_cell = (
            Tw_ss
            - self.T_amb
        ) / R_total

        wall_loss = np.sum(
            Q_loss_cell
        )

        # ======================================================
        # WALL DEBUG
        # ======================================================

        wall_debug = {

            "R_ref": float(
                R_ref
            ),

            "R_conv": float(
                R_conv
            ),

            "R_total": float(
                R_total
            ),

            "q_loss_mean": float(
                np.mean(
                    Q_loss_cell
                    / V_cell
                )
            ),

            "wall_loss_total": float(
                wall_loss
            ),

            "A_wall": float(
                self.A_wall
            ),

            "A_wall_cell": float(
                self.A_wall_cell
            ),

            "V_cell": float(
                V_cell
            ),

            "N": int(N),
        }

        # ======================================================
        # ENTHALPY DEBUG
        # ======================================================

        Hg_in = (
            state.Hgas_transition_in
        )

        Hs_in = (
            state.Hsolid_transition_in
        )

        Hg_out = (
            m_dot_g
            * h_gas(
                Tg_ss[0],
                self.T_ref
            )
        )

        Hs_out = (
            m_dot_s
            * Cp_s
            * (
                Ts_ss[-1]
                - self.T_ref
            )
        )

        # ======================================================
        # HEAT TRANSFER TOTALS
        # ======================================================

        Q_gs = (
            V_cell
            * np.sum(q_gs_cell)
        )

        Q_gw = (
            V_cell
            * np.sum(q_gw_cell)
        )

        Q_ws = (
            V_cell
            * np.sum(q_ws_cell)
        )

        # ======================================================
        # ENERGY BALANCE
        #
        # Hg_in + Hs_in
        # =
        # Hg_out + Hs_out + wall_loss
        # ======================================================

        total_energy_balance = (
            Hg_in
            + Hs_in
            - Hg_out
            - Hs_out
            - wall_loss
        )

        # ======================================================
        # DEBUG
        # ======================================================

        print()
        print(
            "========== TRANSITION STEADY STATE =========="
        )

        print(
            f"Tg_in       = {Tg_in:.3f} K"
        )

        print(
            f"Tg_out      = {Tg_ss[0]:.3f} K"
        )

        print(
            f"Ts_in       = {Ts_in:.3f} K"
        )

        print(
            f"Ts_out      = {Ts_ss[-1]:.3f} K"
        )

        print(
            f"Tw_out      = {Tw_ss[-1]:.3f} K"
        )

        print()

        print(
            f"Hg_in       = {Hg_in:.6e} W"
        )

        print(
            f"Hg_out      = {Hg_out:.6e} W"
        )

        print(
            f"Hs_in       = {Hs_in:.6e} W"
        )

        print(
            f"Hs_out      = {Hs_out:.6e} W"
        )

        print()

        print(
            f"Q_gs        = {Q_gs:.3f} W"
        )

        print(
            f"Q_gw        = {Q_gw:.3f} W"
        )

        print(
            f"Q_ws        = {Q_ws:.3f} W"
        )

        print(
            f"Q_wall_loss = {wall_loss:.3f} W"
        )

        print()

        print(
            "--- ENERGY TRANSFER CHECK ---"
        )

        print(
            f"Delta H gas   = "
            f"{Hg_out - Hg_in:.6e} W"
        )

        print(
            f"Delta H solid = "
            f"{Hs_out - Hs_in:.6e} W"
        )

        print(
            f"Gas expected  = "
            f"{-Q_gs - Q_gw:.6e} W"
        )

        print(
            f"Solid expected = "
            f"{Q_gs - Q_ws:.6e} W"
        )

        print()

        print(
            "--- RADIATION ---"
        )

        print(
            f"Q_gs_rad = "
            f"{V_cell * np.sum(q_gs_rad):.3f} W"
        )

        print(
            f"Q_gw_rad = "
            f"{V_cell * np.sum(q_gw_rad):.3f} W"
        )

        print(
            f"Q_ws_rad = "
            f"{V_cell * np.sum(q_ws_rad):.3f} W"
        )

        print(
            f"Iterations = {iteration + 1}"
        )

        print(
            f"Rad. error = {error:.6e} K"
        )

        print()

        print(
            "--- ENERGY BALANCE ---"
        )

        print(
            f"Total balance = "
            f"{total_energy_balance:.6e} W"
        )

        print()

        print(
            "--- CELL TEMPERATURES ---"
        )

        for i in range(N):

            print(
                f"cell {i}: "
                f"Tg={Tg_ss[i]:.2f} K, "
                f"Ts={Ts_ss[i]:.2f} K, "
                f"Tw={Tw_ss[i]:.2f} K, "
                f"Qgs={V_cell * q_gs_cell[i] / 1e6:.3f} MW, "
                f"Qgw={V_cell * q_gw_cell[i] / 1e6:.3f} MW, "
                f"Qws={V_cell * q_ws_cell[i] / 1e6:.3f} MW, "
                f"Qloss={Q_loss_cell[i] / 1e6:.3f} MW"
            )

        print(
            "==============================================="
        )

        return (
            Tg_ss,
            Ts_ss,
            Tw_ss,
            wall_loss,
            wall_debug,
        )


    # ======================================================
    # STATE UPDATE
    # ======================================================
    def apply(self, state, dt):

        # ======================================================
        # STATE INTEGRITY CHECK
        # ======================================================
        if not isinstance(state.Tg_transition, np.ndarray):
            raise TypeError("Tg_transition must be np.ndarray")

        if state.Tg_transition.shape != (5,):
            raise ValueError(
                f"Transition shape corrupted: {state.Tg_transition.shape}"
            )

        # ======================================================
        # FLOW VARIABLES
        # ======================================================
        state.u_g = getattr(
            state,
            "u_g",
            self.u_g,
        )

        state.u_s = getattr(
            state,
            "u_s",
            self.u_s,
        )

        state.m_dot_g = getattr(
            state,
            "m_dot_g",
            0.0,
        )

        state.m_dot_s = getattr(
            state,
            "m_dot_s",
            0.0,
        )

        # ======================================================
        # INLET ENTHALPY FROM BURNING
        # ======================================================
        state.Hgas_transition_in = (
            state.Hgas_burning_out
        )

        state.Hsolid_transition_in = (
            state.Hsolid_burning_out
        )

        # ======================================================
        # INLET TEMPERATURE FROM ENTHALPY
        # ======================================================
        Tg_in = self.gas_inlet_temperature_from_enthalpy(
            state.Hgas_transition_in,
            state,
        )

        Ts_in = self.solid_inlet_temperature_from_enthalpy(
            state.Hsolid_transition_in,
            state,
        )

        # ======================================================
        # APPLY INLET BOUNDARY CONDITIONS
        # ======================================================


        # ======================================================
        # STEADY-STATE THERMAL STEP
        # ======================================================
        (
            Tg,
            Ts,
            Tw,
            wall_loss,
            wall_debug,
        ) = self.thermal_step(
            state.Tg_transition,
            state.Ts_transition,
            state.Tw_transition,
            state,
            dt,
        )

        # ======================================================
        # UPDATE TEMPERATURE STATES
        # ======================================================
        state.Tg_transition = Tg
        state.Ts_transition = Ts
        state.Tw_transition = Tw

        # ======================================================
        # TRANSITION CHEMISTRY
        # ======================================================
        # Eğer transition reaksiyonları eklenecekse burada çalışacak.
        # state = self.chemistry.apply_transition(state)

        # ======================================================
        # UPDATE ENTHALPY STATES
        # ======================================================
        state.Hg_transition = (
            state.m_dot_g
            * self.Cp_g
            * (
                state.Tg_transition
                - self.T_ref
            )
        )

        state.Hs_transition = (
            state.m_dot_s
            * self.Cp_s
            * (
                state.Ts_transition
                - self.T_ref
            )
        )

        # ======================================================
        # WALL LOSS
        # ======================================================
        state.Wall_loss_transition = float(
            wall_loss
        )

        # ======================================================
        # WALL DEBUG
        # ======================================================
        if wall_debug is None:
            wall_debug = {}

        state.wall_debug_transition = {
            "q_loss_mean": wall_debug.get(
                "q_loss_mean",
                0.0,
            ),
            "q_loss_total": wall_debug.get(
                "wall_loss_total",
                0.0,
            ),
            "A_wall": wall_debug.get(
                "A_wall",
                0.0,
            ),
            "V_cell": wall_debug.get(
                "V_cell",
                0.0,
            ),
            "N": wall_debug.get(
                "N",
                0,
            ),
        }

        state.q_loss_mean_transition = (
            state.wall_debug_transition[
                "q_loss_mean"
            ]
        )

        state.A_wall_transition = (
            state.wall_debug_transition[
                "A_wall"
            ]
        )

        state.V_cell_transition = (
            state.wall_debug_transition[
                "V_cell"
            ]
        )

        state.N_transition = (
            state.wall_debug_transition[
                "N"
            ]
        )

        # ======================================================
        # ENERGY OUT
        # ======================================================
        state.Hgas_transition_out = (
            self.gas_enthalpy_out(
                state.Hg_transition
            )
        )

        state.Hsolid_transition_out = (
            self.solid_enthalpy_out(
                state.Hs_transition
            )
        )

        # ======================================================
        # STEADY-STATE STORED ENERGY
        # ======================================================
        state.Transition_gas_stored = 0.0
        state.Transition_solid_stored = 0.0
        state.Transition_wall_stored = 0.0

        state.Transition_stored_energy_change = 0.0

        # ======================================================
        # STEADY-STATE ENERGY BALANCE
        # ======================================================
        state.Transition_energy_balance = (
            state.Hgas_transition_in
            + state.Hsolid_transition_in
            - state.Hgas_transition_out
            - state.Hsolid_transition_out
            - state.Wall_loss_transition
        )

        return state


    # ======================================================
    # TEMPERATURE FROM INLET ENTHALPY
    # ======================================================
    def gas_inlet_temperature_from_enthalpy(
        self,
        H,
        state,
    ):

        m_dot_g = state.m_dot_g

        if m_dot_g <= self.eps:
            return self.T_ref

        h_target = H / m_dot_g

        T_low = self.T_ref
        T_high = 4000.0

        for _ in range(100):

            T_mid = 0.5 * (T_low + T_high)

            h_mid = h_gas(
                T_mid,
                self.T_ref
            )

            if h_mid < h_target:
                T_low = T_mid
            else:
                T_high = T_mid

        return 0.5 * (T_low + T_high)


    def solid_inlet_temperature_from_enthalpy(
        self,
        H,
        state,
    ):

        return (
            H
            / (
                state.m_dot_s
                * self.Cp_s
                + self.eps
            )
            + self.T_ref
        )


    # ======================================================
    # GAS ENTHALPY TO NEXT ZONE
    # ======================================================
    def gas_enthalpy_out(self, Hg):

        return Hg[0]


    # ======================================================
    # SOLID ENTHALPY TO NEXT ZONE
    # ======================================================
    def solid_enthalpy_out(self, Hs):

        return Hs[-1]