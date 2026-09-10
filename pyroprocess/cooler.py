import numpy as np

from physics.physics import heat_transfer
from physics.physics import interfacial_areas
from physics.physics import kiln_geometry
from physics.physics import wall_geometry
from physics.physics import wall_losses
from physics.physics import ZONE_HT_CONFIG
from physics.physics import wall_thermal_resistance
from physics.physics import cp_gas
from physics.physics import h_gas

class Cooler:

    def __init__(self, N=5, L=20.0):

        self.N = N
        self.L = L
        self.dz = L / N

        # ================= ZONE =================
        self.zone = "cooler"

        # ================= NUMERICAL =================
        self.eps = 1e-9

        # ================= GEOMETRY =================
        self.D = 4.2

        (
            self.A_cross,
            self.V_total,
            self.V_cell,
        ) = kiln_geometry(
            D=self.D,
            L=self.L,
            N=self.N,
        )

        # ================= INTERFACIAL AREA =================
        self.epsilon_bed = 0.35
        self.k_interfacial = 1.0

        (
            self.a_gs,
            self.a_ws,
        ) = interfacial_areas(
            D=self.D,
            epsilon_bed=self.epsilon_bed,
            k_interfacial=self.k_interfacial,
        )

        # ================= WALL GEOMETRY =================
        (
            self.wall_perimeter,
            self.A_wall,
            self.A_wall_cell,
            self.a_gw,
            self.V_wall,
        ) = wall_geometry(
            D=self.D,
            L=self.L,
            N=self.N,
            V_cell=self.V_cell,
        )

        # ================= REFRACTORY =================
        self.refractory_thickness = 0.15
        self.refractory_conductivity = 1.5

        # ================= EXTERNAL =================
        self.h_ext = 18.0

        # ================= THERMODYNAMIC REFERENCE =================
        self.T_ref = 298.15
        self.T_amb = 300.0

        # ================= PROPERTIES =================
        self.rho_g = 1.2
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1005.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ================= HEAT TRANSFER =================
        cfg = ZONE_HT_CONFIG[self.zone]

        self.hv_gs = cfg["hv_gs"]
        self.hv_gw = cfg["hv_gw"]
        self.hv_ws = cfg["hv_ws"]
        

    def thermal_step(self, Tg, Ts, Tw, state):

        # ======================================================
        # MASS FLOW
        # ======================================================

        m_dot_g = state.m_dot_g
        m_dot_s = state.m_dot_s

        N = self.N
        V_cell = self.V_cell

        # ======================================================
        # INLET BOUNDARY CONDITIONS
        # ======================================================

        Tg_in = state.Tg_cooler_in
        Ts_in = state.Ts_cooler_in

        # ======================================================
        # HEAT TRANSFER COEFFICIENTS
        # ======================================================

        K_gs = self.hv_gs * self.a_gs
        K_gw = self.hv_gw * self.a_gw
        K_ws = self.hv_ws * self.a_ws

        # ======================================================
        # WALL THERMAL RESISTANCE
        # ======================================================

        _, _, R_total = wall_thermal_resistance(
            refractory_thickness=self.refractory_thickness,
            refractory_conductivity=self.refractory_conductivity,
            h_ext=self.h_ext,
            A_wall_cell=self.A_wall_cell,
        )

        # ======================================================
        # PICARD ITERATION
        # ======================================================

        max_iter = 100
        tol = 1e-6
        relaxation = 0.5

        Tg_iter = np.asarray(
            Tg,
            dtype=float,
        ).copy()

        Ts_iter = np.asarray(
            Ts,
            dtype=float,
        ).copy()

        Tw_iter = np.asarray(
            Tw,
            dtype=float,
        ).copy()

        for iteration in range(max_iter):

            # --------------------------------------------------
            # Existing heat-transfer model
            # --------------------------------------------------

            q_gs, q_gw, q_ws = heat_transfer(
                Tg=Tg_iter,
                Ts=Ts_iter,
                Tw=Tw_iter,
                hv_gs=self.hv_gs,
                hv_gw=self.hv_gw,
                hv_ws=self.hv_ws,
                a_gs=self.a_gs,
                a_gw=self.a_gw,
                a_ws=self.a_ws,
                zone=self.zone,
            )

            # ==================================================
            # LINEAR SYSTEM
            # ==================================================

            n_unknowns = 3 * N

            A = np.zeros(
                (n_unknowns, n_unknowns),
                dtype=float,
            )

            b = np.zeros(
                n_unknowns,
                dtype=float,
            )

            row = 0

            # ==================================================
            # GAS
            # ==================================================

            # Inlet boundary:
            #
            # Tg[0] = Tg_in
            #
            A[row, 0] = 1.0
            b[row] = Tg_in

            row += 1

            for i in range(1, N):

                Cg = m_dot_g * float(
                    cp_gas(Tg_iter[i])
                )

                gas_i = i
                solid_i = N + i
                wall_i = 2 * N + i

                gas_up = i - 1

                # ----------------------------------------------
                # m_dot_g Cp_g (Tg_i - Tg_up)
                #
                # + Q_gs
                # + Q_gw
                # = 0
                # ----------------------------------------------

                A[row, gas_i] += (
                    Cg
                    + V_cell * K_gs
                    + V_cell * K_gw
                )

                A[row, gas_up] += -Cg

                A[row, solid_i] += (
                    -V_cell * K_gs
                )

                A[row, wall_i] += (
                    -V_cell * K_gw
                )

                # Radiation part is already contained in
                # heat_transfer(). Freeze nonlinear radiation
                # contribution at current Picard iteration.

                q_rad_gs = (
                    q_gs[i]
                    - K_gs * (
                        Tg_iter[i]
                        - Ts_iter[i]
                    )
                )

                q_rad_gw = (
                    q_gw[i]
                    - K_gw * (
                        Tg_iter[i]
                        - Tw_iter[i]
                    )
                )

                b[row] = -V_cell * (
                    q_rad_gs
                    + q_rad_gw
                )

                row += 1

            # ==================================================
            # SOLID
            # ==================================================

            Cs = m_dot_s * self.Cp_s

            # Inlet boundary:
            #
            # Ts[0] = Ts_in
            #
            A[row, N] = 1.0
            b[row] = Ts_in

            row += 1

            for i in range(1, N):

                gas_i = i
                solid_i = N + i
                wall_i = 2 * N + i

                solid_up = N + i - 1

                # ----------------------------------------------
                # m_dot_s Cp_s (Ts_i - Ts_up)
                #
                # - Q_gs
                # + Q_ws
                # = 0
                # ----------------------------------------------

                A[row, gas_i] += (
                    -V_cell * K_gs
                )

                A[row, solid_i] += (
                    Cs
                    + V_cell * K_gs
                    + V_cell * K_ws
                )

                A[row, solid_up] += -Cs

                A[row, wall_i] += (
                    -V_cell * K_ws
                )

                q_rad_gs = (
                    q_gs[i]
                    - K_gs * (
                        Tg_iter[i]
                        - Ts_iter[i]
                    )
                )

                q_rad_ws = (
                    q_ws[i]
                    - K_ws * (
                        Ts_iter[i]
                        - Tw_iter[i]
                    )
                )

                b[row] = V_cell * (
                    q_rad_gs
                    - q_rad_ws
                )

                row += 1

            # ==================================================
            # WALL
            # ==================================================

            for i in range(N):

                gas_i = i
                solid_i = N + i
                wall_i = 2 * N + i

                # ----------------------------------------------
                # Q_gw + Q_ws - Q_loss = 0
                # ----------------------------------------------

                A[row, gas_i] += (
                    V_cell * K_gw
                )

                A[row, solid_i] += (
                    V_cell * K_ws
                )

                A[row, wall_i] += (
                    -V_cell * K_gw
                    -V_cell * K_ws
                    -0.27 / R_total
                )

                q_rad_gw = (
                    q_gw[i]
                    - K_gw * (
                        Tg_iter[i]
                        - Tw_iter[i]
                    )
                )

                q_rad_ws = (
                    q_ws[i]
                    - K_ws * (
                        Ts_iter[i]
                        - Tw_iter[i]
                    )
                )

                b[row] = (
                    -V_cell * (
                        q_rad_gw
                        + q_rad_ws
                    )
                    -0.27 * self.T_amb / R_total
                )

                row += 1

            # ==================================================
            # SOLVE
            # ==================================================

            solution = np.linalg.solve(
                A,
                b,
            )

            Tg_new = solution[:N]

            Ts_new = solution[N:2 * N]

            Tw_new = solution[2 * N:3 * N]

            # ==================================================
            # PICARD RELAXATION
            # ==================================================

            Tg_next = (
                relaxation * Tg_new
                + (1.0 - relaxation) * Tg_iter
            )

            Ts_next = (
                relaxation * Ts_new
                + (1.0 - relaxation) * Ts_iter
            )

            Tw_next = (
                relaxation * Tw_new
                + (1.0 - relaxation) * Tw_iter
            )

            error = max(
                np.max(
                    np.abs(
                        Tg_next - Tg_iter
                    )
                ),
                np.max(
                    np.abs(
                        Ts_next - Ts_iter
                    )
                ),
                np.max(
                    np.abs(
                        Tw_next - Tw_iter
                    )
                ),
            )

            Tg_iter = Tg_next
            Ts_iter = Ts_next
            Tw_iter = Tw_next

            if error < tol:
                break

        else:
            raise RuntimeError(
                "Cooler steady-state thermal solution "
                f"did not converge after {max_iter} iterations. "
                f"error={error:.6e} K"
            )

        # ======================================================
        # FINAL STATE
        # ======================================================

        Tg_ss = Tg_iter
        Ts_ss = Ts_iter
        Tw_ss = Tw_iter

        # ======================================================
        # FINAL HEAT TRANSFER
        # ======================================================

        q_gs, q_gw, q_ws = heat_transfer(
            Tg=Tg_ss,
            Ts=Ts_ss,
            Tw=Tw_ss,
            hv_gs=self.hv_gs,
            hv_gw=self.hv_gw,
            hv_ws=self.hv_ws,
            a_gs=self.a_gs,
            a_gw=self.a_gw,
            a_ws=self.a_ws,
            zone=self.zone,
        )

        # q_* are volumetric [W/m3]
        Qgs = float(
            np.sum(q_gs * V_cell)
        )

        Qgw = float(
            np.sum(q_gw * V_cell)
        )

        Qws = float(
            np.sum(q_ws * V_cell)
        )

        # ======================================================
        # FINAL WALL LOSS
        # ======================================================

        (
            _,
            wall_loss,
            wall_debug,
        ) = wall_losses(
            Tw=Tw_ss,
            h_ext=self.h_ext,
            A_wall_cell=self.A_wall_cell,
            V_cell=self.V_cell,
            T_amb=self.T_amb,
            A_wall_total=self.A_wall,
            N=N,
            refractory_thickness=self.refractory_thickness,
            refractory_conductivity=self.refractory_conductivity,
            eps=self.eps,
        )

        # ======================================================
        # ENTHALPY
        # ======================================================

        Hg_in = (
            m_dot_g
            * float(
                h_gas(
                    Tg_in,
                    self.T_ref,
                )
            )
        )

        Hg_out = (
            m_dot_g
            * float(
                h_gas(
                    Tg_ss[-1],
                    self.T_ref,
                )
            )
        )

        Hs_in = (
            m_dot_s
            * self.Cp_s
            * (Ts_in - self.T_ref)
        )

        Hs_out = (
            m_dot_s
            * self.Cp_s
            * (Ts_ss[-1] - self.T_ref)
        )

        # ======================================================
        # ENERGY BALANCE
        # ======================================================

        energy_in = (
            Hg_in
            + Hs_in
        )

        energy_out = (
            Hg_out
            + Hs_out
            + wall_loss
        )

        total_energy_balance = (
            energy_in
            - energy_out
        )

        # ======================================================
        # STORE DIAGNOSTICS
        # ======================================================

        self.energy_in = float(
            energy_in
        )

        self.energy_out = float(
            energy_out
        )

        self.energy_residual = float(
            total_energy_balance
        )

        # ======================================================
        # RETURN
        # ======================================================

        return (
            Tg_ss,
            Ts_ss,
            Tw_ss,
            wall_loss,
            wall_debug,
            Qgs,
            Qgw,
            Qws,
            energy_in,
            energy_out,
            total_energy_balance,
        )



    
    # ======================================================
    # STATE UPDATE
    # ======================================================

    def apply(self, state):

        if not isinstance(state.Tg_cooler, np.ndarray):
            raise TypeError("Tg_cooler must be np.ndarray")

        if state.Tg_cooler.shape != (self.N,):
            raise ValueError("Cooler state corrupted")

        # ======================================================
        # INLET / OUTLET HANDOFF
        # ======================================================

        state.Hgas_cooler_in = state.Hgas_preheater_out
        state.Hsolid_cooler_in = state.Hsolid_burning_out

        Tg_in = state.Tg_preheater[-1]
        Ts_in = state.Ts_burning[-1]

        state.Tg_cooler_in = Tg_in
        state.Ts_cooler_in = Ts_in

        state.Tg_cooler[0] = Tg_in
        state.Ts_cooler[0] = Ts_in

        # ======================================================
        # STEADY-STATE THERMAL SOLVE
        # ======================================================

        (
            Tg,
            Ts,
            Tw,
            wall_loss,
            wall_debug,
            Qgs,
            Qgw,
            Qws,
            energy_in,
            energy_out,
            total_energy_balance,
        ) = self.thermal_step(
            state.Tg_cooler,
            state.Ts_cooler,
            state.Tw_cooler,
            state,
        )

        # ======================================================
        # UPDATE STATE
        # ======================================================

        state.Tg_cooler = Tg
        state.Ts_cooler = Ts
        state.Tw_cooler = Tw

        state.Wall_loss_cooler = float(wall_loss)

        # ======================================================
        # ENTHALPY
        # ======================================================

        state.Hgas_cooler_out = self.gas_enthalpy_out(
            state.Tg_cooler,
            state,
        )

        state.Hsolid_cooler_out = self.solid_enthalpy_out(
            state.Ts_cooler,
            state,
        )

        # ======================================================
        # HEAT-TRANSFER DIAGNOSTICS
        # ======================================================

        state.Q_gs_cooler = Qgs
        state.Q_gw_cooler = Qgw
        state.Q_ws_cooler = Qws

        # ======================================================
        # ENERGY BALANCE
        # ======================================================

        state.Cooler_energy_in = energy_in
        state.Cooler_energy_out = energy_out
        state.Cooler_energy_balance = total_energy_balance

        return state


    # ======================================================
    # GAS ENTHALPY TO NEXT ZONE
    # ======================================================
    def gas_enthalpy_out(self, Tg, state):

        H_gas_out = (
            state.m_dot_g
            * float(
                h_gas(
                    Tg[-1],
                    self.T_ref,
                )
            )
        )

        return H_gas_out


    # ======================================================
    # SOLID ENTHALPY TO NEXT ZONE
    # ======================================================
    def solid_enthalpy_out(self, Ts, state):

        H_solid_out = (
            state.m_dot_s
            * self.Cp_s
            * (Ts[-1] - self.T_ref)
        )

        return H_solid_out