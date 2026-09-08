import numpy as np

from physics.physics import residence_time
from physics.physics import cp_gas
from physics.physics import h_gas
from physics.physics import radiation
from physics.physics import interfacial_areas
from physics.physics import kiln_geometry
from physics.physics import wall_thermal_resistance
from physics.physics import solid_axial_velocity

from chemistry.reactions import ChemistryModel
from physics.physics import ZONE_HT_CONFIG


class Calciner:

    def __init__(self, N=5, L=25.0):

        self.N = N
        self.L = L
        self.dz = L / N

        # ================= ZONE =================
        self.zone = "calciner"   
        
        self.chemistry = ChemistryModel()

        # ================= NUMERICAL =================
        self.eps = 1e-9

        # ================= GEOMETRY =================
        self.D = 4.2

        self.A_cross, self.V_total, self.V_cell = kiln_geometry(
            D=self.D,
            L=self.L,
            N=self.N,
        )

        # ================= INTERFACIAL AREA =================
        self.epsilon_bed = 0.35
        self.k_interfacial = 1.0

        self.a_gs, self.a_ws = interfacial_areas(
            D=self.D,
            epsilon_bed=self.epsilon_bed,
            k_interfacial=self.k_interfacial,
        )

        # ================= WALL GEOMETRY =================
        self.wall_perimeter = np.pi * self.D
        self.A_wall = self.wall_perimeter * self.L
        self.A_wall_cell = self.A_wall / self.N
        self.a_gw = self.A_wall_cell / self.V_cell

        # ================= REFRACTORY =================
        self.refractory_thickness = 0.05      # m
        self.refractory_conductivity = 1.8    # W/mK

        self.V_wall = self.A_wall * self.refractory_thickness
        self.V_wall_cell = self.V_wall / self.N

        # ================= EXTERNAL WALL =================
        self.h_ext = 12.0
        self.T_ref = 298.15   # K
        self.T_amb = 300.0    # K

        # ================= PROPERTIES =================
        self.rho_g = 0.30
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0
        
        # ======================================================
        # CALCINER OPERATING PARAMETERS
        # ======================================================

        self.slope_deg = 3.0
        self.fill_fraction = 0.10
        self.rpm = 3.0

        # ================= FLOW =================
        self.u_g = 0.0

        self.u_s = solid_axial_velocity(
            L=self.L,
            D=self.D,
            slope_deg=self.slope_deg,
            fill_fraction=self.fill_fraction,
            rpm=self.rpm,
            eps=self.eps,
        )

        # ================= HEAT TRANSFER =================
        cfg = ZONE_HT_CONFIG[self.zone]

        self.hv_gs = cfg["hv_gs"]
        self.hv_gw = cfg["hv_gw"]
        self.hv_ws = cfg["hv_ws"]

        # ================= BUFFERS =================
        self._dTg_dz = np.zeros(N)
        self._dTs_dz = np.zeros(N)

        # ================= CACHE =================
        self._rho_g_Vcell_Cp_g = self.rho_g * self.V_cell * self.Cp_g
        self._rho_s_Vcell_Cp_s = self.rho_s * self.V_cell * self.Cp_s
        self._rho_wall_Vwall_cell_Cp = self.rho_wall * self.V_wall_cell * self.Cp_wall

    # ======================================================
    def thermal_step(
        self,
        Tg,
        Ts,
        Tw,
        state,
        Tg_in,
        Ts_in,
        reaction_sink=0.0,
        reaction_heat_cells=None,
    ):

        # ======================================================
        # INPUTS
        # ======================================================

        T_ref = self.T_ref

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
        # GEOMETRY
        # ======================================================

        N = len(Tg)

        V_cell = self.V_cell

        # ======================================================
        # HEAT TRANSFER PARAMETERS
        # ======================================================

        hv_gs = self.hv_gs
        hv_gw = self.hv_gw
        hv_ws = self.hv_ws

        a_gs = self.a_gs
        a_gw = self.a_gw
        a_ws = self.a_ws

        K_gs = (
            hv_gs
            * a_gs
        )

        K_gw = (
            hv_gw
            * a_gw
        )

        K_ws = (
            hv_ws
            * a_ws
        )

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
        # REACTION DISTRIBUTION
        #
        # reaction_heat_cells[i] : W
        #
        # Each cell receives the actual local
        # calcination heat sink.
        # ======================================================

        if reaction_heat_cells is None:

            reaction_heat_cells = np.zeros(N)

        else:

            reaction_heat_cells = np.asarray(
                reaction_heat_cells,
                dtype=float,
            )

            if reaction_heat_cells.shape != (N,):

                raise ValueError(
                    "reaction_heat_cells must have "
                    f"shape ({N},)."
                )

            if not np.all(
                np.isfinite(
                    reaction_heat_cells
                )
            ):

                raise ValueError(
                    "reaction_heat_cells contains "
                    "non-finite values."
                )

            if np.any(
                reaction_heat_cells < 0.0
            ):

                raise ValueError(
                    "reaction_heat_cells cannot "
                    "contain negative values."
                )

        # ======================================================
        # PICARD ITERATION
        # ======================================================

        max_iter = 100
        tol = 1e-6
        relaxation = 0.5

        Tg_iter = (
            np.asarray(
                Tg,
                dtype=float,
            )
            .copy()
        )

        Ts_iter = (
            np.asarray(
                Ts,
                dtype=float,
            )
            .copy()
        )

        Tw_iter = (
            np.asarray(
                Tw,
                dtype=float,
            )
            .copy()
        )

        converged = False
        error = np.inf

        # ======================================================
        # PICARD LOOP
        # ======================================================

        for iteration in range(max_iter):

            # ==================================================
            # GAS PROPERTIES
            # ==================================================

            Cp_g_iter = cp_gas(
                Tg_iter
            )

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

            n_unknowns = (
                3 * N
            )

            A = np.zeros(
                (
                    n_unknowns,
                    n_unknowns,
                )
            )

            b = np.zeros(
                n_unknowns
            )

            row = 0

            # ==================================================
            # CELL LOOP
            # ==================================================

            for i in range(N):

                Tg_i = i

                Ts_i = (
                    N + i
                )

                Tw_i = (
                    2 * N + i
                )

                # ==================================================
                # GAS LOCAL HEAT CAPACITY
                # ==================================================

                Cp_g_i = cp_gas(
                    Tg_iter[i]
                )

                Cg_i = (
                    m_dot_g
                    * Cp_g_i
                )

                # ==================================================
                # GAS ENTHALPY LINEARIZATION
                #
                # h(T) ≈ Cp*T + constant
                # ==================================================

                h_i_iter = h_gas(
                    Tg_iter[i],
                    T_ref,
                )

                h_linear_const_i = (
                    h_i_iter
                    - Cp_g_i
                    * Tg_iter[i]
                )

                # ==================================================
                # GAS ENERGY BALANCE
                #
                # Gas direction:
                #
                # N-1  --->  0
                #
                # Gas receives no reaction sink directly.
                # Calcination reaction is a solid-phase sink.
                # ==================================================

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
                # GAS RADIATION SINK
                # --------------------------------------------------

                radiation_gas_sink = (
                    V_cell
                    * (
                        q_gs_rad[i]
                        + q_gw_rad[i]
                    )
                )

                # --------------------------------------------------
                # GAS INLET CELL
                #
                # Gas enters at cell N-1
                # --------------------------------------------------

                if i == N - 1:

                    h_in = h_gas(
                        Tg_in,
                        T_ref,
                    )

                    b[row] = (
                        m_dot_g * h_in
                        - m_dot_g
                        * h_linear_const_i
                        - radiation_gas_sink
                    )

                # --------------------------------------------------
                # INTERNAL GAS CELLS
                # --------------------------------------------------

                else:

                    Tg_up_i = (
                        i + 1
                    )

                    Cp_g_up = cp_gas(
                        Tg_iter[i + 1]
                    )

                    h_up_iter = h_gas(
                        Tg_iter[i + 1],
                        T_ref,
                    )

                    h_linear_const_up = (
                        h_up_iter
                        - Cp_g_up
                        * Tg_iter[i + 1]
                    )

                    A[row, Tg_up_i] += (
                        -m_dot_g
                        * Cp_g_up
                    )

                    b[row] = (
                        -m_dot_g
                        * h_linear_const_i
                        + m_dot_g
                        * h_linear_const_up
                        - radiation_gas_sink
                    )

                row += 1

                # ==================================================
                # SOLID ENERGY BALANCE
                #
                # Solid direction:
                #
                # 0  --->  N-1
                #
                # Calcination reaction is a solid-phase
                # energy sink.
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
                # SOLID RADIATION
                # --------------------------------------------------

                radiation_solid_source = (
                    V_cell
                    * (
                        q_gs_rad[i]
                        - q_ws_rad[i]
                    )
                )

                # --------------------------------------------------
                # SOLID INLET CELL
                # --------------------------------------------------

                if i == 0:

                    b[row] = (
                        Cs * Ts_in
                        + radiation_solid_source
                        - reaction_heat_cells[i]
                    )

                # --------------------------------------------------
                # INTERNAL SOLID CELLS
                # --------------------------------------------------

                else:

                    Ts_up_i = (
                        N + i - 1
                    )

                    A[row, Ts_up_i] += (
                        -Cs
                    )

                    b[row] = (
                        radiation_solid_source
                        - reaction_heat_cells[i]
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
                    -self.T_amb
                    / R_total
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

            Tg_new = (
                x_solution[:N]
            )

            Ts_new = (
                x_solution[
                    N:2 * N
                ]
            )

            Tw_new = (
                x_solution[
                    2 * N:3 * N
                ]
            )

            # ======================================================
            # CONVERGENCE ERROR
            # ======================================================

            error = max(
                np.max(
                    np.abs(
                        Tg_new
                        - Tg_iter
                    )
                ),
                np.max(
                    np.abs(
                        Ts_new
                        - Ts_iter
                    )
                ),
                np.max(
                    np.abs(
                        Tw_new
                        - Tw_iter
                    )
                ),
            )

            # ======================================================
            # RELAXATION
            # ======================================================

            Tg_iter = (
                relaxation
                * Tg_new
                + (
                    1.0
                    - relaxation
                )
                * Tg_iter
            )

            Ts_iter = (
                relaxation
                * Ts_new
                + (
                    1.0
                    - relaxation
                )
                * Ts_iter
            )

            Tw_iter = (
                relaxation
                * Tw_new
                + (
                    1.0
                    - relaxation
                )
                * Tw_iter
            )

            # ======================================================
            # CONVERGENCE
            # ======================================================

            if error < tol:

                converged = True

                break

        # ======================================================
        # CONVERGENCE WARNING
        # ======================================================

        if not converged:

            print(
                f"[CALCINER WARNING] "
                f"Thermal iteration did not converge. "
                f"error = {error:.6e} K"
            )

        # ======================================================
        # STEADY-STATE TEMPERATURES
        # ======================================================

        Tg_ss = Tg_iter
        Ts_ss = Ts_iter
        Tw_ss = Tw_iter

        # ======================================================
        # INLET / OUTLET
        #
        # Gas:
        #     N-1 -> 0
        #
        # Solid:
        #     0 -> N-1
        # ======================================================

        Tg_out = Tg_ss[0]

        Ts_out = Ts_ss[-1]

        Tw_out = Tw_ss[-1]

        # ======================================================
        # FINAL RADIATION
        # ======================================================

        q_gs_rad_final = radiation(
            Tg_ss,
            Ts_ss,
            zone=self.zone,
            area=a_gs,
        )

        q_gw_rad_final = radiation(
            Tg_ss,
            Tw_ss,
            zone=self.zone,
            area=a_gw,
        )

        q_ws_rad_final = radiation(
            Ts_ss,
            Tw_ss,
            zone=self.zone,
            area=a_ws,
        )

        # ======================================================
        # CONVECTIVE HEAT TRANSFER
        # ======================================================

        Qgs_conv = (
            V_cell
            * K_gs
            * (
                Tg_ss
                - Ts_ss
            )
        )

        Qgw_conv = (
            V_cell
            * K_gw
            * (
                Tg_ss
                - Tw_ss
            )
        )

        Qws_conv = (
            V_cell
            * K_ws
            * (
                Ts_ss
                - Tw_ss
            )
        )

        # ======================================================
        # TOTAL HEAT TRANSFER
        # ======================================================

        Qgs = np.sum(
            Qgs_conv
            + V_cell
            * q_gs_rad_final
        )

        Qgw = np.sum(
            Qgw_conv
            + V_cell
            * q_gw_rad_final
        )

        Qws = np.sum(
            Qws_conv
            + V_cell
            * q_ws_rad_final
        )

        # ======================================================
        # WALL LOSS
        # ======================================================

        Q_wall_loss = np.sum(
            (
                Tw_ss
                - self.T_amb
            )
            / R_total
        )

        # ======================================================
        # GAS ENTHALPY BALANCE
        # ======================================================

        Hg_in = (
            m_dot_g
            * h_gas(
                Tg_in,
                T_ref,
            )
        )

        Hg_out = (
            m_dot_g
            * h_gas(
                Tg_out,
                T_ref,
            )
        )

        gas_energy_change = (
            Hg_out
            - Hg_in
        )

        # ------------------------------------------------------
        # Gas only exchanges energy through:
        #
        #   gas -> solid
        #   gas -> wall
        #
        # Reaction is NOT a gas sink.
        # ------------------------------------------------------

        gas_expected = (
            -Qgs
            -Qgw
        )

        gas_energy_balance = (
            gas_energy_change
            - gas_expected
        )

        # ======================================================
        # SOLID ENTHALPY BALANCE
        # ======================================================

        Hs_in = (
            m_dot_s
            * Cp_s
            * (
                Ts_in
                - T_ref
            )
        )

        Hs_out = (
            m_dot_s
            * Cp_s
            * (
                Ts_out
                - T_ref
            )
        )

        solid_energy_change = (
            Hs_out
            - Hs_in
        )

        # ------------------------------------------------------
        # Solid receives:
        #
        #   +Qgs
        #   -Qws
        #   -Qcalcination
        # ------------------------------------------------------

        solid_expected = (
            Qgs
            - Qws
            - reaction_sink
        )

        solid_energy_balance = (
            solid_energy_change
            - solid_expected
        )

        # ======================================================
        # TOTAL EQUIPMENT ENERGY BALANCE
        #
        # Hgas_in
        # + Hsolid_in
        #
        # =
        #
        # Hgas_out
        # + Hsolid_out
        # + Q_wall_loss
        # + Q_calcination
        #
        # Therefore:
        #
        # residual -> 0
        # ======================================================

        total_energy_balance = (
            Hg_in
            + Hs_in
            - Hg_out
            - Hs_out
            - Q_wall_loss
            - reaction_sink
        )
        
        print("\n========== CALCINER PHASE AFTER TOTAL ENERGY BALANCE ==========")

        print(
            f"Gas ΔH                 = "
            f"{gas_energy_change:.6e} W"
        )

        print(
            f"Gas expected           = "
            f"{gas_expected:.6e} W"
        )

        print(
            f"Gas balance            = "
            f"{gas_energy_balance:.6e} W"
        )

        print(
            f"Solid ΔH               = "
            f"{solid_energy_change:.6e} W"
        )

        print(
            f"Solid expected         = "
            f"{solid_expected:.6e} W"
        )

        print(
            f"Solid balance          = "
            f"{solid_energy_balance:.6e} W"
        )

        print(
            f"Qgs                    = "
            f"{Qgs:.6e} W"
        )

        print(
            f"Qgw                    = "
            f"{Qgw:.6e} W"
        )

        print(
            f"Qws                    = "
            f"{Qws:.6e} W"
        )

        print(
            f"Qwall                  = "
            f"{Q_wall_loss:.6e} W"
        )

        print(
            f"Qreaction              = "
            f"{reaction_sink:.6e} W"
        )

        print(
            f"Qgw + Qws              = "
            f"{Qgw + Qws:.6e} W"
        )

        print(
            f"Qwall consistency      = "
            f"{Q_wall_loss - (Qgw + Qws):.6e} W"
        )

        print(
            f"Total balance          = "
            f"{total_energy_balance:.6e} W"
        )

        print(
            "===================================================="
        )

        # ======================================================
        # RETURN
        # ======================================================

        return (
            Tg_ss,
            Ts_ss,
            Tw_ss,
            Q_wall_loss,
            {
                "Q_gs": Qgs,
                "Q_gw": Qgw,
                "Q_ws": Qws,
                "Q_wall_loss": Q_wall_loss,
                "Q_reaction": reaction_sink,
                "gas_energy_balance": gas_energy_balance,
                "solid_energy_balance": solid_energy_balance,
                "total_energy_balance": total_energy_balance,
                "iterations": iteration + 1,
                "converged": converged,
            },
        )

    # ======================================================
    # STATE UPDATE
    # ======================================================
    def apply(self, state):

        print("\n========== CALCINER APPLY ENTER ==========", flush=True)
        print(f"m_dot_g                = {state.m_dot_g:.6e} kg/s", flush=True)
        print(f"Hgas_transition_out    = {state.Hgas_transition_out:.6e} W", flush=True)
        print("==========================================", flush=True)

        # ======================================================
        # STATE INTEGRITY CHECK
        # ======================================================

        if not isinstance(
            state.Tg_calciner,
            np.ndarray,
        ):
            raise TypeError(
                "Tg_calciner must be np.ndarray"
            )

        if state.Tg_calciner.shape != (
            self.N,
        ):
            raise ValueError(
                f"Calciner state corrupted: "
                f"{state.Tg_calciner.shape}"
            )

        # ======================================================
        # INCOMING ENTHALPY FROM TRANSITION
        # ======================================================

        state.Hgas_calciner_in = (
            state.Hgas_transition_out
        )

        state.Hsolid_calciner_in = (
            state.Hsolid_transition_out
        )

        # ======================================================
        # INLET TEMPERATURES FROM ENTHALPY
        #
        # These are physical inlet conditions.
        #
        # Gas:
        #     N-1 -> 0
        #
        # Solid:
        #     0 -> N-1
        # ======================================================

        Tg_in = (
            self.gas_temperature_from_enthalpy(
                state.Hgas_calciner_in,
                state,
            )
        )

        Ts_in = (
            self.solid_temperature_from_enthalpy(
                state.Hsolid_calciner_in,
                state,
            )
        )
        
        # ======================================================
        # CALCINER SOLID AXIAL VELOCITY
        #
        # Steady-state spatial reaction:
        #
        #     u_s * dm/dz = -r
        #
        # No residence-time chemistry is used here.
        # ======================================================

        rpm = float(
            getattr(
                state,
                "rpm",
                self.rpm,
            )
        )

        self.u_s = solid_axial_velocity(
            L=self.L,
            D=self.D,
            slope_deg=self.slope_deg,
            fill_fraction=self.fill_fraction,
            rpm=rpm,
            eps=self.eps,
        )

        self.u_s = max(
            float(self.u_s),
            self.eps,
        )


        # ======================================================
        # CHEMISTRY <-> THERMAL STEADY-STATE ITERATION
        #
        # Spatial chemistry:
        #
        #     Ts(z)
        #       ↓
        #     k(T)
        #       ↓
        #     X(z)
        #       ↓
        #     Q_reaction(z)
        #       ↓
        #     thermal_step()
        #       ↓
        #     new Ts(z)
        #
        # No dt is used.
        # ======================================================

        max_coupling_iter = 50
        coupling_tol = 1.0e-5
        coupling_relaxation = 0.5

        # ------------------------------------------------------
        # INITIAL GUESS
        # ------------------------------------------------------

        Tg_iter = (
            np.asarray(
                state.Tg_calciner,
                dtype=float,
            )
            .copy()
        )

        Ts_iter = (
            np.asarray(
                state.Ts_calciner,
                dtype=float,
            )
            .copy()
        )

        Tw_iter = (
            np.asarray(
                state.Tw_calciner,
                dtype=float,
            )
            .copy()
        )

        previous_Q_reaction = 0.0

        coupling_converged = False

        # ======================================================
        # OUTER STEADY-STATE ITERATION
        # ======================================================

        for coupling_iteration in range(
            max_coupling_iter
        ):

            # ==================================================
            # 1. UPDATE REACTION TEMPERATURE FIELD
            # ==================================================

            state.Ts_calciner = (
                Ts_iter.copy()
            )

            # ==================================================
            # 2. SPATIAL CHEMISTRY
            # ==================================================

            state = self.chemistry.apply_calciner(
                state,
                self.dz,
                self.u_s,
            )

            print("\n========== ILC CHEMISTRY DEBUG ==========")
            #print(f"CaCO3 in      = {state.m_dot_CaCO3_in_calciner:.6f} kg/s")
            #print(f"CaCO3 reacted = {state.m_dot_CaCO3_reacted_calciner:.6f} kg/s")
            #print(f"CaCO3 out     = {state.m_dot_CaCO3_out_calciner:.6f} kg/s")
            #print(f"Conversion    = {state.X_CaCO3_calciner:.6f}")

            Q_reaction = float(
                state.Calcination_Q_sink
            )
            
            print(
                f"[CALCINER -> TRANSITION] "
                f"CaCO3_out = "
                f"{state.m_dot_CaCO3_out_calciner:.6f} kg/s"
            )

            # ==================================================
            # 3. THERMAL SOLUTION
            # ==================================================

            (
                Tg_new,
                Ts_new,
                Tw_new,
                wall_loss_new,
                wall_debug_new,
            ) = self.thermal_step(
                Tg_iter,
                Ts_iter,
                Tw_iter,
                state,
                Tg_in,
                Ts_in,
                reaction_sink=Q_reaction,
                reaction_heat_cells=(
                    state.Calcination_Q_cells
                ),
            )

            # ==================================================
            # 4. RELAXATION
            # ==================================================

            Tg_relaxed = (
                coupling_relaxation
                * Tg_new
                +
                (1.0 - coupling_relaxation)
                * Tg_iter
            )

            Ts_relaxed = (
                coupling_relaxation
                * Ts_new
                +
                (1.0 - coupling_relaxation)
                * Ts_iter
            )

            Tw_relaxed = (
                coupling_relaxation
                * Tw_new
                +
                (1.0 - coupling_relaxation)
                * Tw_iter
            )

            # ==================================================
            # 5. CONVERGENCE ERRORS
            # ==================================================

            temperature_error = max(
                np.max(
                    np.abs(
                        Tg_relaxed
                        - Tg_iter
                    )
                ),
                np.max(
                    np.abs(
                        Ts_relaxed
                        - Ts_iter
                    )
                ),
                np.max(
                    np.abs(
                        Tw_relaxed
                        - Tw_iter
                    )
                ),
            )

            reaction_error = abs(
                Q_reaction
                - previous_Q_reaction
            )

            reaction_scale = max(
                abs(Q_reaction),
                1.0,
            )

            reaction_relative_error = (
                reaction_error
                / reaction_scale
            )

            # ==================================================
            # 6. UPDATE ITERATION STATES
            # ==================================================

            Tg_iter = Tg_relaxed
            Ts_iter = Ts_relaxed
            Tw_iter = Tw_relaxed

            previous_Q_reaction = (
                Q_reaction
            )

            # ==================================================
            # 7. STEADY-STATE CONVERGENCE
            # ==================================================

            if (
                temperature_error
                < coupling_tol
                and
                reaction_relative_error
                < coupling_tol
            ):

                coupling_converged = True

                break

        # ======================================================
        # FINAL STATE
        # ======================================================

        state.Tg_calciner = Tg_iter
        state.Ts_calciner = Ts_iter
        state.Tw_calciner = Tw_iter

        state.Wall_loss_calciner = float(
            wall_loss_new
        )

        state.Calciner_coupling_iterations = (
            coupling_iteration + 1
        )

        state.Calciner_coupling_converged = (
            coupling_converged
        )

        state.Calciner_coupling_temperature_error = (
            float(temperature_error)
        )

        state.Calciner_coupling_reaction_error = (
            float(reaction_relative_error)
        )

        # ======================================================
        # FINAL CALCINER REACTION REPORT
        # ======================================================

        print(
            "\n========== CALCINER REACTION =========="
        )

        print(
            f"Coupling iterations    = "
            f"{state.Calciner_coupling_iterations}"
        )

        print(
            f"Coupling converged     = "
            f"{state.Calciner_coupling_converged}"
        )

        print(
            f"Temperature error      = "
            f"{state.Calciner_coupling_temperature_error:.6e} K"
        )

        print(
            f"Reaction relative err  = "
            f"{state.Calciner_coupling_reaction_error:.6e}"
        )


        #print(
        #    f"CaCO3 reacted flow     = "
        #    f"{state.m_dot_CaCO3_reacted_calciner:.6f} kg/s"
        #)

   

        print(
            f"Calcination conversion = "
            f"{state.X_calcination:.6f}"
        )

        print(
            f"Calcination heat       = "
            f"{state.Calcination_Q_sink:.6e} W"
        )

        print("----------------------------------------")

        print(
            "Cell conversion        =",
            state.X_CaCO3_cells
        )

        print(
            "Cell reacted flow      =",
            state.m_dot_CaCO3_reacted_cells
        )

        print(
            "Cell inlet flow        =",
            state.m_dot_CaCO3_in_cells
        )

        print(
            "Cell outlet flow       =",
            state.m_dot_CaCO3_out_cells
        )

        print("========================================")
        state.Tw_calciner = Tw_iter

        state.Wall_loss_calciner = (
            float(wall_loss_new)
        )
        
        print("\n========== CALCINER ENTHALPY INPUT CHECK ==========")

        print(
            f"state.m_dot_g        = "
            f"{state.m_dot_g:.6e} kg/s"
        )

        print(
            f"Tg_calciner          = "
            f"{state.Tg_calciner}"
        )

        print(
            f"Tg_calciner[0]       = "
            f"{state.Tg_calciner[0]:.6f} K"
        )

        print(
            f"Tg_calciner[-1]      = "
            f"{state.Tg_calciner[-1]:.6f} K"
        )

        print(
            f"h_gas(Tg[0])         = "
            f"{h_gas(state.Tg_calciner[0], self.T_ref):.6e} J/kg"
        )

        print(
            f"h_gas(Tg[-1])        = "
            f"{h_gas(state.Tg_calciner[-1], self.T_ref):.6e} J/kg"
        )

        print(
            f"Hgas_calciner_in     = "
            f"{state.Hgas_calciner_in:.6e} W"
        )

        print(
            f"Hsolid_calciner_in   = "
            f"{state.Hsolid_calciner_in:.6e} W"
        )

        print("====================================================")
        
        print("\n========== CALCINER ENTHALPY INPUT CHECK ==========")

        print(f"state.m_dot_g        = {state.m_dot_g:.6e} kg/s")

        print(f"Tg_calciner          = {state.Tg_calciner}")
        print(f"Tg_calciner[0]       = {state.Tg_calciner[0]:.6f} K")
        print(f"Tg_calciner[-1]      = {state.Tg_calciner[-1]:.6f} K")

        print(
            f"h_gas(Tg[0])         = "
            f"{h_gas(state.Tg_calciner[0], self.T_ref):.6e} J/kg"
        )

        print(
            f"h_gas(Tg[-1])        = "
            f"{h_gas(state.Tg_calciner[-1], self.T_ref):.6e} J/kg"
        )

        print(f"Hgas_calciner_in     = {state.Hgas_calciner_in:.6e} W")
        print(f"Hsolid_calciner_in   = {state.Hsolid_calciner_in:.6e} W")

        print("====================================================")

        # ======================================================
        # UPDATE GAS ENTHALPY
        #
        # Must use the same h_gas() definition
        # used by thermal_step().
        # ======================================================
        print("\n========== BEFORE CALCINER HG ==========", flush=True)
        print(f"m_dot_g       = {state.m_dot_g:.6e} kg/s", flush=True)
        print(f"Tg[0]         = {state.Tg_calciner[0]:.6f} K", flush=True)
        print(f"Tg[-1]        = {state.Tg_calciner[-1]:.6f} K", flush=True)

        h0 = h_gas(state.Tg_calciner[0], self.T_ref)
        hN = h_gas(state.Tg_calciner[-1], self.T_ref)

        print(f"h_gas[0]      = {h0:.6e} J/kg", flush=True)
        print(f"h_gas[-1]     = {hN:.6e} J/kg", flush=True)
        print("==========================================", flush=True)

        state.Hg_calciner = (
            state.m_dot_g
            * h_gas(
                state.Tg_calciner,
                self.T_ref,
            )
        )


        # ======================================================
        # UPDATE SOLID ENTHALPY
        # ======================================================

        state.Hs_calciner = (
            state.m_dot_s
            * self.Cp_s
            * (
                state.Ts_calciner
                - self.T_ref
            )
        )


        # ======================================================
        # ENTHALPY TO NEXT ZONE
        # ======================================================

        state.Hgas_calciner_out = state.Hg_calciner[0]


        # ======================================================
        # CALCINER OUTPUT CHECK
        # ======================================================

        print("\n========== CALCINER OUTPUT CHECK ==========")
        print(f"Hg_calciner[0]        = {state.Hg_calciner[0]:.6e} W")
        print(f"Hg_calciner[-1]       = {state.Hg_calciner[-1]:.6e} W")
        print(f"Hgas_calciner_out     = {state.Hgas_calciner_out:.6e} W")
        print(f"m_dot_g               = {state.m_dot_g:.6e} kg/s")
        print("===========================================\n")



        # ======================================================
        # SOLID ENTHALPY TO NEXT ZONE
        # ======================================================

        state.Hsolid_calciner_out = (
            state.Hs_calciner[-1]
        )

        # ======================================================
        # STEADY-STATE:
        # NO ACCUMULATION
        # ======================================================

        state.Calciner_gas_stored = 0.0

        state.Calciner_solid_stored = 0.0

        state.Calciner_wall_stored = 0.0

        state.Calciner_stored_energy_change = (
            0.0
        )

        # ======================================================
        # STEADY-STATE ENERGY BALANCE
        #
        # Energy in:
        #
        #   Hgas_in
        # + Hsolid_in
        #
        # Energy out:
        #
        #   Hgas_out
        # + Hsolid_out
        # + Q_wall_loss
        # + Q_calcination
        #
        # Residual:
        #
        #   Energy_in - Energy_out
        #
        # Target:
        #
        #   residual -> 0
        # ======================================================

        state.Calciner_energy_balance = (
            state.Hgas_calciner_in
            + state.Hsolid_calciner_in
            - state.Hgas_calciner_out
            - state.Hsolid_calciner_out
            - state.Wall_loss_calciner
            - state.Calcination_Q_sink
        )

        # ======================================================
        # RELATIVE ENERGY BALANCE
        # ======================================================

        energy_scale = (
            abs(
                state.Hgas_calciner_in
            )
            + abs(
                state.Hsolid_calciner_in
            )
            + abs(
                state.Calcination_Q_sink
            )
            + self.eps
        )

        state.Calciner_energy_balance_relative = (
            state.Calciner_energy_balance
            / energy_scale
        )

        # ======================================================
        # CALCINER FLOW DEBUG
        # ======================================================

        print(
            "\n========== CALCINER FLOW DEBUG =========="
        )

        print(
            f"u_s = {state.u_s:.6e} m/s"
        )

        print("------------------------------------------")

        print(
            f"m_dot_g = "
            f"{state.m_dot_g:.6e} kg/s"
        )

        print(
            f"m_dot_s = "
            f"{state.m_dot_s:.6e} kg/s"
        )

        print("------------------------------------------")

        print(
            f"Tg_in = "
            f"{Tg_in:.3f} K"
        )

        print(
            f"Tg_out = "
            f"{state.Tg_calciner[0]:.3f} K"
        )

        print(
            f"Ts_in = "
            f"{Ts_in:.3f} K"
        )

        print(
            f"Ts_out = "
            f"{state.Ts_calciner[-1]:.3f} K"
        )

        print(
            f"Tw_in = "
            f"{state.Tw_calciner[0]:.3f} K"
        )

        print(
            f"Tw_out = "
            f"{state.Tw_calciner[-1]:.3f} K"
        )

        print(
            "=========================================="
        )

        # ======================================================
        # CALCINER ENERGY BALANCE DEBUG
        # ======================================================

        print(
            "\n========== "
            "CALCINER ENERGY BALANCE "
            "=========="
        )

        print(
            f"Hgas_calciner_in       = "
            f"{state.Hgas_calciner_in:.6e} W"
        )

        print(
            f"Hgas_calciner_out      = "
            f"{state.Hgas_calciner_out:.6e} W"
        )

        print(
            f"Hsolid_calciner_in     = "
            f"{state.Hsolid_calciner_in:.6e} W"
        )

        print(
            f"Hsolid_calciner_out    = "
            f"{state.Hsolid_calciner_out:.6e} W"
        )

        print(
            f"Wall_loss_calciner     = "
            f"{state.Wall_loss_calciner:.6e} W"
        )

        print(
            f"Calcination_Q_sink     = "
            f"{state.Calcination_Q_sink:.6e} W"
        )

        print("----------------------------------------------")

        # ======================================================
        # ENERGY IN
        # ======================================================

        calciner_energy_in = (
            state.Hgas_calciner_in
            + state.Hsolid_calciner_in
        )

        # ======================================================
        # ENERGY OUT
        #
        # Includes:
        #   gas
        #   solid
        #   wall loss
        #   calcination reaction
        # ======================================================

        calciner_energy_out = (
            state.Hgas_calciner_out
            + state.Hsolid_calciner_out
            + state.Wall_loss_calciner
            + state.Calcination_Q_sink
        )

        # ======================================================
        # ENERGY RESIDUAL
        # ======================================================

        calciner_residual = (
            calciner_energy_in
            - calciner_energy_out
        )

        print(
            f"Energy_in              = "
            f"{calciner_energy_in:.6e} W"
        )

        print(
            f"Energy_out             = "
            f"{calciner_energy_out:.6e} W"
        )

        print(
            f"Residual               = "
            f"{calciner_residual:.6e} W"
        )

        print("----------------------------------------------")

        print(
            f"Calciner_energy_balance = "
            f"{state.Calciner_energy_balance:.6e} W"
        )

        print(
            f"Calciner_energy_balance_relative = "
            f"{state.Calciner_energy_balance_relative:.6e}"
        )

        print("----------------------------------------------")

        # ======================================================
        # CALCINER TEMPERATURE DEBUG
        # ======================================================

        print(
            "\n========== "
            "CALCINER TEMPERATURES "
            "=========="
        )

        print(
            f"Tg_in  = {Tg_in:.3f} K"
        )

        print(
            f"Tg_out = "
            f"{state.Tg_calciner[0]:.3f} K"
        )

        print(
            f"Ts_in  = {Ts_in:.3f} K"
        )

        print(
            f"Ts_out = "
            f"{state.Ts_calciner[-1]:.3f} K"
        )

        print(
            f"Tw_in  = "
            f"{state.Tw_calciner[0]:.3f} K"
        )

        print(
            f"Tw_out = "
            f"{state.Tw_calciner[-1]:.3f} K"
        )

        print(
            "============================================"
        )
        
        print(
            "[CALCINER FINAL CHECK] "
            f"m_dot_CaCO3_out_calciner = "
            f"{getattr(state, 'm_dot_CaCO3_out_calciner', 'MISSING')}"
        )

        return state


    # ======================================================
    # TEMPERATURE FROM ENTHALPY
    # ======================================================

    def gas_temperature_from_enthalpy(self, H, state):

        m_dot_g = state.m_dot_g

        # H = m_dot_g * h_gas(T, T_ref)
        #
        # Therefore solve:
        #
        # h_gas(T, T_ref) = H / m_dot_g
        #
        # using numerical inversion.

        h_target = (
            H
            / (
                m_dot_g
                + self.eps
            )
        )

        T_low = 200.0
        T_high = 4000.0

        for _ in range(100):

            T_mid = (
                0.5
                * (
                    T_low
                    + T_high
                )
            )

            h_mid = h_gas(
                T_mid,
                self.T_ref,
            )

            if h_mid < h_target:

                T_low = T_mid

            else:

                T_high = T_mid

        return (
            0.5
            * (
                T_low
                + T_high
            )
        )


    def solid_temperature_from_enthalpy(
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




