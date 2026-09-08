import numpy as np
from physics.physics import solid_mass_flow
from physics.physics import cp_gas, h_gas
from physics.physics import fuel_heat_release
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
from chemistry.reactions import ChemistryModel


class Preheater:

    def __init__(self, N=5, L=25.0):

        self.N = N
        self.L = L
        self.dz = L / N

        # ================= ZONE =================
        self.zone = "preheater"
        
        self.chemistry = ChemistryModel()

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
        self.refractory_thickness = 0.20
        self.refractory_conductivity = 1.8

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

        # ================= FLOW =================
        self.u_g = 0.0
        self.u_s = 0.0
        self.fill_fraction = 0.10

        # ================= HEAT TRANSFER =================
        cfg = ZONE_HT_CONFIG[self.zone]

        self.hv_gs = cfg["hv_gs"]
        self.hv_gw = cfg["hv_gw"]
        self.hv_ws = cfg["hv_ws"]

        # ================= BUFFERS =================
        self._dTg_dz = np.zeros(N)
        self._dTs_dz = np.zeros(N)

        # ================= CACHE (AFTER V_CELL) =================
        self._rho_g_Vcell_Cp_g = (
            self.rho_g * self.V_cell * self.Cp_g
        )

        self._rho_s_Vcell_Cp_s = (
            self.rho_s * self.V_cell * self.Cp_s
        )

        self.V_wall_cell = self.V_wall / self.N

        self._rho_wall_Vwall_cell_Cp = (
            self.rho_wall * self.V_wall_cell * self.Cp_wall
        )

    # ======================================================
    # STEADY-STATE THERMAL STEP
    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, state, reaction_sink=0.0):

        # ======================================================
        # INPUTS
        # ======================================================
        m_dot_g = state.m_dot_g
        m_dot_s = state.m_dot_s

        # ======================================================
        # GAS INLET TEMPERATURE FROM ENTHALPY
        # ======================================================
        # Zone-to-zone gas energy handoff is defined by enthalpy.

        H_in = state.Hgas_preheater_in

        print("\n========== PREHEATER INPUT CHECK ==========")
        print(f"Hgas_preheater_in = {H_in:.6e} W")
        print(f"Hgas_calciner_out = {state.Hgas_calciner_out:.6e} W")
        print(f"m_dot_g           = {state.m_dot_g:.6e} kg/s")
        print("============================================")

        if H_in <= 0.0:
            raise RuntimeError(
                "Preheater gas inlet enthalpy is zero or negative: "
                f"Hgas_preheater_in={H_in:.6e} W, "
                f"Hgas_calciner_out={state.Hgas_calciner_out:.6e} W"
            )

        Tg_in = self.gas_temperature_from_enthalpy(
            H_in,
            state,
        )

        # Solid inlet remains temperature-based for now
        Ts_in = Ts[0]

        # ======================================================
        # PREHEATER THERMAL INLET DEBUG
        # ======================================================

        print(
            "========== PREHEATER THERMAL INLET =========="
        )

        print(
            f"Hgas_preheater_in = "
            f"{H_in:.6e} W"
        )

        print(
            f"Tg_from_H         = "
            f"{Tg_in:.3f} K"
        )

        print(
            f"Tg_array[0]       = "
            f"{Tg[0]:.3f} K"
        )

        print(
            f"Ts_in             = "
            f"{Ts_in:.3f} K"
        )

        print(
            "=============================================="
        )

        # ======================================================
        # REACTION SINK
        # ======================================================

        q_vol = -reaction_sink / (
            self.V_total + self.eps
        )

        # ======================================================
        # INITIAL ARRAYS
        # ======================================================

        Tg_new = Tg.copy()
        Ts_new = Ts.copy()
        Tw_new = Tw.copy()

        # ======================================================
        # INLET CONDITIONS
        # ======================================================

        Tg_new[0] = Tg_in
        Ts_new[0] = Ts_in

        # ======================================================
        # ENERGY ACCUMULATORS
        # ======================================================

        Q_gs_total = 0.0
        Q_gw_total = 0.0
        Q_ws_total = 0.0
        Q_reaction_total = 0.0

        # ======================================================
        # AXIAL INTEGRATION
        # ======================================================
        for i in range(self.N - 1):

            Tg_i = Tg_new[i]
            Ts_i = Ts_new[i]
            Tw_i = Tw_new[i]

            print(
                f"\n[PREHEATER CELL {i}]"
                f"\n  Tg_i       = {Tg_i:.3f} K"
                f"\n  Ts_i       = {Ts_i:.3f} K"
                f"\n  Tw_i       = {Tw_i:.3f} K"
                f"\n  Tg_in     = {Tg_in:.3f} K"
                f"\n  Tg[0]     = {Tg[0]:.3f} K"
                f"\n  Tg_new[0] = {Tg_new[0]:.3f} K"
                f"\n  Hgas_in   = {state.Hgas_preheater_in:.6e} W"
            )

            # ==================================================
            # LOCAL HEAT TRANSFER
            # q -> W/m3
            # ==================================================
            q_gs, q_gw, q_ws = heat_transfer(
                Tg=np.array([Tg_i]),
                Ts=np.array([Ts_i]),
                Tw=np.array([Tw_i]),
                hv_gs=self.hv_gs,
                hv_gw=self.hv_gw,
                hv_ws=self.hv_ws,
                a_gs=self.a_gs,
                a_gw=self.a_gw,
                a_ws=self.a_ws,
                zone=self.zone,
            )

            # ==================================================
            # LOCAL WALL LOSS
            # q_loss -> W/m3
            # ==================================================
            q_loss, _, wall_debug = wall_losses(
                Tw=np.array([Tw_i]),
                h_ext=self.h_ext,
                A_wall_cell=self.A_wall_cell,
                V_cell=self.V_cell,
                T_amb=self.T_amb,
                A_wall_total=self.A_wall,
                N=self.N,
                refractory_thickness=self.refractory_thickness,
                refractory_conductivity=self.refractory_conductivity,
                eps=self.eps,
            )

            # ==================================================
            # SCALAR CONVERSION
            # ==================================================
            q_gs = float(np.asarray(q_gs).ravel()[0])
            q_gw = float(np.asarray(q_gw).ravel()[0])
            q_ws = float(np.asarray(q_ws).ravel()[0])
            q_loss = float(np.asarray(q_loss).ravel()[0])

            # ==================================================
            # CELL POWER
            # W/m3 -> W
            # ==================================================
            Q_gs_cell = q_gs * self.V_cell
            Q_gw_cell = q_gw * self.V_cell
            Q_ws_cell = q_ws * self.V_cell
            Q_reaction_cell = q_vol * self.V_cell

            # ==================================================
            # ACCUMULATE ENERGY TRANSFERS
            # ==================================================
            Q_gs_total += Q_gs_cell
            Q_gw_total += Q_gw_cell
            Q_ws_total += Q_ws_cell
            Q_reaction_total += Q_reaction_cell

            # ==================================================
            # GAS ENERGY BALANCE — ENTHALPY FORM
            # ==================================================

            H_gas_in_cell = (
                m_dot_g * float(h_gas(Tg_i, self.T_ref))
            )
            
            
            # ==================================================
            # GAS INLET ENTHALPY CONSISTENCY CHECK
            # ==================================================

            if i == 0:

                H_state_in = state.Hgas_preheater_in
                H_thermal_in = H_gas_in_cell

                difference = H_thermal_in - H_state_in

                relative_difference = (
                    difference
                    / (abs(H_state_in) + self.eps)
                )

                print(
                    "\n========== PREHEATER GAS INLET ENTHALPY CHECK =========="
                )

                print(
                    f"Tg_in             = {Tg_i:.3f} K"
                )

                print(
                    f"Hgas_preheater_in = {H_state_in:.6e} W"
                )

                print(
                    f"H_thermal_from_T  = {H_thermal_in:.6e} W"
                )

                print(
                    f"Difference        = {difference:.6e} W"
                )

                print(
                    f"Relative difference = {relative_difference:.6e}"
                )

                print(
                    "=========================================================\n"
                )

            H_gas_out_cell = (
                H_gas_in_cell
                + Q_reaction_cell
                - Q_gs_cell
                - Q_gw_cell
            )

            Tg_new[i + 1] = self.gas_temperature_from_enthalpy(
                H_gas_out_cell,
                state,
            )


            # ==================================================
            # SOLID ENERGY BALANCE
            # ==================================================

            dTs_dz = (
                Q_gs_cell
                - Q_ws_cell
            ) / (
                (m_dot_s * self.Cp_s + self.eps) * self.dz
            )

            Ts_new[i + 1] = (
                Ts_i
                + self.dz * dTs_dz
            )


            # ==================================================
            # LOCAL ENERGY CHECK
            # ==================================================

            H_gas_expected_out = (
                H_gas_in_cell
                + Q_reaction_cell
                - Q_gs_cell
                - Q_gw_cell
            )

            H_solid_in_cell = (
                m_dot_s
                * self.Cp_s
                * (Ts_i - self.T_ref)
            )

            H_solid_expected_out = (
                H_solid_in_cell
                + Q_gs_cell
                - Q_ws_cell
            )

            print(
                f"cell {i}: "
                f"Gas Qin={H_gas_in_cell/1e6:.3f} MW, "
                f"Gas Qout={H_gas_expected_out/1e6:.3f} MW, "
                f"Solid Qin={H_solid_in_cell/1e6:.3f} MW, "
                f"Solid Qout={H_solid_expected_out/1e6:.3f} MW"
            )


            # ======================================================
            # WALL STEADY-STATE BALANCE
            # ======================================================
            # Steady-state wall has no energy accumulation:
            #
            #     q_gw(Tw) + q_ws(Tw) - q_loss(Tw) = 0
            #
            # Solve directly for Tw.

            def wall_residual(Tw_trial):

                # ----------------------------------------------
                # Gas -> wall + solid -> wall
                # ----------------------------------------------
                _, q_gw_trial, q_ws_trial = heat_transfer(
                    Tg=np.array([Tg_i]),
                    Ts=np.array([Ts_i]),
                    Tw=np.array([Tw_trial]),
                    hv_gs=self.hv_gs,
                    hv_gw=self.hv_gw,
                    hv_ws=self.hv_ws,
                    a_gs=self.a_gs,
                    a_gw=self.a_gw,
                    a_ws=self.a_ws,
                    zone=self.zone,
                )

                # ----------------------------------------------
                # Wall -> environment
                # ----------------------------------------------
                q_loss_trial, _, _ = wall_losses(
                    Tw=np.array([Tw_trial]),
                    h_ext=self.h_ext,
                    A_wall_cell=self.A_wall_cell,
                    V_cell=self.V_cell,
                    T_amb=self.T_amb,
                    A_wall_total=self.A_wall,
                    N=self.N,
                    refractory_thickness=self.refractory_thickness,
                    refractory_conductivity=self.refractory_conductivity,
                    eps=self.eps,
                )

                q_gw_trial = float(
                    np.asarray(q_gw_trial).ravel()[0]
                )

                q_ws_trial = float(
                    np.asarray(q_ws_trial).ravel()[0]
                )

                q_loss_trial = float(
                    np.asarray(q_loss_trial).ravel()[0]
                )

                return (
                    q_gw_trial
                    + q_ws_trial
                    - q_loss_trial
                )


            # ==================================================
            # BISECTION
            # ==================================================

            T_low = self.T_amb
            T_high = max(
                Tg_i,
                Ts_i,
                self.T_amb + 1.0,
            )

            f_low = wall_residual(T_low)
            f_high = wall_residual(T_high)

            # Expand upper bound if necessary
            for _ in range(20):

                if f_low * f_high <= 0.0:
                    break

                T_high *= 1.25

                f_high = wall_residual(T_high)


            if f_low * f_high > 0.0:

                raise RuntimeError(
                    f"Preheater wall steady-state root not bracketed "
                    f"at cell {i}: "
                    f"T_low={T_low:.3f} K, "
                    f"T_high={T_high:.3f} K, "
                    f"f_low={f_low:.6e}, "
                    f"f_high={f_high:.6e}"
                )


            # ==================================================
            # BISECTION SOLVE
            # ==================================================

            for _ in range(60):

                T_mid = 0.5 * (
                    T_low
                    + T_high
                )

                f_mid = wall_residual(T_mid)

                if abs(f_mid) < 1e-9:
                    break

                if f_low * f_mid <= 0.0:

                    T_high = T_mid
                    f_high = f_mid

                else:

                    T_low = T_mid
                    f_low = f_mid


            Tw_new[i] = float(T_mid)

            # ==================================================
            # AXIAL TEMPERATURE UPDATE
            # ==================================================


            Ts_new[i + 1] = (
                Ts_i
                + self.dz * dTs_dz
            )

        # ======================================================
        # OUTLET WALL TEMPERATURE
        # ======================================================
        Tw_new[-1] = Tw_new[-2]

        # ======================================================
        # RE-ENFORCE INLET CONDITIONS
        # ======================================================
        Tg_new[0] = Tg_in
        Ts_new[0] = Ts_in

        # ======================================================
        # TOTAL WALL LOSS
        # ======================================================
        _, wall_loss, wall_debug = wall_losses(
            Tw=Tw_new,
            h_ext=self.h_ext,
            A_wall_cell=self.A_wall_cell,
            V_cell=self.V_cell,
            T_amb=self.T_amb,
            A_wall_total=self.A_wall,
            N=self.N,
            refractory_thickness=self.refractory_thickness,
            refractory_conductivity=self.refractory_conductivity,
            eps=self.eps,
        )

        # ======================================================
        # DEBUG OUTPUT
        # ======================================================
        print("\n========== PREHEATER THERMAL DEBUG ==========")

        print(f"Q_gs_total       = {Q_gs_total:.6e} W")
        print(f"Q_gw_total       = {Q_gw_total:.6e} W")
        print(f"Q_ws_total       = {Q_ws_total:.6e} W")
        print(f"Q_reaction_total = {Q_reaction_total:.6e} W")
        print(f"Wall_loss        = {wall_loss:.6e} W")

        print("----------------------------------------------")

        print(f"Tg_in            = {Tg_new[0]:.3f} K")
        print(f"Tg_out           = {Tg_new[-1]:.3f} K")
        print(f"Ts_in            = {Ts_new[0]:.3f} K")
        print(f"Ts_out           = {Ts_new[-1]:.3f} K")
        print(f"Tw_out           = {Tw_new[-1]:.3f} K")

        print("==============================================")

        return (
            Tg_new,
            Ts_new,
            Tw_new,
            float(wall_loss),
            wall_debug,
        )
        
        
    def gas_temperature_from_enthalpy(self, H, state):

        target_h = H / (state.m_dot_g + self.eps)

        T = 1200.0

        for _ in range(50):

            h = float(h_gas(T, self.T_ref))
            cp = float(cp_gas(T))

            residual = h - target_h

            if abs(residual) < 1e-6:
                break

            T_new = T - residual / (cp + self.eps)

            T = np.clip(
                T_new,
                250.0,
                4000.0,
            )

        return float(T)


    # ======================================================
    # STATE UPDATE
    # ======================================================
    def apply(self, state, dt):

        # ======================================================
        # STATE CHECK
        # ======================================================
        if not isinstance(state.Tg_preheater, np.ndarray):
            raise TypeError("Tg_preheater must be np.ndarray")

        if state.Tg_preheater.shape != (self.N,):
            raise ValueError("Preheater state corrupted")

        
        # ======================================================
        # ENERGY IN
        # ======================================================
        state.Hgas_preheater_in = state.Hgas_calciner_out
        
        
        print("\n========== CALCINER -> PREHEATER GAS HANDOFF ==========")

        print(
            f"Hg_calciner[0]   = {state.Hg_calciner[0]:.6e} W"
        )

        print(
            f"Hg_calciner[-1]  = {state.Hg_calciner[-1]:.6e} W"
        )

        print(
            f"Tg_calciner[0]   = {state.Tg_calciner[0]:.3f} K"
        )

        print(
            f"Tg_calciner[-1]  = {state.Tg_calciner[-1]:.3f} K"
        )

        print(
            f"Hgas_calciner_out = {state.Hgas_calciner_out:.6e} W"
        )

        print(
            f"Hgas_preheater_in = {state.Hgas_preheater_in:.6e} W"
        )

        print("=========================================================\n")

        state.Hsolid_preheater_in = (
            state.m_dot_s
            * self.Cp_s
            * (state.Feed_temperature - self.T_ref)
        )

        # ======================================================
        # BOUNDARY CONDITIONS
        # ======================================================
        state.Tg_preheater[0] = state.Tg_calciner[0]
        state.Ts_preheater[0] = state.Feed_temperature
        
        # ======================================================
        # PREHEATER CHEMISTRY
        # ======================================================

        state = self.chemistry.apply_preheater(
            state
        )

        # ======================================================
        # THERMAL STEP
        # ======================================================
        Tg, Ts, Tw, wall_loss, wall_debug = self.thermal_step(
            state.Tg_preheater,
            state.Ts_preheater,
            state.Tw_preheater,
            state,
            reaction_sink=state.Preheater_Q_sink,
        )

        state.Tg_preheater = Tg
        state.Ts_preheater = Ts
        state.Tw_preheater = Tw

        state.Wall_loss_preheater = float(wall_loss)
        

        # ======================================================
        # ENERGY OUT
        # ======================================================
        state.Hgas_preheater_out = self.gas_enthalpy_out(
            state.Tg_preheater,
            state
        )

        state.Hsolid_preheater_out = self.solid_enthalpy_out(
            state.Ts_preheater,
            state
        )

        # ======================================================
        # ENERGY BALANCE
        # ======================================================
        state.Preheater_energy_balance = (
            state.Hgas_preheater_in
            + state.Hsolid_preheater_in
            - state.Hgas_preheater_out
            - state.Hsolid_preheater_out
            - state.Wall_loss_preheater
        )
        
        
        # ======================================================
        # PREHEATER DEBUG
        # ======================================================
        print()
        print("========== PREHEATER STEADY STATE ==========")

        print(f"Tg_in       = {state.Tg_preheater[0]:.3f} K")
        print(f"Tg_out      = {state.Tg_preheater[-1]:.3f} K")
        print(f"Ts_in       = {state.Ts_preheater[0]:.3f} K")
        print(f"Ts_out      = {state.Ts_preheater[-1]:.3f} K")
        print(f"Tw_out      = {state.Tw_preheater[-1]:.3f} K")

        print()

        print(f"Hg_in       = {state.Hgas_preheater_in:.6e} W")
        print(f"Hg_out      = {state.Hgas_preheater_out:.6e} W")
        print(f"Hs_in       = {state.Hsolid_preheater_in:.6e} W")
        print(f"Hs_out      = {state.Hsolid_preheater_out:.6e} W")

        print()

        print(f"Wall_loss   = {state.Wall_loss_preheater:.6e} W")
        print(f"Q_reaction  = {state.Preheater_Q_sink:.6e} W")

        print()

        energy_in = (
            state.Hgas_preheater_in
            + state.Hsolid_preheater_in
        )

        energy_out = (
            state.Hgas_preheater_out
            + state.Hsolid_preheater_out
            + state.Wall_loss_preheater
            + state.Preheater_Q_sink
        )

        residual = energy_in - energy_out

        print(f"Energy in   = {energy_in:.6e} W")
        print(f"Energy out  = {energy_out:.6e} W")
        print(f"Residual    = {residual:.6e} W")

        if abs(energy_in) > self.eps:
            print(
                f"Rel. error  = "
                f"{residual / energy_in:.6e}"
            )

        print("============================================")

        return state


    # ======================================================
    # GAS ENTHALPY TO NEXT ZONE
    # ======================================================
    def gas_enthalpy_out(self, Tg, state):

        H_gas_out = (
            state.m_dot_g
            * float(h_gas(Tg[-1], self.T_ref))
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