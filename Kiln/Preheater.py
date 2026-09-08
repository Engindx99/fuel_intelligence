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
from dataclasses import dataclass

@dataclass
class PreheaterStage:
    """
    Single physical cyclone-preheater stage.

    Physical interpretation:
        Stage = riser / heat-transfer zone + cyclone

    Initial version:
        - No reaction kinetics
        - No cyclone pressure model
        - No detailed CFD
        - Stage-level thermal bookkeeping only
    """

    stage_id: int

    # ======================================================
    # GAS SIDE
    # ======================================================

    gas_inlet_temperature: float = 0.0
    gas_outlet_temperature: float = 0.0

    gas_inlet_enthalpy: float = 0.0
    gas_outlet_enthalpy: float = 0.0

    # ======================================================
    # SOLID / RAW MEAL SIDE
    # ======================================================

    solid_inlet_temperature: float = 0.0
    solid_outlet_temperature: float = 0.0

    solid_inlet_enthalpy: float = 0.0
    solid_outlet_enthalpy: float = 0.0

    # ======================================================
    # WALL
    # ======================================================

    wall_temperature: float = 0.0

    # ======================================================
    # HEAT TRANSFER
    # ======================================================

    Q_gs: float = 0.0
    Q_gw: float = 0.0
    Q_ws: float = 0.0
    Q_wall_loss: float = 0.0
    Q_reaction: float = 0.0

    # ======================================================
    # ENERGY BALANCE
    # ======================================================

    energy_in: float = 0.0
    energy_out: float = 0.0
    energy_residual: float = 0.0

    # ======================================================
    # REFERENCE OPERATING RANGE
    # ======================================================

    gas_T_min: float = 0.0
    gas_T_max: float = 0.0

    # ======================================================
    # VALIDATION
    # ======================================================

    def __post_init__(self):

        if self.stage_id < 1:
            raise ValueError(
                "stage_id must be >= 1"
            )

    # ======================================================
    # RESET
    # ======================================================

    def reset_diagnostics(self):

        self.Q_gs = 0.0
        self.Q_gw = 0.0
        self.Q_ws = 0.0
        self.Q_wall_loss = 0.0
        self.Q_reaction = 0.0

        self.energy_in = 0.0
        self.energy_out = 0.0
        self.energy_residual = 0.0
        
    def solve(
        self,
        gas_inlet_temperature,
        solid_inlet_temperature,
        m_dot_g,
        m_dot_s,
        state,
        model,
        reaction_power=0.0,
    ):
        """
        Solve one physical preheater stage at steady state.

        The stage receives independent gas and solid inlet conditions.
        Flow direction and inter-stage coupling are handled by Preheater.

        Parameters
        ----------
        gas_inlet_temperature : float
            Gas inlet temperature [K]

        solid_inlet_temperature : float
            Solid inlet temperature [K]

        m_dot_g : float
            Gas mass flow rate [kg/s]

        m_dot_s : float
            Solid mass flow rate [kg/s]

        state : GlobalState
            Current global process state.

        model : Preheater
            Parent Preheater model containing geometry and physical properties.

        reaction_power : float
            Reaction heat contribution [W].
            For the thermal backbone this remains zero.
        """

        self.reset_diagnostics()

        eps = model.eps
        T_ref = model.T_ref
        T_amb = model.T_amb

        # ======================================================
        # INLET CONDITIONS
        # ======================================================

        Tg_in = float(gas_inlet_temperature)
        Ts_in = float(solid_inlet_temperature)

        # ======================================================
        # INITIAL WALL TEMPERATURE
        # ======================================================

        T_low = T_amb
        T_high = max(Tg_in, Ts_in, T_amb + 1.0)

        # ======================================================
        # WALL TEMPERATURE SOLUTION
        #
        # Q_gw + Q_ws - Q_wall_loss = 0
        # ======================================================

        def wall_residual(Tw):

            q_gs, q_gw, q_ws = heat_transfer(
                np.array([Tg_in]),
                np.array([Ts_in]),
                np.array([Tw]),
                model.hv_gs,
                model.hv_gw,
                model.hv_ws,
                model.a_gs,
                model.a_gw,
                model.a_ws,
                zone=model.zone,
            )

            q_loss, Q_wall_loss, _ = wall_losses(
                Tw=np.array([Tw]),
                h_ext=model.h_ext,
                A_wall_cell=model.A_wall_cell,
                V_cell=model.V_cell,
                T_amb=T_amb,
                A_wall_total=model.A_wall,
                N=1,
                refractory_thickness=model.refractory_thickness,
                refractory_conductivity=model.refractory_conductivity,
                eps=eps,
            )

            return (
                float(q_gw[0] + q_ws[0]) * model.V_cell
                - Q_wall_loss
            )

        f_low = wall_residual(T_low)
        f_high = wall_residual(T_high)

        # Expand the upper bracket until a sign change is found.
        for _ in range(50):

            if f_low * f_high <= 0.0:
                break

            T_high *= 1.10
            f_high = wall_residual(T_high)

        # ======================================================
        # BISECTION
        # ======================================================

        if f_low * f_high <= 0.0:

            for _ in range(60):

                T_mid = 0.5 * (T_low + T_high)
                f_mid = wall_residual(T_mid)

                if abs(f_mid) < 1e-6:
                    break

                if f_low * f_mid <= 0.0:
                    T_high = T_mid
                    f_high = f_mid
                else:
                    T_low = T_mid
                    f_low = f_mid

            Tw = 0.5 * (T_low + T_high)

        else:
            # Fallback if the wall equation cannot be bracketed.
            Tw = max(
                T_amb,
                min(Tg_in, Ts_in)
            )

        # ======================================================
        # HEAT TRANSFER AT FINAL WALL TEMPERATURE
        # ======================================================

        q_gs, q_gw, q_ws = heat_transfer(
            np.array([Tg_in]),
            np.array([Ts_in]),
            np.array([Tw]),
            model.hv_gs,
            model.hv_gw,
            model.hv_ws,
            model.a_gs,
            model.a_gw,
            model.a_ws,
            zone=model.zone,
        )

        q_gs = float(q_gs[0])
        q_gw = float(q_gw[0])
        q_ws = float(q_ws[0])

        Q_gs = q_gs * model.V_cell
        Q_gw = q_gw * model.V_cell
        Q_ws = q_ws * model.V_cell

        # ======================================================
        # WALL LOSS
        # ======================================================

        _, Q_wall_loss, _ = wall_losses(
            Tw=np.array([Tw]),
            h_ext=model.h_ext,
            A_wall_cell=model.A_wall_cell,
            V_cell=model.V_cell,
            T_amb=T_amb,
            A_wall_total=model.A_wall,
            N=1,
            refractory_thickness=model.refractory_thickness,
            refractory_conductivity=model.refractory_conductivity,
            eps=eps,
        )

        # ======================================================
        # GAS ENTHALPY
        # ======================================================

        H_gas_in = (
            m_dot_g
            * h_gas(Tg_in, T_ref)
        )

        H_gas_out = (
            H_gas_in
            + reaction_power
            - Q_gs
            - Q_gw
        )

        Tg_out = model.gas_temperature_from_enthalpy(
            H_gas_out,
            state,
        )

        # ======================================================
        # SOLID ENTHALPY
        # ======================================================

        H_solid_in = (
            m_dot_s
            * model.Cp_s
            * (Ts_in - T_ref)
        )

        H_solid_out = (
            H_solid_in
            + Q_gs
            - Q_ws
        )

        Ts_out = (
            T_ref
            + H_solid_out
            / (m_dot_s * model.Cp_s + eps)
        )

        # ======================================================
        # ENERGY BALANCE
        # ======================================================

        energy_in = (
            H_gas_in
            + H_solid_in
        )

        energy_out = (
            H_gas_out
            + H_solid_out
            + Q_wall_loss
            - reaction_power
        )

        energy_residual = energy_in - energy_out

        # ======================================================
        # STORE RESULTS
        # ======================================================

        self.gas_inlet_temperature = Tg_in
        self.gas_outlet_temperature = Tg_out

        self.gas_inlet_enthalpy = H_gas_in
        self.gas_outlet_enthalpy = H_gas_out

        self.solid_inlet_temperature = Ts_in
        self.solid_outlet_temperature = Ts_out

        self.solid_inlet_enthalpy = H_solid_in
        self.solid_outlet_enthalpy = H_solid_out

        self.wall_temperature = Tw

        self.Q_gs = Q_gs
        self.Q_gw = Q_gw
        self.Q_ws = Q_ws
        self.Q_wall_loss = Q_wall_loss
        self.Q_reaction = reaction_power

        self.energy_in = energy_in
        self.energy_out = energy_out
        self.energy_residual = energy_residual

        return self


class Preheater:

    def __init__(self, N=5, L=25.0):

        self.N = N
        self.L = L
        self.dz = L / N

        self.zone = "preheater"
        
        self.stages = [
            PreheaterStage(stage_id=1),
            PreheaterStage(stage_id=2),
            PreheaterStage(stage_id=3),
            PreheaterStage(stage_id=4),
            PreheaterStage(stage_id=5),
        ]
        
        # ======================================================
        # STAGE REFERENCE GAS TEMPERATURE RANGES
        # ======================================================
        # These are operating reference bounds only.
        # They are NOT hard temperature constraints.

        gas_T_bounds_K = [
            (573.15, 583.15),    # Stage 1
            (763.15, 773.15),    # Stage 2
            (903.15, 923.15),    # Stage 3
            (1023.15, 1043.15),  # Stage 4
            (1113.15, 1143.15),  # Stage 5
        ]

        for stage, (T_min, T_max) in zip(
            self.stages,
            gas_T_bounds_K,
        ):
            stage.gas_T_min = T_min
            stage.gas_T_max = T_max


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
    def thermal_step(self, Tg, Ts, Tw, state, reaction_sink=0.0, reaction_heat_cells=None):

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
        # INITIAL ARRAYS
        # ======================================================

        Tg_new = np.empty(self.N, dtype=float)
        Ts_new = np.empty(self.N, dtype=float)
        Tw_new = np.empty(self.N, dtype=float)

        # ======================================================
        # PHYSICAL STAGE 1
        # ======================================================

        stage1 = self.stages[0]

        Tg_1_in = Tg_in
        Ts_1_in = Ts_in

        stage1.solve(
            gas_inlet_temperature=Tg_1_in,
            solid_inlet_temperature=Ts_1_in,
            m_dot_g=m_dot_g,
            m_dot_s=m_dot_s,
            state=state,
            model=self,
            reaction_power=-reaction_heat_cells[0],
        )

        Tg_1_out = stage1.gas_outlet_temperature
        Ts_1_out = stage1.solid_outlet_temperature
        Tw_1 = stage1.wall_temperature

        Tg_new[0] = Tg_1_out
        Ts_new[0] = Ts_1_out
        Tw_new[0] = Tw_1
        
        # ======================================================
        # PHYSICAL STAGE 2
        # ======================================================

        stage2 = self.stages[1]

        # ------------------------------------------------------
        # Stage 2 inlet
        # ------------------------------------------------------

        Tg_2_in = stage1.gas_outlet_temperature
        Ts_2_in = stage1.solid_outlet_temperature

        # ------------------------------------------------------
        # Stage 2 solver
        # ------------------------------------------------------

        stage2.solve(
            gas_inlet_temperature=Tg_2_in,
            solid_inlet_temperature=Ts_2_in,
            m_dot_g=m_dot_g,
            m_dot_s=m_dot_s,
            state=state,
            model=self,
            reaction_power=-reaction_heat_cells[1],
        )

        # ------------------------------------------------------
        # Stage 2 outputs
        # ------------------------------------------------------

        Tg_2_out = stage2.gas_outlet_temperature
        Ts_2_out = stage2.solid_outlet_temperature
        Tw_2 = stage2.wall_temperature

        # ------------------------------------------------------
        # Stage 2 state update
        # ------------------------------------------------------

        Tg_new[1] = Tg_2_out
        Ts_new[1] = Ts_2_out
        Tw_new[1] = Tw_2

        # ------------------------------------------------------
        # Stage 2 diagnostics
        # ------------------------------------------------------

        print()
        print("========== PREHEATER STAGE 2 ==========")

        print(f"Tg_in       = {stage2.gas_inlet_temperature:.3f} K")
        print(f"Tg_out      = {stage2.gas_outlet_temperature:.3f} K")
        print(f"Ts_in       = {stage2.solid_inlet_temperature:.3f} K")
        print(f"Ts_out      = {stage2.solid_outlet_temperature:.3f} K")
        print(f"Tw          = {stage2.wall_temperature:.3f} K")

        print()

        print(f"Hg_in       = {stage2.gas_inlet_enthalpy:.6e} W")
        print(f"Hg_out      = {stage2.gas_outlet_enthalpy:.6e} W")
        print(f"Hs_in       = {stage2.solid_inlet_enthalpy:.6e} W")
        print(f"Hs_out      = {stage2.solid_outlet_enthalpy:.6e} W")

        print()

        print(f"Q_gs        = {stage2.Q_gs:.6e} W")
        print(f"Q_gw        = {stage2.Q_gw:.6e} W")
        print(f"Q_ws        = {stage2.Q_ws:.6e} W")
        print(f"Q_wall_loss = {stage2.Q_wall_loss:.6e} W")
        print(f"Q_reaction  = {stage2.Q_reaction:.6e} W")

        print()

        print(f"Energy in   = {stage2.energy_in:.6e} W")
        print(f"Energy out  = {stage2.energy_out:.6e} W")
        print(f"Residual    = {stage2.energy_residual:.6e} W")

        if abs(stage2.energy_in) > self.eps:
            print(
                f"Rel. error  = "
                f"{stage2.energy_residual / stage2.energy_in:.6e}"
            )

        print("========================================")
        
        
        # ======================================================
        # PHYSICAL STAGE 3
        # ======================================================

        stage3 = self.stages[2]

        # ------------------------------------------------------
        # Stage 3 inlet
        # ------------------------------------------------------

        Tg_3_in = stage2.gas_outlet_temperature
        Ts_3_in = stage2.solid_outlet_temperature

        # ------------------------------------------------------
        # Stage 3 solver
        # ------------------------------------------------------

        stage3.solve(
            gas_inlet_temperature=Tg_3_in,
            solid_inlet_temperature=Ts_3_in,
            m_dot_g=m_dot_g,
            m_dot_s=m_dot_s,
            state=state,
            model=self,
            reaction_power=-reaction_heat_cells[2],
        )

        # ------------------------------------------------------
        # Stage 3 outputs
        # ------------------------------------------------------

        Tg_3_out = stage3.gas_outlet_temperature
        Ts_3_out = stage3.solid_outlet_temperature
        Tw_3 = stage3.wall_temperature

        # ------------------------------------------------------
        # Stage 3 state update
        # ------------------------------------------------------

        Tg_new[2] = Tg_3_out
        Ts_new[2] = Ts_3_out
        Tw_new[2] = Tw_3

        # ------------------------------------------------------
        # Stage 3 diagnostics
        # ------------------------------------------------------

        print()
        print("========== PREHEATER STAGE 3 ==========")

        print(f"Tg_in       = {stage3.gas_inlet_temperature:.3f} K")
        print(f"Tg_out      = {stage3.gas_outlet_temperature:.3f} K")
        print(f"Ts_in       = {stage3.solid_inlet_temperature:.3f} K")
        print(f"Ts_out      = {stage3.solid_outlet_temperature:.3f} K")
        print(f"Tw          = {stage3.wall_temperature:.3f} K")

        print()

        print(f"Hg_in       = {stage3.gas_inlet_enthalpy:.6e} W")
        print(f"Hg_out      = {stage3.gas_outlet_enthalpy:.6e} W")
        print(f"Hs_in       = {stage3.solid_inlet_enthalpy:.6e} W")
        print(f"Hs_out      = {stage3.solid_outlet_enthalpy:.6e} W")

        print()

        print(f"Q_gs        = {stage3.Q_gs:.6e} W")
        print(f"Q_gw        = {stage3.Q_gw:.6e} W")
        print(f"Q_ws        = {stage3.Q_ws:.6e} W")
        print(f"Q_wall_loss = {stage3.Q_wall_loss:.6e} W")
        print(f"Q_reaction  = {stage3.Q_reaction:.6e} W")

        print()

        print(f"Energy in   = {stage3.energy_in:.6e} W")
        print(f"Energy out  = {stage3.energy_out:.6e} W")
        print(f"Residual    = {stage3.energy_residual:.6e} W")

        if abs(stage3.energy_in) > self.eps:
            print(
                f"Rel. error  = "
                f"{stage3.energy_residual / stage3.energy_in:.6e}"
            )

        print("========================================")
        
        # ======================================================
        # PHYSICAL STAGE 4
        # ======================================================

        stage4 = self.stages[3]

        # ------------------------------------------------------
        # Stage 4 inlet
        # ------------------------------------------------------

        Tg_4_in = stage3.gas_outlet_temperature
        Ts_4_in = stage3.solid_outlet_temperature

        # ------------------------------------------------------
        # Stage 4 solver
        # ------------------------------------------------------

        stage4.solve(
            gas_inlet_temperature=Tg_4_in,
            solid_inlet_temperature=Ts_4_in,
            m_dot_g=m_dot_g,
            m_dot_s=m_dot_s,
            state=state,
            model=self,
            reaction_power=-reaction_heat_cells[3],
        )

        # ------------------------------------------------------
        # Stage 4 outputs
        # ------------------------------------------------------

        Tg_4_out = stage4.gas_outlet_temperature
        Ts_4_out = stage4.solid_outlet_temperature
        Tw_4 = stage4.wall_temperature

        # ------------------------------------------------------
        # Stage 4 state update
        # ------------------------------------------------------

        Tg_new[3] = Tg_4_out
        Ts_new[3] = Ts_4_out
        Tw_new[3] = Tw_4

        # ------------------------------------------------------
        # Stage 4 diagnostics
        # ------------------------------------------------------

        print()
        print("========== PREHEATER STAGE 4 ==========")

        print(f"Tg_in       = {stage4.gas_inlet_temperature:.3f} K")
        print(f"Tg_out      = {stage4.gas_outlet_temperature:.3f} K")
        print(f"Ts_in       = {stage4.solid_inlet_temperature:.3f} K")
        print(f"Ts_out      = {stage4.solid_outlet_temperature:.3f} K")
        print(f"Tw          = {stage4.wall_temperature:.3f} K")

        print()

        print(f"Hg_in       = {stage4.gas_inlet_enthalpy:.6e} W")
        print(f"Hg_out      = {stage4.gas_outlet_enthalpy:.6e} W")
        print(f"Hs_in       = {stage4.solid_inlet_enthalpy:.6e} W")
        print(f"Hs_out      = {stage4.solid_outlet_enthalpy:.6e} W")

        print()

        print(f"Q_gs        = {stage4.Q_gs:.6e} W")
        print(f"Q_gw        = {stage4.Q_gw:.6e} W")
        print(f"Q_ws        = {stage4.Q_ws:.6e} W")
        print(f"Q_wall_loss = {stage4.Q_wall_loss:.6e} W")
        print(f"Q_reaction  = {stage4.Q_reaction:.6e} W")

        print()

        print(f"Energy in   = {stage4.energy_in:.6e} W")
        print(f"Energy out  = {stage4.energy_out:.6e} W")
        print(f"Residual    = {stage4.energy_residual:.6e} W")

        if abs(stage4.energy_in) > self.eps:
            print(
                f"Rel. error  = "
                f"{stage4.energy_residual / stage4.energy_in:.6e}"
            )

        print("========================================")
        
        # ======================================================
        # STAGE 5
        # ======================================================

        stage5 = self.stages[4]

        Tg_5_in = stage4.gas_outlet_temperature
        Ts_5_in = stage4.solid_outlet_temperature

        stage5.solve(
            gas_inlet_temperature=Tg_5_in,
            solid_inlet_temperature=Ts_5_in,
            m_dot_g=m_dot_g,
            m_dot_s=m_dot_s,
            state=state,
            model=self,
            reaction_power=-reaction_heat_cells[4],
        )

        Tg_5_out = stage5.gas_outlet_temperature
        Ts_5_out = stage5.solid_outlet_temperature
        Tw_5 = stage5.wall_temperature

        Tg_new[4] = Tg_5_out
        Ts_new[4] = Ts_5_out
        Tw_new[4] = Tw_5


        print("\n========== PREHEATER STAGE 5 ==========")
        print(f"Tg_in       = {Tg_5_in:.3f} K")
        print(f"Tg_out      = {Tg_5_out:.3f} K")
        print(f"Ts_in       = {Ts_5_in:.3f} K")
        print(f"Ts_out      = {Ts_5_out:.3f} K")
        print(f"Tw          = {Tw_5:.3f} K")

        print(f"Hg_in       = {stage5.gas_inlet_enthalpy:.6e} W")
        print(f"Hg_out      = {stage5.gas_outlet_enthalpy:.6e} W")
        print(f"Hs_in       = {stage5.solid_inlet_enthalpy:.6e} W")
        print(f"Hs_out      = {stage5.solid_outlet_enthalpy:.6e} W")

        print(f"Q_gs        = {stage5.Q_gs:.6e} W")
        print(f"Q_gw        = {stage5.Q_gw:.6e} W")
        print(f"Q_ws        = {stage5.Q_ws:.6e} W")
        print(f"Q_wall_loss = {stage5.Q_wall_loss:.6e} W")
        print(f"Q_reaction  = {stage5.Q_reaction:.6e} W")

        print(f"Energy residual = {stage5.energy_residual:.6e} W")
        print("========================================")
     
                
        # ======================================================
        # TOTAL STAGE ENERGY TRANSFERS
        # ======================================================

        Q_gs_total = sum(
            stage.Q_gs
            for stage in self.stages
        )

        Q_gw_total = sum(
            stage.Q_gw
            for stage in self.stages
        )

        Q_ws_total = sum(
            stage.Q_ws
            for stage in self.stages
        )

        Q_reaction_total = sum(
            stage.Q_reaction
            for stage in self.stages
        )

        Q_wall_loss_total = sum(
            stage.Q_wall_loss
            for stage in self.stages
        )
        
        wall_loss = Q_wall_loss_total
        wall_debug = {}
        
        # ======================================================
        # DRYING REACTION DIAGNOSTIC
        # ======================================================

        print("\n========== PREHEATER DRYING REACTION CHECK ==========")

        print(
            f"Drying Q sink total = "
            f"{state.Drying_Q_sink:.6f} W"
        )

        print(
            f"Drying Q cells sum  = "
            f"{np.sum(state.Drying_Q_sink_cells):.6f} W"
        )

        print(
            f"Stage reaction sum  = "
            f"{sum(stage.Q_reaction for stage in self.stages):.6f} W"
        )

        for i, q in enumerate(state.Drying_Q_sink_cells):
            print(
                f"Stage {i+1}: "
                f"Drying Q sink = {q:.6f} W | "
                f"Stage Q reaction = "
                f"{self.stages[i].Q_reaction:.6f} W"
            )

        print("=====================================================")

        # ======================================================
        # DEBUG OUTPUT
        # ======================================================
        print("\n========== PREHEATER THERMAL DEBUG ==========")

        print(f"Q_gs_total       = {Q_gs_total:.6e} W")
        print(f"Q_gw_total       = {Q_gw_total:.6e} W")
        print(f"Q_ws_total       = {Q_ws_total:.6e} W")
        print(f"Q_reaction_total = {Q_reaction_total:.6e} W")
        print(f"Wall_loss        = {Q_wall_loss_total:.6e} W")

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

        Tg_new, Ts_new, Tw_new, wall_loss, wall_debug = (
            self.thermal_step(
                state.Tg_preheater,
                state.Ts_preheater,
                state.Tw_preheater,
                state,
                reaction_sink=state.Preheater_Q_sink,
                reaction_heat_cells=state.Drying_Q_sink_cells,
            )
        )

        state.Tg_preheater = Tg_new
        state.Ts_preheater = Ts_new
        state.Tw_preheater = Tw_new

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

        print(f"Tg_stage1_out = {state.Tg_preheater[0]:.3f} K")
        print(f"Tg_out      = {state.Tg_preheater[-1]:.3f} K")
        print(f"Ts_stage1_out = {state.Ts_preheater[0]:.3f} K")
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
    
    
if __name__ == "__main__":

    stage = PreheaterStage(stage_id=1)

    print()
    print("========== PREHEATER STAGE TEST ==========")
    print(f"stage_id                = {stage.stage_id}")
    print(
        f"gas_inlet_temperature  = "
        f"{stage.gas_inlet_temperature:.3f} K"
    )
    print(
        f"solid_inlet_temperature = "
        f"{stage.solid_inlet_temperature:.3f} K"
    )
    print(
        f"energy_residual        = "
        f"{stage.energy_residual:.3e} W"
    )
    print("PreheaterStage OK")
    print("==========================================")