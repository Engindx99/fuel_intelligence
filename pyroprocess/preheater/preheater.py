import numpy as np

from physics.physics import cp_gas
from physics.physics import h_gas
from physics.physics import interfacial_areas
from physics.physics import kiln_geometry
from physics.physics import wall_geometry
from physics.physics import ZONE_HT_CONFIG

from chemistry.reactions import ChemistryModel

from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4
from .stage5 import Stage5


class Preheater:

    def __init__(self, N=5, L=25.0):

        self.N = N
        self.L = L
        self.dz = L / N

        self.zone = "preheater"
        
        self.stages = [
            Stage1(),
            Stage2(),
            Stage3(),
            Stage4(),
            Stage5(),
        ]
        
        # ======================================================
        # ENERGY DIAGNOSTICS
        # ======================================================
        self.energy_in = 0.0
        self.energy_out = 0.0
        self.energy_residual = 0.0
        self.Q_reaction_total = 0.0
        
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
        # HANDOFF DIAGNOSTICS
        # ======================================================

        self.gas_handoff_residuals = []
        self.solid_handoff_residuals = []

    # ======================================================
    # STEADY-STATE THERMAL STEP
    # ======================================================
    def thermal_step(
        self,
        Tg,
        Ts,
        Tw,
        state,
        reaction_sink=0.0,
        reaction_heat_cells=None,
    ):

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

        # Fresh raw meal enters Stage 5.
        Ts_feed = float(state.Feed_temperature)

        # ======================================================
        # INITIAL ARRAYS
        # ======================================================

        Tg_new = np.empty(self.N, dtype=float)
        Ts_new = np.empty(self.N, dtype=float)
        Tw_new = np.empty(self.N, dtype=float)

        # ======================================================
        # REACTION HEAT CELLS
        # ======================================================

        if reaction_heat_cells is None:
            reaction_heat_cells = np.zeros(self.N)

        if len(reaction_heat_cells) != self.N:
            raise ValueError(
                "reaction_heat_cells must have length equal to N"
            )
            
        # ======================================================
        # RESET HANDOFF DIAGNOSTICS
        # ======================================================

        self.gas_handoff_residuals = []
        self.solid_handoff_residuals = []
        
        # ======================================================
        # COUNTER-CURRENT STAGE SOLUTION
        # ======================================================

        # Gas:
        # Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5
        #
        # Solid:
        # Fresh feed -> Stage 5 -> Stage 4 -> Stage 3
        #            -> Stage 2 -> Stage 1

        max_iterations = 50
        tolerance = 1e-5

        # Initial guesses for solid inlet temperature of
        # each stage.
        solid_in_guess = np.full(
            self.N,
            Ts_feed,
            dtype=float,
        )

        solid_out = np.empty(
            self.N,
            dtype=float,
        )

        # ======================================================
        # FIXED-POINT ITERATION
        # ======================================================

        for iteration in range(max_iterations):

            Tg_current = Tg_in

            # --------------------------------------------------
            # GAS SWEEP: Stage 1 -> Stage 5
            # --------------------------------------------------

            for i, stage in enumerate(self.stages):

                # Fresh feed enters Stage 5.
                if i == self.N - 1:
                    Ts_current = Ts_feed
                else:
                    # Solid comes from the next stage.
                    Ts_current = solid_in_guess[i]

                stage.solve(
                    gas_inlet_temperature=Tg_current,
                    solid_inlet_temperature=Ts_current,
                    m_dot_g=m_dot_g,
                    m_dot_s=m_dot_s,
                    state=state,
                    model=self,
                    reaction_power=-reaction_heat_cells[i],
                )

                solid_out[i] = (
                    stage.solid_outlet_temperature
                )

                Tg_current = (
                    stage.gas_outlet_temperature
                )

            # --------------------------------------------------
            # UPDATE SOLID INLET PROFILE
            # --------------------------------------------------

            new_solid_in = np.empty(
                self.N,
                dtype=float,
            )

            # Fresh feed enters Stage 5.
            new_solid_in[-1] = Ts_feed

            # Stage i receives solids from Stage i+1.
            new_solid_in[:-1] = solid_out[1:]

            # --------------------------------------------------
            # CONVERGENCE
            # --------------------------------------------------

            solid_error = np.max(
                np.abs(
                    new_solid_in
                    - solid_in_guess
                )
            )

            solid_in_guess = new_solid_in

            if solid_error < tolerance:
                break

        else:
            raise RuntimeError(
                "Preheater counter-current solution did not converge: "
                f"solid_error={solid_error:.6e} K"
            )

        # ======================================================
        # FINAL CONSISTENT SWEEP
        # ======================================================

        Tg_current = Tg_in

        for i, stage in enumerate(self.stages):

            if i == self.N - 1:
                Ts_current = Ts_feed
            else:
                Ts_current = solid_in_guess[i]

            stage.solve(
                gas_inlet_temperature=Tg_current,
                solid_inlet_temperature=Ts_current,
                m_dot_g=m_dot_g,
                m_dot_s=m_dot_s,
                state=state,
                model=self,
                reaction_power=-reaction_heat_cells[i],
            )

            Tg_new[i] = (
                stage.gas_outlet_temperature
            )

            Ts_new[i] = (
                stage.solid_outlet_temperature
            )

            Tw_new[i] = (
                stage.wall_temperature
            )

            Tg_current = (
                stage.gas_outlet_temperature
            )


        # ======================================================
        # ENTHALPY HANDOFF VALIDATION
        # ======================================================

        self.gas_handoff_residuals = []
        self.solid_handoff_residuals = []

        for i, stage in enumerate(self.stages):

            if i == 0:
                Hgas_expected = H_in
            else:
                Hgas_expected = (
                    self.stages[i - 1].gas_outlet_enthalpy
                )

            if i == self.N - 1:
                Hsolid_expected = (
                    m_dot_s
                    * self.Cp_s
                    * (Ts_feed - self.T_ref)
                )
            else:
                Hsolid_expected = (
                    self.stages[i + 1].solid_outlet_enthalpy
                )

            gas_handoff_residual = (
                stage.gas_inlet_enthalpy
                - Hgas_expected
            )

            solid_handoff_residual = (
                stage.solid_inlet_enthalpy
                - Hsolid_expected
            )

            self.gas_handoff_residuals.append(
                float(gas_handoff_residual)
            )

            self.solid_handoff_residuals.append(
                float(solid_handoff_residual)
            )

        # ======================================================
        # TOTAL STAGE ENERGY TRANSFERS
        # ======================================================

        self.Q_reaction_total = sum(
            stage.Q_reaction
            for stage in self.stages
        )

        Q_wall_loss_total = sum(
            stage.Q_wall_loss
            for stage in self.stages
        )



        # ======================================================
        # GLOBAL PREHEATER ENERGY BALANCE
        # ======================================================

        Hgas_in = state.Hgas_preheater_in

        Hsolid_in = (
            m_dot_s
            * self.Cp_s
            * (Ts_feed - self.T_ref)
        )

        Hgas_out = self.stages[-1].gas_outlet_enthalpy

        Hsolid_out = self.stages[0].solid_outlet_enthalpy

        self.energy_in = (
            Hgas_in
            + Hsolid_in
        )

        self.energy_out = (
            Hgas_out
            + Hsolid_out
            + Q_wall_loss_total
        )

        self.energy_residual = (
            self.energy_in
            + self.Q_reaction_total
            - self.energy_out
        )

        # ======================================================
        # WALL LOSS
        # ======================================================

        wall_loss = Q_wall_loss_total
        wall_debug = {}

        # ======================================================
        # RETURN
        # ======================================================

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
    def apply(self, state):

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

        state.Hsolid_preheater_in = (
            state.m_dot_s
            * self.Cp_s
            * (state.Feed_temperature - self.T_ref)
        )

        # ======================================================
        # BOUNDARY CONDITIONS
        # ======================================================

        state.Tg_preheater[0] = state.Tg_calciner[0]
        state.Ts_preheater[-1] = state.Feed_temperature

        # ======================================================
        # PREHEATER CHEMISTRY
        # ======================================================

        state = self.chemistry.apply_preheater(state)

        # ======================================================
        # THERMAL STEP
        # ======================================================

        (
            Tg_new,
            Ts_new,
            Tw_new,
            wall_loss,
            wall_debug,
        ) = self.thermal_step(
            state.Tg_preheater,
            state.Ts_preheater,
            state.Tw_preheater,
            state,
            reaction_sink=state.Preheater_Q_sink,
            reaction_heat_cells=state.Drying_Q_sink_cells,
        )

        # Reaction energy accounting is validated through
        # self.Q_reaction_total and Preheater_energy_balance.

        state.Tg_preheater = Tg_new
        state.Ts_preheater = Ts_new
        state.Tw_preheater = Tw_new

        state.Wall_loss_preheater = float(wall_loss)

        # ======================================================
        # ENERGY OUT
        # ======================================================

        state.Hgas_preheater_out = self.gas_enthalpy_out(
            state.Tg_preheater,
            state,
        )

        state.Hsolid_preheater_out = self.solid_enthalpy_out(
            state.Ts_preheater,
            state,
        )

        # ======================================================
        # ENERGY BALANCE
        # ======================================================

        state.Preheater_energy_balance = (
            state.Hgas_preheater_in
            + state.Hsolid_preheater_in
            - state.Preheater_Q_sink
            - state.Hgas_preheater_out
            - state.Hsolid_preheater_out
            - state.Wall_loss_preheater
        )

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
            * (Ts[0] - self.T_ref)
        )

        return H_solid_out
    