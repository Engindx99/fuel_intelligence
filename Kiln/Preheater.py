import numpy as np
from physics.physics import solid_mass_flow
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
    def thermal_step(self, Tg, Ts, Tw, state, dt, reaction_sink=0.0):
        
        # ======================================================
        # INLET BOUNDARY
        # ======================================================
        Tg_in = Tg[0]
        Ts_in = Ts[0]

        # ======================================================
        # GRADIENTS (NO ALLOCATION)
        # ======================================================
        dTg_dz = self._dTg_dz
        dTs_dz = self._dTs_dz

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]

        # ======================================================
        # REACTION SOURCE / SINK
        # ======================================================

        q_vol = (
            -reaction_sink
            /
            (self.V_total + self.eps)
        )

        # ======================================================
        # HEAT TRANSFER
        # ======================================================
        q_gs, q_gw, q_ws = heat_transfer(
            Tg=Tg,
            Ts=Ts,
            Tw=Tw,
            hv_gs=self.hv_gs,
            hv_gw=self.hv_gw,
            hv_ws=self.hv_ws,
            a_gs=self.a_gs,
            a_gw=self.a_gw,
            a_ws=self.a_ws,
            zone=self.zone,
        )

        # ======================================================
        # WALL LOSSES
        # ======================================================
        q_loss, wall_loss, wall_debug = wall_losses(
            Tw=Tw,
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
        # THERMAL CAPACITIES
        # ======================================================
        C_g, effective_C_s, C_w = thermal_capacities(
            rho_g_Vcell_Cp_g=self._rho_g_Vcell_Cp_g,
            rho_s_Vcell_Cp_s=self._rho_s_Vcell_Cp_s,
            rho_wall_Vwall_cell_Cp=self._rho_wall_Vwall_cell_Cp,
            effective= 0.1,
        )

        # ======================================================
        # ENERGY EQUATIONS
        # ======================================================
        Tg_n = Tg + dt * (
            -state.u_g * dTg_dz
            + (q_vol - q_gs - q_gw) / C_g
        )

        Ts_n = Ts + dt * (
            -state.u_s * dTs_dz
            + (q_gs - q_ws) / effective_C_s
        )

        Tw_n = Tw + dt * (
            (q_gw + q_ws - q_loss) / C_w
        )
        
        # ======================================================
        # ENFORCE INLET BOUNDARY
        # ======================================================
        Tg_n[0] = Tg_in
        Ts_n[0] = Ts_in

        return Tg_n, Ts_n, Tw_n, wall_loss, wall_debug


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
        # STORE OLD STATES
        # ======================================================
        state.Tg_preheater_old = state.Tg_preheater.copy()
        state.Ts_preheater_old = state.Ts_preheater.copy()
        state.Tw_preheater_old = state.Tw_preheater.copy()
        
        # ======================================================
        # ENERGY IN
        # ======================================================
        state.Hgas_preheater_in = state.Hgas_calciner_out
        state.Hsolid_preheater_in = state.Hsolid_calciner_out

        # ======================================================
        # BOUNDARY CONDITIONS (FROM CALCINER)
        # ======================================================
        state.Tg_preheater[0] = state.Tg_calciner[-1]
        state.Ts_preheater[0] = state.Ts_calciner[-1]
        
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
            dt,
            reaction_sink=state.Preheater_Q_sink,
        )

        state.Tg_preheater = Tg
        state.Ts_preheater = Ts
        state.Tw_preheater = Tw

        state.Wall_loss_preheater = float(wall_loss)
        
        
        # ======================================================
        # STORED ENERGY
        # ======================================================

        state.Preheater_gas_stored = np.sum(
            self._rho_g_Vcell_Cp_g *
            (state.Tg_preheater - state.Tg_preheater_old) / dt
        )

        state.Preheater_solid_stored = np.sum(
            self._rho_s_Vcell_Cp_s *
            (state.Ts_preheater - state.Ts_preheater_old) / dt
        )

        state.Preheater_wall_stored = np.sum(
            self._rho_wall_Vwall_cell_Cp *
            (state.Tw_preheater - state.Tw_preheater_old) / dt
        )


        state.Preheater_stored_energy_change = (
            state.Preheater_gas_stored
            + state.Preheater_solid_stored
            + state.Preheater_wall_stored
        )

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

        return state


    # ======================================================
    # GAS ENTHALPY TO NEXT ZONE
    # ======================================================
    def gas_enthalpy_out(self, Tg, state):

        H_gas_out = (
            state.m_dot_g
            * self.Cp_g
            * (Tg[-1] - self.T_ref)
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