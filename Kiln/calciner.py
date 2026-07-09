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

        # ================= FLOW =================
        self.u_g = 0.0
        self.u_s = 0.0

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
    def thermal_step(self, Tg, Ts, Tw, state, dt, reaction_sink=0.0):

        # ======================================================
        # FLOW FIELDS (READ ONLY)
        # ======================================================
        u_g = state.u_g
        u_s = state.u_s 

        # ======================================================
        # GRADIENTS (NO ALLOCATION)
        # ======================================================
        dTg_dz = self._dTg_dz
        dTs_dz = self._dTs_dz

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]


        q_vol = -reaction_sink / (self.V_total + self.eps)

        # ======================================================
        # HEAT TRANSFER (CONVECTION + RADIATION)
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
            effective=0.1,
        )

        # ======================================================
        # ENERGY BALANCE EQUATIONS
        # ======================================================
        Tg_n = Tg + dt * (-u_g * dTg_dz + (q_vol - q_gs - q_gw) / C_g)

        Ts_n = Ts + dt * (-u_s * dTs_dz + (q_gs - q_ws) / effective_C_s)

        Tw_n = Tw + dt * ((q_gw + q_ws - q_loss) / C_w)

        return Tg_n, Ts_n, Tw_n, wall_loss, wall_debug

    # ======================================================
    # STATE UPDATE
    # ======================================================
    def apply(self, state, dt):

        # ======================================================
        # STATE INTEGRITY CHECK
        # ======================================================
        if not isinstance(state.Tg_calciner, np.ndarray):
            raise TypeError("Tg_calciner must be np.ndarray")

        if state.Tg_calciner.shape != (self.N,):
            raise ValueError(
                f"Calciner state corrupted: {state.Tg_calciner.shape}"
            )

        # ======================================================
        # STORE OLD STATES
        # ======================================================
        state.Tg_calciner_old = state.Tg_calciner.copy()
        state.Ts_calciner_old = state.Ts_calciner.copy()
        state.Tw_calciner_old = state.Tw_calciner.copy()

        # ======================================================
        # INCOMING ENTHALPY FROM TRANSITION
        # ======================================================
        state.Hgas_calciner_in = state.Hgas_transition_out
        state.Hsolid_calciner_in = state.Hsolid_transition_out

        # ======================================================
        # INLET TEMPERATURES FROM ENTHALPY
        # ======================================================
        state.Tg_calciner[0] = self.gas_temperature_from_enthalpy(
            state.Hgas_calciner_in,
            state,
        )

        state.Ts_calciner[0] = self.solid_temperature_from_enthalpy(
            state.Hsolid_calciner_in,
            state,
        )

        # ======================================================
        # CALCINER CHEMISTRY
        # ======================================================
        state = self.chemistry.apply_calciner(state)

        # ======================================================
        # THERMAL STEP
        # (Reaction heat sink is handled INSIDE thermal_step)
        # ======================================================
        Tg, Ts, Tw, wall_loss, wall_debug = self.thermal_step(
            state.Tg_calciner,
            state.Ts_calciner,
            state.Tw_calciner,
            state,
            dt,
            reaction_sink=state.Calcination_Q_sink,
        )

        # ======================================================
        # UPDATE STATES
        # ======================================================
        state.Tg_calciner = Tg
        state.Ts_calciner = Ts
        state.Tw_calciner = Tw

        state.Wall_loss_calciner = float(wall_loss)

        # ======================================================
        # UPDATE ENTHALPIES FROM FINAL TEMPERATURES
        # ======================================================
        state.Hg_calciner = (
            state.m_dot_g
            * self.Cp_g
            * (state.Tg_calciner - self.T_ref)
        )

        state.Hs_calciner = (
            state.m_dot_s
            * self.Cp_s
            * (state.Ts_calciner - self.T_ref)
        )

        # ======================================================
        # WALL DEBUG
        # ======================================================
        if wall_debug is None:
            wall_debug = {}

        state.wall_debug_calciner = {
            "q_loss_mean": wall_debug.get("q_loss_mean", 0.0),
            "q_loss_total": wall_debug.get("wall_loss_total", 0.0),
            "A_wall": wall_debug.get("A_wall", 0.0),
            "V_cell": wall_debug.get("V_cell", 0.0),
            "N": wall_debug.get("N", self.N),
        }

        state.q_loss_mean_calciner = (
            state.wall_debug_calciner["q_loss_mean"]
        )

        state.A_wall_calciner = (
            state.wall_debug_calciner["A_wall"]
        )

        state.V_cell_calciner = (
            state.wall_debug_calciner["V_cell"]
        )

        state.N_calciner = (
            state.wall_debug_calciner["N"]
        )

        # ======================================================
        # ENERGY OUT
        # ======================================================
        state.Hgas_calciner_out = state.Hg_calciner[-1]
        state.Hsolid_calciner_out = state.Hs_calciner[-1]

        # ======================================================
        # STORED ENERGY
        # ======================================================
        state.Calciner_gas_stored = np.sum(
            self._rho_g_Vcell_Cp_g
            * (state.Tg_calciner - state.Tg_calciner_old)
            / dt
        )

        state.Calciner_solid_stored = np.sum(
            self._rho_s_Vcell_Cp_s
            * (state.Ts_calciner - state.Ts_calciner_old)
            / dt
        )

        state.Calciner_wall_stored = np.sum(
            self._rho_wall_Vwall_cell_Cp
            * (state.Tw_calciner - state.Tw_calciner_old)
            / dt
        )

        state.Calciner_stored_energy_change = (
            state.Calciner_gas_stored
            + state.Calciner_solid_stored
            + state.Calciner_wall_stored
        )

        # ======================================================
        # ENERGY BALANCE
        # ======================================================
        state.Calciner_energy_balance = (
            state.Hgas_calciner_in
            + state.Hsolid_calciner_in
            - state.Hgas_calciner_out
            - state.Hsolid_calciner_out
            - state.Calciner_stored_energy_change
            - state.Wall_loss_calciner
            - state.Calcination_Q_sink
        )

        # ======================================================
        # RELATIVE ENERGY BALANCE
        # ======================================================
        state.Calciner_energy_balance_relative = (
            state.Calciner_energy_balance
            /
            (
                abs(state.Hgas_calciner_in)
                + abs(state.Hsolid_calciner_in)
                + self.eps
            )
        )

        return state


    # ======================================================
    # TEMPERATURE FROM ENTHALPY
    # ======================================================

    def gas_temperature_from_enthalpy(self, H, state):

        return (
            H /
            (state.m_dot_g * self.Cp_g + self.eps)
            +
            self.T_ref
        )


    def solid_temperature_from_enthalpy(self, H, state):

        return (
            H /
            (state.m_dot_s * self.Cp_s + self.eps)
            +
            self.T_ref
        )


    # ======================================================
    # ENTHALPY TO NEXT ZONE
    # ======================================================

    def gas_enthalpy_out(self, Hg):

        return Hg[-1]


    def solid_enthalpy_out(self, Hs):

        return Hs[-1]