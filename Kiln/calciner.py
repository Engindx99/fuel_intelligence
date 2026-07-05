import numpy as np
from physics.physics import fuel_heat_release
from physics.physics import residence_time
from physics.physics import gas_axial_velocity
from physics.physics import heat_transfer
from physics.physics import interfacial_areas
from physics.physics import kiln_geometry
from physics.physics import solid_axial_velocity
from physics.physics import thermal_capacities
from physics.physics import wall_geometry
from physics.physics import wall_losses
from physics.physics import gas_mass_balance


class Calciner:

    def __init__(self, N=5, L=25.0):

        self.N = N
        self.L = L
        self.dz = L / N

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
        self.refractory_thickness = 0.20      # m
        self.refractory_conductivity = 1.8    # W/mK

        # ================= EXTERNAL WALL =================
        self.h_ext = 12.0
        self.T_amb = 300.0

        # ================= PROPERTIES =================
        self.rho_g = 0.30
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ================= FLOW (UPDATED IN APPLY) =================
        self.u_g = 0.0
        self.u_s = 0.0

        # ================= HEAT TRANSFER =================
        self.hv_gs = 1300.0
        self.hv_gw = 250.0
        self.hv_ws = 300.0

        # ================= PERFORMANCE BUFFERS =================
        self._dTg_dz = np.zeros(N)
        self._dTs_dz = np.zeros(N)

        # ================= CACHE CONSTANTS =================

        # Gas
        self._rho_g_Vcell_Cp_g = (
            self.rho_g
            * self.V_cell
            * self.Cp_g
        )

        # Solid
        self._rho_s_Vcell_Cp_s = (
            self.rho_s
            * self.V_cell
            * self.Cp_s
        )

        # Wall
        self.V_wall_cell = self.V_wall / self.N

        self._rho_wall_Vwall_cell_Cp = (
            self.rho_wall
            * self.V_wall_cell
            * self.Cp_wall
        )

    # ======================================================
    def thermal_step(self,Tg,Ts,Tw,state,dt,reaction_sink=0.0):

        # ======================================================
        # GRADIENTS (NO ALLOC)
        # ======================================================
        dTg_dz = self._dTg_dz
        dTs_dz = self._dTs_dz

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]

        # ======================================================
        # NO INTERNAL HEAT GENERATION
        # ======================================================
        q_vol = 0.0

        # ======================================================
        # CALCINATION ENERGY SINK
        # ======================================================
        sink_density = reaction_sink / (self.V_total + self.eps)
        q_vol -= sink_density

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
            effective=0.01,
        )

        # ======================================================
        # DYNAMICS
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

        return (
            Tg_n,
            Ts_n,
            Tw_n,
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
        if not isinstance(state.Tg_calciner, np.ndarray):
            raise TypeError("Tg_calciner must be np.ndarray")

        if state.Tg_calciner.shape != (5,):
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
        # INCOMING FLOW FROM TRANSITION
        # ======================================================
        state.Hgas_calciner_in = state.Hgas_transition_out
        state.Hsolid_calciner_in = state.Hsolid_transition_out

        state.u_g = getattr(state, "u_g", self.u_g)
        state.u_s = getattr(state, "u_s", self.u_s)
        state.m_dot_g = getattr(state, "m_dot_g", 0.0)

        self.u_g = state.u_g
        self.u_s = state.u_s

        # ======================================================
        # BOUNDARY CONDITION FROM TRANSITION
        # ======================================================
        state.Tg_calciner[0] = state.Tg_transition[-1]
        state.Ts_calciner[0] = state.Ts_transition[-1]

        # Wall temperature is NOT overwritten.
        # It evolves according to the wall energy equation.

        # ======================================================
        # THERMAL STEP
        # ======================================================
        (
            Tg,
            Ts,
            Tw,
            wall_loss,
            wall_debug,
        ) = self.thermal_step(
            state.Tg_calciner,
            state.Ts_calciner,
            state.Tw_calciner,
            state,
            dt,
            reaction_sink=getattr(state, "Calciner_Q_sink", 0.0),
        )

        # ======================================================
        # UPDATE STATES
        # ======================================================
        state.Tg_calciner = Tg
        state.Ts_calciner = Ts
        state.Tw_calciner = Tw

        # ======================================================
        # WALL LOSS
        # ======================================================
        state.Wall_loss_calciner = float(wall_loss)

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
            "N": wall_debug.get("N", 0),
        }

        state.q_loss_mean_calciner = state.wall_debug_calciner["q_loss_mean"]
        state.A_wall_calciner = state.wall_debug_calciner["A_wall"]
        state.V_cell_calciner = state.wall_debug_calciner["V_cell"]
        state.N_calciner = state.wall_debug_calciner["N"]

        # ======================================================
        # ENERGY TO NEXT ZONE
        # ======================================================
        state.Hgas_calciner_out = self.gas_enthalpy_out(
            state.Tg_calciner,
            state,
        )

        state.Hsolid_calciner_out = self.solid_enthalpy_out(
            state.Ts_calciner,
            state,
        )

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
            - state.Calciner_Q_sink
            - state.Calciner_stored_energy_change
            - state.Wall_loss_calciner
        )

        return state


    # ======================================================
    # GAS ENTHALPY TO NEXT ZONE
    # ======================================================
    def gas_enthalpy_out(self, Tg, state):

        H_gas_out = (
            state.m_dot_g
            * self.Cp_g
            * Tg[-1]
        )

        return H_gas_out


    # ======================================================
    # SOLID ENTHALPY TO NEXT ZONE
    # ======================================================
    def solid_enthalpy_out(self, Ts, state):

        fill_fraction = 0.10

        m_dot_s = (
            self.rho_s
            * state.u_s
            * self.A_cross
            * fill_fraction
        )

        H_solid_out = (
            m_dot_s
            * self.Cp_s
            * Ts[-1]
        )

        return H_solid_out