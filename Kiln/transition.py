import numpy as np
from physics.physics import fuel_heat_release
from physics.physics import residence_time
from physics.physics import gas_axial_velocity
from physics.physics import heat_transfer
from physics.physics import radiation_linear
from physics.physics import interfacial_areas
from physics.physics import kiln_geometry
from physics.physics import solid_axial_velocity
from physics.physics import thermal_capacities
from physics.physics import wall_geometry
from physics.physics import wall_losses
from physics.physics import gas_mass_balance


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

        # Gas-solid interfacial area density (m²/m³)
        a_gs_base = 6.0 * (1.0 - self.epsilon_bed) / self.D

        self.k_interfacial = 1.0

        self.a_gs = self.k_interfacial * a_gs_base
        self.a_ws = 0.6 * self.a_gs

        # ================= WALL GEOMETRY =================

        # Kiln inner perimeter (m)
        self.wall_perimeter = np.pi * self.D

        # Total wall area (m²)
        self.A_wall = self.wall_perimeter * self.L

        # Wall area per computational cell (m²)
        self.A_wall_cell = self.A_wall / self.N

        # Gas-wall interfacial area density (m²/m³)
        self.a_gw = self.A_wall_cell / self.V_cell

        # External convection
        self.h_ext = 12.0

        self.T_amb = 300.0

        # Wall volume (5 cm refractory thickness)
        self.V_wall = self.A_wall * 0.05

        # ================= PROPERTIES =================
        self.rho_g = 0.30
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ================= BED =================
        self.fill_fraction = 0.10

        # ================= FLOW (UPDATED IN APPLY) =================
        self.u_g = 0.0
        self.u_s = 0.0

        # ================= HEAT TRANSFER =================
        self.hv_gs = 1300.0
        self.hv_gw = 250.0
        self.hv_ws = 300.0

        # ================= NUMERICAL =================
        self.eps = 1e-9

        # ================= PERFORMANCE BUFFERS =================
        self._dTg_dz = np.zeros(N)
        self._dTs_dz = np.zeros(N)

        # ================= CACHE CONSTANTS =================

        # Gas
        self._rho_g_Vcell_Cp_g = (
            self.rho_g * self.V_cell * self.Cp_g
        )

        # Solid
        self._rho_s_Vcell_Cp_s = (
            self.rho_s * self.V_cell * self.Cp_s
        )

        # Wall (per computational cell)
        self.V_wall_cell = self.V_wall / self.N

        self._rho_wall_Vwall_cell_Cp = (
            self.rho_wall
            * self.V_wall_cell
            * self.Cp_wall
        )
        
        # ======================================================
    def thermal_step(self, Tg, Ts, Tw, state, dt):

        # ======================================================
        # FLOW FIELDS (READ ONLY)
        # ======================================================
        u_g = state.u_g
        u_s = state.u_s

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
            refractory_thickness=0.05,
            refractory_conductivity=1.8,
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
            -u_g * dTg_dz
            + (q_vol - q_gs - q_gw) / C_g
        )

        Ts_n = Ts + dt * (
            -u_s * dTs_dz
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
        if not isinstance(state.Tg_transition, np.ndarray):
            raise TypeError("Tg_transition must be np.ndarray")

        if state.Tg_transition.shape != (5,):
            raise ValueError(
                f"Transition state corrupted: {state.Tg_transition.shape}"
            )

        # ======================================================
        # STORE OLD STATES
        # ======================================================
        state.Tg_transition_old = state.Tg_transition.copy()
        state.Ts_transition_old = state.Ts_transition.copy()
        state.Tw_transition_old = state.Tw_transition.copy()

        # ======================================================
        # INCOMING FLOW FROM BURNING
        # ======================================================
        state.Hgas_transition_in = state.Hgas_burning_out
        state.Hsolid_transition_in = state.Hsolid_burning_out

        state.u_g = getattr(state, "u_g", self.u_g)
        state.u_s = getattr(state, "u_s", self.u_s)
        state.m_dot_g = getattr(state, "m_dot_g", 0.0)

        self.u_g = state.u_g
        self.u_s = state.u_s

        # ======================================================
        # BOUNDARY CONDITION FROM BURNING
        # ======================================================
        state.Tg_transition[0] = state.Tg_burning[-1]
        state.Ts_transition[0] = state.Ts_burning[-1]

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
            state.Tg_transition,
            state.Ts_transition,
            state.Tw_transition,
            state,
            dt,
        )

        # ======================================================
        # UPDATE STATES
        # ======================================================
        state.Tg_transition = Tg
        state.Ts_transition = Ts
        state.Tw_transition = Tw

        # ======================================================
        # WALL LOSS
        # ======================================================
        state.Wall_loss_transition = float(wall_loss)

        # ======================================================
        # WALL DEBUG
        # ======================================================
        if wall_debug is None:
            wall_debug = {}

        state.wall_debug_transition = {
            "q_loss_mean": wall_debug.get("q_loss_mean", 0.0),
            "q_loss_total": wall_debug.get("wall_loss_total", 0.0),
            "A_wall": wall_debug.get("A_wall", 0.0),
            "V_cell": wall_debug.get("V_cell", 0.0),
            "N": wall_debug.get("N", 0),
        }

        state.q_loss_mean_transition = state.wall_debug_transition["q_loss_mean"]
        state.A_wall_transition = state.wall_debug_transition["A_wall"]
        state.V_cell_transition = state.wall_debug_transition["V_cell"]
        state.N_transition = state.wall_debug_transition["N"]

        # ======================================================
        # ENERGY TO NEXT ZONE
        # ======================================================
        state.Hgas_transition_out = self.gas_enthalpy_out(
            state.Tg_transition,
            state,
        )

        state.Hsolid_transition_out = self.solid_enthalpy_out(
            state.Ts_transition,
            state,
        )

        # ======================================================
        # STORED ENERGY
        # ======================================================
        state.Transition_gas_stored = np.sum(
            self._rho_g_Vcell_Cp_g
            * (state.Tg_transition - state.Tg_transition_old)
            / dt
        )

        state.Transition_solid_stored = np.sum(
            self._rho_s_Vcell_Cp_s
            * (state.Ts_transition - state.Ts_transition_old)
            / dt
        )

        state.Transition_wall_stored = np.sum(
            self._rho_wall_Vwall_cell_Cp
            * (state.Tw_transition - state.Tw_transition_old)
            / dt
        )

        state.Transition_stored_energy_change = (
            state.Transition_gas_stored
            + state.Transition_solid_stored
            + state.Transition_wall_stored
        )

        # ======================================================
        # ENERGY BALANCE
        # ======================================================
        state.Transition_energy_balance = (
            state.Hgas_transition_in
            + state.Hsolid_transition_in
            - state.Hgas_transition_out
            - state.Hsolid_transition_out
            - state.Transition_stored_energy_change
            - state.Wall_loss_transition
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