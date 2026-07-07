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
        self.rho_g = 1.2        # air-like
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1005.0
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
    def thermal_step(self, Tg, Ts, Tw, state, dt):
        
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
        # STATE CHECK
        # ======================================================
        if not isinstance(state.Tg_cooler, np.ndarray):
            raise TypeError("Tg_cooler must be np.ndarray")

        if state.Tg_cooler.shape != (self.N,):
            raise ValueError("Cooler state corrupted")

        # ======================================================
        # STORE OLD STATES
        # ======================================================
        state.Tg_cooler_old = state.Tg_cooler.copy()
        state.Ts_cooler_old = state.Ts_cooler.copy()
        state.Tw_cooler_old = state.Tw_cooler.copy()


        # ======================================================
        # ENERGY IN
        # ======================================================
        state.Hgas_cooler_in = state.Hgas_preheater_out
        state.Hsolid_cooler_in = state.Hsolid_preheater_out

        # ======================================================
        # BOUNDARY CONDITIONS (FROM PREHEATER)
        # ======================================================
        state.Tg_cooler[0] = state.Tg_preheater[-1]
        state.Ts_cooler[0] = state.Ts_preheater[-1]
        

        # ======================================================
        # THERMAL STEP
        # ======================================================
        Tg, Ts, Tw, wall_loss, wall_debug = self.thermal_step(
            state.Tg_cooler,
            state.Ts_cooler,
            state.Tw_cooler,
            state,
            dt,
        )
        


        # ======================================================
        # UPDATE STATES
        # ======================================================
        state.Tg_cooler = Tg
        state.Ts_cooler = Ts
        state.Tw_cooler = Tw

        # ======================================================
        # WALL LOSS
        # ======================================================
        state.Wall_loss_cooler = float(wall_loss)

        # ======================================================
        # ENERGY OUT
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
        # STORED ENERGY
        # ======================================================
        state.Cooler_gas_stored = np.sum(
            self._rho_g_Vcell_Cp_g
            * (state.Tg_cooler - state.Tg_cooler_old)
            / dt
        )

        state.Cooler_solid_stored = np.sum(
            self._rho_s_Vcell_Cp_s
            * (state.Ts_cooler - state.Ts_cooler_old)
            / dt
        )

        state.Cooler_wall_stored = np.sum(
            self._rho_wall_Vwall_cell_Cp
            * (state.Tw_cooler - state.Tw_cooler_old)
            / dt
        )

        state.Cooler_stored_energy_change = (
            state.Cooler_gas_stored
            + state.Cooler_solid_stored
            + state.Cooler_wall_stored
        )

        # ======================================================
        # ENERGY BALANCE
        # ======================================================
        state.Cooler_energy_balance = (
            state.Hgas_cooler_in
            + state.Hsolid_cooler_in
            - state.Hgas_cooler_out
            - state.Hsolid_cooler_out
            - state.Cooler_stored_energy_change
            - state.Wall_loss_cooler
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