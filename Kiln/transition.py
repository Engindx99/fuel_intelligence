import numpy as np


class Transition:

    def __init__(self, N=5, L=25.0):

        self.N = N
        self.L = L
        self.dz = L / N

        # ================= GEOMETRY =================
        self.D = 4.2
        self.A_cross = np.pi * self.D**2 / 4.0
        self.V_total = self.A_cross * self.L
        self.V_cell = self.V_total / self.N

        # ================= INTERFACIAL AREA =================
        self.epsilon_bed = 0.35

        a_gs_base = 6.0 * (1.0 - self.epsilon_bed) / self.D

        self.k_interfacial = 1.0

        self.a_gs = self.k_interfacial * a_gs_base
        self.a_gw = 2.0 * np.pi * self.D
        self.a_ws = 0.6 * self.a_gs

        # ================= WALL =================
        self.A_wall = self.a_gw * self.L
        self.h_ext = 12.0

        self.T_amb = 300.0
        self.V_wall = self.A_wall * 0.05

        # ================= PROPERTIES =================
        self.rho_g = 4.1
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ================= VELOCITIES =================
        self.u_g = 1.4
        self.u_s = 0.005

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
        self._rho_s_Cp_s = self.rho_s * self.Cp_s
        self._rho_g_Vcell_Cp_g = self.rho_g * self.V_cell * self.Cp_g
        self._rho_wall_Vwall_Cp = self.rho_wall * self.V_wall * self.Cp_wall
        
        # ======================================================
    def thermal_step(self, Tg, Ts, Tw, Q_in_transition, dt, reaction_sink=0.0):

        # ================= GRADIENTS (NO ALLOC) =================
        dTg_dz = self._dTg_dz
        dTs_dz = self._dTs_dz

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]

        # ======================================================
        # HEAT SOURCE
        # Energy received from Burning Zone (W)
        # ======================================================
        m_dot_g = self.rho_g * self.u_g * self.A_cross
        q_in_vol = Q_in_transition / self.V_total

        # ======================================================
        # REACTION ENERGY SINK
        # ======================================================
        sink_density = reaction_sink / (m_dot_g * self.Cp_g * self.L + self.eps)
        
        q_vol = q_in_vol
        q_vol = q_vol - sink_density

        # ================= HEAT TRANSFER =================
        q_gs = (self.hv_gs * self.a_gs * (Tg - Ts)) / self.V_cell
        q_gw = (self.hv_gw * self.a_gw * (Tg - Tw)) / self.V_cell
        q_ws = (self.hv_ws * self.a_ws * (Ts - Tw)) / self.V_cell

        # ================= SOLID CAPACITY =================

        effective = 0.01
        C_s = self._rho_s_Cp_s
        effective_C_s = effective * C_s

        # ================= GAS CAPACITY =================

        C_g = self._rho_g_Vcell_Cp_g

        # ================= WALL CAPACITY =================

        C_w = self._rho_wall_Vwall_Cp

        # ================= HEAT LOSS =================

        q_loss = (
            self.h_ext
            * self.A_wall
            * (Tw - self.T_amb)
        ) / (self.V_cell + self.eps)

        # ================= DYNAMICS =================

        Tg_n = Tg + dt * (
            -self.u_g * dTg_dz
            + (q_vol - q_gs - q_gw) / C_g
        )

        Ts_n = Ts + dt * (
            -self.u_s * dTs_dz
            + (q_gs - q_ws) / effective_C_s
        )

        Tw_n = Tw + dt * (
            (q_gw + q_ws - q_loss) / C_w
        )

        return Tg_n, Ts_n, Tw_n
    
    
    # ======================================================
    # STATE UPDATE
    # ======================================================
    def apply(self, state, dt):
        
        state.Tg_transition_old = state.Tg_transition.copy()
        state.Ts_transition_old = state.Ts_transition.copy()
        state.Tw_transition_old = state.Tw_transition.copy()

        Tg, Ts, Tw = self.thermal_step(
            state.Tg_transition,
            state.Ts_transition,
            state.Tw_transition,
            Q_in_transition=state.Hgas_burning_out,
            dt=dt,
            reaction_sink=getattr(state, "Calcination_Q_sink", 0.0),
        )

        # ================= ENERGY STORED =================
        state.Calcination_stored_energy_change = np.sum(
            self._rho_g_Vcell_Cp_g * (Tg - state.Tg_transition_old) / dt
        )

        # ================= UPDATE STATES =================
        state.Tg_transition = Tg
        state.Ts_transition = Ts
        state.Tw_transition = Tw

        # ================= OUTPUT ENTHALPY =================
        state.Hgas_transition_out = self.gas_enthalpy_out(state.Tg_transition)

        # ================= ENERGY BALANCE =================
        state.Transition_energy_balance = (
            state.Hgas_burning_out
            - state.Hgas_transition_out
            - state.Transition_stored_energy_change
        )

        return state
    
    # ======================================================
    # GAS ENTHALPY TO NEXT ZONE
    # ======================================================
    def gas_enthalpy_out(self, Tg):

        m_dot_g = self.rho_g * self.u_g * self.A_cross
        
        H_out = m_dot_g * self.Cp_g * Tg[-1]

        return H_out
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    