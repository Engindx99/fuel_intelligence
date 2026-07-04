import numpy as np


class Cooler:

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
            self.rho_wall * self.V_wall_cell * self.Cp_wall
        )

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, Q_in_cooler, dt):

        # ================= GRADIENTS (NO ALLOC) =================
        dTg_dz = self._dTg_dz
        dTs_dz = self._dTs_dz

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]

        # ======================================================
        # NO INTERNAL HEAT GENERATION
        # Cooler only transports incoming hot gas.
        # ======================================================
        q_vol = 0.0


        # ================= HEAT TRANSFER =================
        q_gs = (self.hv_gs * self.a_gs * (Tg - Ts))
        q_gw = (self.hv_gw * self.a_gw * (Tg - Tw))
        q_ws = (self.hv_ws * self.a_ws * (Ts - Tw))

        # ================= SOLID CAPACITY =================
        effective = 0.01
        C_s = self._rho_s_Vcell_Cp_s
        effective_C_s = effective * C_s

        # ================= GAS CAPACITY =================
        C_g = self._rho_g_Vcell_Cp_g

        # ================= WALL CAPACITY =================
        C_w = self._rho_wall_Vwall_cell_Cp

        # ================= WALL HEAT LOSS =================
        q_loss = (
            self.h_ext
            * self.A_wall
            * (Tw - self.T_amb)
        ) / (self.V_cell + self.eps)

        wall_loss = np.sum(q_loss * self.V_cell)
        
        wall_debug = {
        "q_loss_mean": float(np.mean(q_loss)),
        "q_loss_total": float(wall_loss),
        "A_wall": float(self.A_wall),
        "V_cell": float(self.V_cell),
        "N": len(Tw),
    }

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
        if not isinstance(state.Tg_cooler, np.ndarray):
            raise TypeError("Tg_cooler must be np.ndarray")

        if state.Tg_cooler.shape != (5,):
            raise ValueError(
                f"Cooler state corrupted: {state.Tg_cooler.shape}"
            )

        # ================= STORE OLD STATES =================
        state.Tg_cooler_old = state.Tg_cooler.copy()
        state.Ts_cooler_old = state.Ts_cooler.copy()
        state.Tw_cooler_old = state.Tw_cooler.copy()

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
            state.Tg_cooler,
            state.Ts_cooler,
            state.Tw_cooler,
            Q_in_cooler=state.Hgas_preheater_out,
            dt=dt,
        )

        # ================= UPDATE STATES =================
        state.Tg_cooler = Tg
        state.Ts_cooler = Ts
        state.Tw_cooler = Tw

        # ================= WALL LOSS =================
        state.Wall_loss_cooler = float(wall_loss)

        # ================= DEBUG STRUCT (SAFE) =================
        state.wall_debug_cooler = wall_debug or {
            "q_loss_mean": 0.0,
            "q_loss_total": 0.0,
            "A_wall": 0.0,
            "V_cell": 0.0,
            "N": 0,
        }
        
        state.q_loss_mean_cooler = state.wall_debug_cooler["q_loss_mean"]
        state.A_wall_cooler      = state.wall_debug_cooler["A_wall"]
        state.V_cell_cooler      = state.wall_debug_cooler["V_cell"]
        state.N_cooler           = state.wall_debug_cooler["N"]

        # ================= OUTPUT ENTHALPY =================
        state.Hgas_cooler_out = self.gas_enthalpy_out(
            state.Tg_cooler
        )

        # ======================================================
        # STORED ENERGY
        # ======================================================

        # Gas
        state.Cooler_gas_stored = np.sum(
            self._rho_g_Vcell_Cp_g
            * (state.Tg_cooler - state.Tg_cooler_old)
            / dt
        )

        # Solid
        state.Cooler_solid_stored = np.sum(
            self._rho_s_Vcell_Cp_s
            * (state.Ts_cooler - state.Ts_cooler_old)
            / dt
        )

        # Wall
        state.Cooler_wall_stored = np.sum(
            self._rho_wall_Vwall_cell_Cp
            * (state.Tw_cooler - state.Tw_cooler_old)
            / dt
        )

        # Total
        state.Cooler_stored_energy_change = (
            state.Cooler_gas_stored
            + state.Cooler_solid_stored
            + state.Cooler_wall_stored
        )

        # ======================================================
        # ENERGY BALANCE
        # ======================================================
        state.Cooler_energy_balance = (
            state.Hgas_preheater_out
            - state.Hgas_cooler_out
            - state.Cooler_stored_energy_change
            - state.Wall_loss_cooler
        )

        return state


    # ======================================================
    # GAS ENTHALPY TO NEXT ZONE
    # ======================================================
    def gas_enthalpy_out(self, Tg):

        m_dot_g = self.rho_g * self.u_g * self.A_cross

        H_out = m_dot_g * self.Cp_g * Tg[-1]

        return H_out