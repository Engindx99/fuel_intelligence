import json
import os
import pickle

import numpy as np
import yaml

from physics.physics import solid_mass_flow
from physics.physics import fuel_heat_release
from physics.physics import combustion_axial_distribution
from physics.physics import residence_time
from physics.physics import gas_axial_velocity
from physics.physics import cp_gas, h_gas
from physics.physics import heat_transfer
from physics.physics import radiation
from physics.physics import interfacial_areas
from physics.physics import kiln_geometry
from physics.physics import solid_axial_velocity
from physics.physics import thermal_capacities
from physics.physics import wall_geometry
from physics.physics import wall_losses
from physics.physics import gas_mass_balance
from physics.physics import ZONE_ENERGY_WEIGHTS
from physics.physics import ZONE_HT_CONFIG
from physics.physics import wall_thermal_resistance


from chemistry.reactions import ChemistryModel




def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Burning:
    def __init__(self, N=5, L=60.0, chemistry=None):
        cfg = load_cfg("configs/twin_cfg.yaml")
        plant = cfg.get("plant", {})
        motion = cfg.get("motion", {})
        op = cfg.get("operational", {})

        self.zone = "burning"
        self.chemistry = chemistry or ChemistryModel()
        self.eps = 1e-9

        # ======================================================
        # GEOMETRY
        # ======================================================
        self.N = plant.get("N", N)
        self.L = plant.get("length", L)

        self.dz = self.L / self.N
        self.D = 4.2

        self.A_cross, self.V_total, self.V_cell = kiln_geometry(
            D=self.D,
            L=self.L,
            N=self.N
        )
        
        # ======================================================
        # ENERGY DIAGNOSTICS
        # ======================================================
        self.energy_in = 0.0
        self.energy_out = 0.0
        self.energy_residual = 0.0

        # ======================================================
        # INTERFACIAL & WALL GEOMETRY
        # ======================================================
        self.epsilon_bed = 0.35
        self.k_interfacial = 1.0

        self.a_gs, self.a_ws = interfacial_areas(
            D=self.D,
            epsilon_bed=self.epsilon_bed,
            k_interfacial=self.k_interfacial
        )

        # ======================================================
        # REFRACTORY
        # ======================================================
        self.refractory_thickness = op.get(
            "refractory_thickness",
            0.20
        )

        self.refractory_conductivity = op.get(
            "refractory_conductivity",
            1.8
        )

        self.h_ext = op.get(
            "h_ext",
            10.0
        )

        # ======================================================
        # WALL GEOMETRY
        # ======================================================
        (
            self.wall_perimeter,
            self.A_wall_total,
            self.A_wall_cell,
            self.a_gw,
            self.V_wall
        ) = wall_geometry(
            D=self.D,
            L=self.L,
            N=self.N,
            V_cell=self.V_cell,
            refractory_thickness=self.refractory_thickness
        )

        # ======================================================
        # PHYSICAL PROPERTIES
        # ======================================================
        self.rho_g = 0.30
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ======================================================
        # MOTION
        # ======================================================
        self.rpm_default = motion.get(
            "rpm_default",
            1.5
        )

        self.rpm_min = motion.get(
            "rpm_min",
            1.0
        )

        self.rpm_max = motion.get(
            "rpm_max",
            3.0
        )

        self.slope_deg = motion.get(
            "inclination_deg",
            3.0
        )

        self.fill_fraction = op.get(
            "kiln_load",
            0.10
        )

        self.u_s = 0.0
        self.u_g = 0.0

        # ======================================================
        # HEAT TRANSFER
        # ======================================================
        ht = ZONE_HT_CONFIG[self.zone]

        self.hv_gs = ht["hv_gs"]
        self.hv_gw = ht["hv_gw"]
        self.hv_ws = ht["hv_ws"]

        self.h_ext = op.get(
            "h_ext",
            12.0
        )

        self.T_ref = op.get(
            "T_ref",
            298.15
        )

        self.T_amb = op.get(
            "T_amb",
            300.0
        )

        # ======================================================
        # FUEL
        # ======================================================

        self.O2_opt = 3.5
        self.O2_sigma2 = 25.0

        # ======================================================
        # BUFFERS & CACHE
        # ======================================================
        self._dTg_dz = np.zeros(self.N)
        self._dTs_dz = np.zeros(self.N)

        self._rho_g_Vcell_Cp_g = (
            self.rho_g
            * self.V_cell
            * self.Cp_g
        )

        self._rho_s_Vcell_Cp_s = (
            self.rho_s
            * self.V_cell
            * self.Cp_s
        )

        self.V_wall_cell = self.V_wall / self.N

        self._rho_wall_Vwall_cell_Cp = (
            self.rho_wall
            * self.V_wall_cell
            * self.Cp_wall
        )
        

    def thermal_step(self, Tg, Ts, Tw, state, inputs, u_g, u_s):

        # ======================================================
        # INPUTS
        # ======================================================

        fuel_rate_total = inputs.get("Fuel_rate_total", 1.0)
        O2 = inputs.get("O2", 3.5)

        T_ref = self.T_ref

        # ======================================================
        # MASS FLOW / THERMAL CAPACITY
        # ======================================================

        # m_dot_g is calculated centrally in main.py.
        m_dot_g = state.m_dot_g

        m_dot_s = state.m_dot_s

        # ======================================================
        # SOLID THERMAL CAPACITY
        # ======================================================

        Cp_s = self.Cp_s
        Cs = m_dot_s * Cp_s

        # ======================================================
        # GEOMETRY
        # ======================================================

        N = len(Tg)
        V_cell = self.V_cell

        # ======================================================
        # HEAT TRANSFER PARAMETERS
        # ======================================================

        hv_gs = self.hv_gs
        hv_gw = self.hv_gw
        hv_ws = self.hv_ws

        a_gs = self.a_gs
        a_gw = self.a_gw
        a_ws = self.a_ws

        K_gs = hv_gs * a_gs
        K_gw = hv_gw * a_gw
        K_ws = hv_ws * a_ws

        # ======================================================
        # FUEL HEAT RELEASE
        # ======================================================

        (
            Q_petcoke,
            Q_burning
        ) = fuel_heat_release(
            fuel_rate_total=fuel_rate_total,
            O2=O2,
            O2_opt=self.O2_opt,
            O2_sigma2=self.O2_sigma2,
            LHV={
                "petcoke": 32.0e6,
            },
            inputs=inputs,
            eps=self.eps,
        )

        # ======================================================
        # COMBUSTION DISTRIBUTION
        # ======================================================

        weights = ZONE_ENERGY_WEIGHTS["burning"]["axial"]

        if len(weights) != N:
            raise ValueError(
                f"Burning axial weights length ({len(weights)}) "
                f"must match N ({N})."
            )

        q_cell = combustion_axial_distribution(
            Q_total=Q_burning,
            weights=weights,
        )

        # ======================================================
        # WALL THERMAL RESISTANCE
        # ======================================================

        R_ref, R_conv, R_total = wall_thermal_resistance(
            refractory_thickness=self.refractory_thickness,
            refractory_conductivity=self.refractory_conductivity,
            h_ext=self.h_ext,
            A_wall_cell=self.A_wall_cell,
        )

        # ======================================================
        # PICARD ITERATION
        # ======================================================

        max_iter = 100
        tol = 1e-6
        relaxation = 0.5

        Tg_iter = np.asarray(Tg, dtype=float).copy()
        Ts_iter = np.asarray(Ts, dtype=float).copy()
        Tw_iter = np.asarray(Tw, dtype=float).copy()

        converged = False
        error = np.inf

        for iteration in range(max_iter):

            # ==================================================
            # GAS PROPERTIES AT CURRENT ITERATE
            # ==================================================

            Cp_g_iter = cp_gas(Tg_iter)

            # ==================================================
            # RADIATION
            # ==================================================

            q_gs_rad = radiation(
                Tg_iter,
                Ts_iter,
                zone="burning",
                area=a_gs
            )

            q_gw_rad = radiation(
                Tg_iter,
                Tw_iter,
                zone="burning",
                area=a_gw
            )

            q_ws_rad = radiation(
                Ts_iter,
                Tw_iter,
                zone="burning",
                area=a_ws
            )

            # ==================================================
            # LINEAR SYSTEM
            # ==================================================

            n_unknowns = 3 * N

            A = np.zeros((n_unknowns, n_unknowns))
            b = np.zeros(n_unknowns)

            row = 0

            # ==================================================
            # CELL LOOP
            # ==================================================

            for i in range(N):

                Tg_i = i
                Ts_i = N + i
                Tw_i = 2 * N + i

                # ==================================================
                # GAS LOCAL HEAT CAPACITY
                # ==================================================

                Cp_g_i = cp_gas(Tg_iter[i])

                Cg_i = m_dot_g * Cp_g_i

                # ==================================================
                # GAS ENERGY BALANCE
                #
                # m_dot_g * [h(Tg_i) - h(Tg_up)]
                #
                # + Q_gs_conv
                # + Q_gw_conv
                #
                # + Q_rad
                # - Q_comb
                #
                # = 0
                #
                # Enthalpy is linearized around Tg_iter.
                # ==================================================

                h_i_iter = h_gas(
                    Tg_iter[i],
                    T_ref
                )

                h_linear_const_i = (
                    h_i_iter
                    - Cp_g_i * Tg_iter[i]
                )

                # --------------------------------------------------
                # GAS -> SOLID + GAS -> WALL
                # --------------------------------------------------

                A[row, Tg_i] += (
                    Cg_i
                    + V_cell * K_gs
                    + V_cell * K_gw
                )

                A[row, Ts_i] += (
                    -V_cell * K_gs
                )

                A[row, Tw_i] += (
                    -V_cell * K_gw
                )

                # --------------------------------------------------
                # RADIATION GAS SINK
                # --------------------------------------------------

                radiation_gas_sink = (
                    V_cell
                    * (
                        q_gs_rad[i]
                        + q_gw_rad[i]
                    )
                )

                # --------------------------------------------------
                # GAS UPSTREAM CELL
                # --------------------------------------------------

                if i == N - 1:

                    # Inlet gas temperature is fixed.
                    Tg_in = state.Tg_burning_in

                    h_in = h_gas(
                        Tg_in,
                        T_ref
                    )

                    b[row] = (
                        q_cell[i]
                        - radiation_gas_sink
                        + m_dot_g * h_in
                        - m_dot_g * h_linear_const_i
                    )

                else:

                    Tg_up_i = i + 1

                    Cp_g_up = cp_gas(
                        Tg_iter[i + 1]
                    )

                    h_up_iter = h_gas(
                        Tg_iter[i + 1],
                        T_ref
                    )

                    h_linear_const_up = (
                        h_up_iter
                        - Cp_g_up * Tg_iter[i + 1]
                    )

                    # Upstream enthalpy:
                    #
                    # -m_dot_g * h(T_up)

                    A[row, Tg_up_i] += (
                        -m_dot_g * Cp_g_up
                    )

                    b[row] = (
                        q_cell[i]
                        - radiation_gas_sink
                        - m_dot_g * h_linear_const_i
                        + m_dot_g * h_linear_const_up
                    )

                row += 1

                # ==================================================
                # SOLID ENERGY BALANCE
                # ==================================================

                A[row, Ts_i] += (
                    Cs
                    + V_cell * K_gs
                    + V_cell * K_ws
                )

                A[row, Tg_i] += (
                    -V_cell * K_gs
                )

                A[row, Tw_i] += (
                    -V_cell * K_ws
                )

                # --------------------------------------------------
                # RADIATION SOLID SOURCE
                # --------------------------------------------------

                radiation_solid_source = (
                    V_cell
                    * (
                        q_gs_rad[i]
                        - q_ws_rad[i]
                    )
                )

                # --------------------------------------------------
                # SOLID FLOW DIRECTION
                # --------------------------------------------------

                if i == 0:

                    b[row] = (
                        Cs * state.Ts_burning_in
                        + radiation_solid_source
                    )

                else:

                    Ts_up_i = N + i - 1

                    A[row, Ts_up_i] += (
                        -Cs
                    )

                    b[row] = (
                        radiation_solid_source
                    )

                row += 1

                # ==================================================
                # WALL ENERGY BALANCE
                # ==================================================

                A[row, Tg_i] += (
                    V_cell * K_gw
                )

                A[row, Ts_i] += (
                    V_cell * K_ws
                )

                A[row, Tw_i] += (
                    -V_cell * K_gw
                    -V_cell * K_ws
                    -1.0 / R_total
                )

                # --------------------------------------------------
                # WALL RADIATION SOURCE
                # --------------------------------------------------

                radiation_wall_source = (
                    V_cell
                    * (
                        q_gw_rad[i]
                        + q_ws_rad[i]
                    )
                )

                b[row] = (
                    -self.T_amb / R_total
                    -radiation_wall_source
                )

                row += 1

            # ==================================================
            # SOLVE LINEAR SYSTEM
            # ==================================================

            x_solution = np.linalg.solve(
                A,
                b
            )

            Tg_new = x_solution[:N]

            Ts_new = x_solution[
                N:2 * N
            ]

            Tw_new = x_solution[
                2 * N:3 * N
            ]


            # ==================================================
            # RELAXATION
            # ==================================================

            Tg_iter = (
                relaxation * Tg_new
                + (1.0 - relaxation) * Tg_iter
            )

            Ts_iter = (
                relaxation * Ts_new
                + (1.0 - relaxation) * Ts_iter
            )

            Tw_iter = (
                relaxation * Tw_new
                + (1.0 - relaxation) * Tw_iter
            )


        # ======================================================
        # STEADY-STATE TEMPERATURES
        # ======================================================

        Tg_ss = Tg_iter
        Ts_ss = Ts_iter
        Tw_ss = Tw_iter

        # ======================================================
        # OUTLET TEMPERATURES
        #
        # Gas flows from N-1 -> 0
        # Solid flows from 0 -> N-1
        # ======================================================

        Tg_in = state.Tg_burning_in
        Ts_in = state.Ts_burning_in

        Tg_out = Tg_ss[0]
        Ts_out = Ts_ss[-1]
        Tw_out = Tw_ss[-1]

        # ======================================================
        # FINAL RADIATION
        # ======================================================

        q_gs_rad_final = radiation(
            Tg_ss,
            Ts_ss,
            zone="burning",
            area=a_gs
        )

        q_gw_rad_final = radiation(
            Tg_ss,
            Tw_ss,
            zone="burning",
            area=a_gw
        )

        q_ws_rad_final = radiation(
            Ts_ss,
            Tw_ss,
            zone="burning",
            area=a_ws
        )

        # ======================================================
        # CONVECTIVE HEAT TRANSFER
        # ======================================================

        Qgs_conv = (
            V_cell
            * K_gs
            * (Tg_ss - Ts_ss)
        )

        Qgw_conv = (
            V_cell
            * K_gw
            * (Tg_ss - Tw_ss)
        )

        Qws_conv = (
            V_cell
            * K_ws
            * (Ts_ss - Tw_ss)
        )

        # ======================================================
        # TOTAL HEAT TRANSFER
        # ======================================================

        Qgs = np.sum(
            Qgs_conv
            + V_cell * q_gs_rad_final
        )

        Qgw = np.sum(
            Qgw_conv
            + V_cell * q_gw_rad_final
        )

        Qws = np.sum(
            Qws_conv
            + V_cell * q_ws_rad_final
        )

        # ======================================================
        # WALL LOSS
        # ======================================================

        Q_wall_loss = np.sum(
            (
                Tw_ss - self.T_amb
            ) / R_total
        )

        # ======================================================
        # GAS ENTHALPY BALANCE
        # ======================================================

        Hg_in = (
            m_dot_g
            * h_gas(
                Tg_in,
                T_ref
            )
        )

        Hg_out = (
            m_dot_g
            * h_gas(
                Tg_out,
                T_ref
            )
        )

        gas_energy_change = (
            Hg_out
            - Hg_in
        )

        gas_expected = (
            Q_burning
            - Qgs
            - Qgw
        )

        gas_energy_balance = (
            gas_energy_change
            - gas_expected
        )

        # ======================================================
        # SOLID ENTHALPY BALANCE
        # ======================================================

        Hs_in = (
            m_dot_s
            * Cp_s
            * (
                Ts_in - T_ref
            )
        )

        Hs_out = (
            m_dot_s
            * Cp_s
            * (
                Ts_out - T_ref
            )
        )

        solid_energy_change = (
            Hs_out
            - Hs_in
        )

        solid_expected = (
            Qgs
            - Qws
        )

        solid_energy_balance = (
            solid_energy_change
            - solid_expected
        )

        # ======================================================
        # TOTAL ENERGY BALANCE
        # ======================================================

        energy_in = (
            Hg_in
            + Hs_in
            + Q_burning
        )

        energy_out = (
            Hg_out
            + Hs_out
            + Q_wall_loss
        )

        total_energy_balance = (
            energy_in
            - energy_out
        )
        
        self.energy_in = float(energy_in)
        self.energy_out = float(energy_out)
        self.energy_residual = float(total_energy_balance)


        for i in range(N):

            # ==================================================
            # GAS -> SOLID
            # ==================================================

            Qgs_conv_cell = Qgs_conv[i]

            Qgs_rad_cell = (
                V_cell
                * q_gs_rad_final[i]
            )

            Qgs_cell = (
                Qgs_conv_cell
                + Qgs_rad_cell
            )

            # ==================================================
            # GAS -> WALL
            # ==================================================

            Qgw_conv_cell = Qgw_conv[i]

            Qgw_rad_cell = (
                V_cell
                * q_gw_rad_final[i]
            )

            Qgw_cell = (
                Qgw_conv_cell
                + Qgw_rad_cell
            )

            # ==================================================
            # SOLID -> WALL
            # ==================================================

            Qws_conv_cell = Qws_conv[i]

            Qws_rad_cell = (
                V_cell
                * q_ws_rad_final[i]
            )

            Qws_cell = (
                Qws_conv_cell
                + Qws_rad_cell
            )

            # ==================================================
            # WALL LOSS
            # ==================================================

            Qloss_cell = (
                (
                    Tw_ss[i]
                    - self.T_amb
                )
                / R_total
            )


        # ======================================================
        # RETURN
        # ======================================================

        return (
            Tg_ss,
            Ts_ss,
            Tw_ss,
            Q_petcoke,
            Q_burning,
            energy_in,
            energy_out,
            total_energy_balance,
        )

    
    # ======================================================
    # STATE UPDATE
    # ======================================================

    def apply(self, state, inputs):

        # ======================================================
        # STATE INTEGRITY CHECK
        # ======================================================
        if not isinstance(state.Tg_burning, np.ndarray):
            raise TypeError("Tg_burning must be np.ndarray")

        if state.Tg_burning.shape != (self.N,):
            raise ValueError(
                f"Burning shape corrupted: {state.Tg_burning.shape}"
            )

        # ======================================================
        # INPUTS
        # ======================================================
        fuel_rate_total = inputs.get("Fuel_rate_total", 0.0)
        O2 = inputs.get("O2", 3.5)

        # ======================================================
        # SOLID MOTION
        # ======================================================
        rpm = np.clip(
            inputs.get("rpm", self.rpm_default),
            self.rpm_min,
            self.rpm_max,
        )

        tau = residence_time(
            L=self.L,
            D=self.D,
            slope_deg=self.slope_deg,
            fill_fraction=self.fill_fraction,
            rpm=rpm,
            eps=self.eps,
        )

        u_s = solid_axial_velocity(
            L=self.L,
            D=self.D,
            slope_deg=self.slope_deg,
            fill_fraction=self.fill_fraction,
            rpm=rpm,
            eps=self.eps,
        )

        state.rpm = rpm
        state.residence_time = tau
        state.u_s = u_s
        state.solid_velocity = u_s

        # ======================================================
        # SOLID MASS FLOW
        # ======================================================
        state.m_dot_s = solid_mass_flow(
            inputs.get("Feed_rate_kg_s", 0.0)
        )

        # ======================================================
        # GAS MASS FLOW
        # ======================================================
        m_dot_g = gas_mass_balance(
            fuel_rate_total=fuel_rate_total,
            O2=O2,
            eps=self.eps,
        )

        state.m_dot_g = float(m_dot_g)

        # ======================================================
        # GAS VELOCITY
        # ======================================================
        rho_g = getattr(
            self,
            "rho_g_avg",
            None,
        )

        if rho_g is None:
            rho_g = self.rho_g

        u_g = gas_axial_velocity(
            m_dot_g=state.m_dot_g,
            rho_g=rho_g,
            A_cross=self.A_cross,
            eps=self.eps,
        )

        state.u_g = u_g
        
        
        # ======================================================
        # SOLID INLET HANDOFF
        # ======================================================

        state.Hsolid_burning_in = state.Hsolid_transition_out

        state.Ts_burning_in = (
            self.T_ref
            + state.Hsolid_burning_in
            / (state.m_dot_s * self.Cp_s)
        )

        # ======================================================
        # STEADY-STATE THERMAL SOLUTION
        # ======================================================

        (
            Tg_new,
            Ts_new,
            Tw_new,
            Q_petcoke,
            Q_burning,
            energy_in,
            energy_out,
            total_energy_balance,
        ) = self.thermal_step(
            state.Tg_burning,
            state.Ts_burning,
            state.Tw_burning,
            state,
            inputs,
            u_g,
            u_s,
        )

        # ======================================================
        # UPDATE TEMPERATURE STATES
        # ======================================================

        state.Tg_burning = Tg_new
        state.Ts_burning = Ts_new
        state.Tw_burning = Tw_new

        # ======================================================
        # BURNING CHEMISTRY
        # ======================================================
        state = self.chemistry.apply_burning(state)

        # ======================================================
        # ENTHALPY STATES
        # ======================================================
        state.Hg_burning = (
            state.m_dot_g
            * h_gas(
                state.Tg_burning,
                self.T_ref
            )
        )

        state.Hs_burning = (
            state.m_dot_s
            * self.Cp_s
            * (state.Ts_burning - self.T_ref)
        )

        # ======================================================
        # FUEL HEAT RELEASE STATES
        # ======================================================
        state.Q_petcoke = Q_petcoke
        state.Q_burning = Q_burning

        # ======================================================
        # GAS ENTHALPY OUT
        # ======================================================
        state.Hgas_burning_out = (
            self.gas_enthalpy_out(
                state.Hg_burning
            )
        )

        # ======================================================
        # SOLID ENTHALPY OUT
        # ======================================================
        state.Hsolid_burning_out = (
            self.solid_enthalpy_out(
                state.Hs_burning
            )
        )

        # ======================================================
        # STEADY-STATE STORED ENERGY
        # ======================================================
        state.Burning_gas_stored = 0.0
        state.Burning_solid_stored = 0.0
        state.Burning_wall_stored = 0.0

        state.Burning_stored_energy_change = 0.0

        # ======================================================
        # STEADY-STATE ENERGY BALANCE
        # ======================================================
        Hg_in = state.m_dot_g * h_gas(
            state.Tg_burning_in,
            self.T_ref
        )

        Hs_in = state.m_dot_s * self.Cp_s * (
            state.Ts_burning_in - self.T_ref
        )


        state.Burning_energy_balance = total_energy_balance

        return state

    
    # ======================================================
    # GAS ENTHALPY TO NEXT ZONE
    # ======================================================
    def gas_enthalpy_out(self, Hg):
        return Hg[0]



    # ======================================================
    # SOLID ENTHALPY TO NEXT ZONE
    # ======================================================
    def solid_enthalpy_out(self, Hs):

        return Hs[-1]
