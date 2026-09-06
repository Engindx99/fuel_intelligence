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
from physics.physics import wall_thermal_resistance


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

        self.refractory_conductivity = 1.8

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
        self.LHV = {
            "petcoke": 32e6,
            "coal": 18e6,
            "rdf": 20e6,
            "h2": 120e6
        }

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

        fuel_rate_total = inputs.get(
            "Fuel_rate_total",
            1.0,
        )

        O2 = inputs.get(
            "O2",
            3.5,
        )

        T_ref = self.T_ref

        # ======================================================
        # MASS FLOW / THERMAL CAPACITY
        # ======================================================

        m_dot_g = gas_mass_balance(
            fuel_rate_total=fuel_rate_total,
            O2=O2,
            eps=self.eps
        )

        state.m_dot_g = float(m_dot_g)

        m_dot_s = state.m_dot_s

        Cp_g = self.Cp_g
        Cp_s = self.Cp_s

        Cg = m_dot_g * Cp_g
        Cs = m_dot_s * Cp_s
        
        
        print("\n--- MASS / CAPACITY CHECK ---")
        print(f"m_dot_g = {m_dot_g:.6f} kg/s")
        print(f"m_dot_s = {m_dot_s:.6f} kg/s")
        print(f"Cp_g = {Cp_g:.3f} J/(kg K)")
        print(f"Cp_s = {Cp_s:.3f} J/(kg K)")
        print(f"Cg = {Cg:.3f} W/K")
        print(f"Cs = {Cs:.3f} W/K")
        print(f"Cg/Cs = {Cg/Cs:.6f}")
        
        # ======================================================
        # GRID / GEOMETRY
        # ======================================================

        N = len(Tg)

        V_cell = self.V_cell

        # ======================================================
        # HEAT TRANSFER COEFFICIENTS
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
            Q_coal,
            Q_RDF,
            Q_H2,
            Q_burning,
        ) = fuel_heat_release(
            fuel_rate_total=fuel_rate_total,
            O2=O2,
            O2_opt=self.O2_opt,
            O2_sigma2=self.O2_sigma2,
            LHV=self.LHV,
            inputs=inputs,
            eps=self.eps,
        )

        # ======================================================
        # AXIAL COMBUSTION HEAT DISTRIBUTION
        # ======================================================

        q_cell = combustion_axial_distribution(
            Q_total=Q_burning,
            N=N,
            center=0.65,
            sigma=0.20,
        )
        
        print("\n--- COMBUSTION DISTRIBUTION CHECK ---")
        print(f"Q_burning = {Q_burning/1e6:.6f} MW")
        for i, q in enumerate(q_cell):
            print(f"cell {i}: Qcomb = {q/1e6:.6f} MW")
        print(f"Qcomb sum = {np.sum(q_cell)/1e6:.6f} MW")

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
        # STEADY-STATE ITERATION SETTINGS
        # ======================================================

        max_iter = 100
        tol = 1.0e-6

        # Under-relaxation
        relaxation = 0.5

        # ======================================================
        # INITIAL TEMPERATURE GUESS
        # ======================================================

        Tg_iter = np.asarray(
            Tg,
            dtype=float,
        ).copy()

        Ts_iter = np.asarray(
            Ts,
            dtype=float,
        ).copy()

        Tw_iter = np.asarray(
            Tw,
            dtype=float,
        ).copy()

        converged = False
        error = np.inf

        # ======================================================
        # PICARD ITERATION
        # ======================================================

        for iteration in range(max_iter):

            # ==================================================
            # RADIATION
            #
            # IMPORTANT:
            # Radiation physics is handled by the existing
            # radiation() API.
            # ==================================================

            q_gs_rad = radiation(
                Tg_iter,
                Ts_iter,
                zone="burning",
                area=a_gs,
            )

            q_gw_rad = radiation(
                Tg_iter,
                Tw_iter,
                zone="burning",
                area=a_gw,
            )

            q_ws_rad = radiation(
                Ts_iter,
                Tw_iter,
                zone="burning",
                area=a_ws,
            )

            # ==================================================
            # LINEAR SYSTEM
            # ==================================================

            n_unknowns = 3 * N

            A = np.zeros(
                (
                    n_unknowns,
                    n_unknowns,
                )
            )

            b = np.zeros(
                n_unknowns
            )

            row = 0

            # ==================================================
            # CELL EQUATIONS
            # ==================================================

            for i in range(N):

                Tg_i = i
                Ts_i = N + i
                Tw_i = 2 * N + i

                # ==================================================
                # GAS ENERGY BALANCE
                #
                # Cg*(Tg_i - Tg_up)
                #
                # + V*K_gs*(Tg_i - Ts_i)
                # + V*K_gw*(Tg_i - Tw_i)
                #
                # + V*q_gs_rad
                # + V*q_gw_rad
                #
                # - Q_comb_i = 0
                # ==================================================

                A[row, Tg_i] += (
                    Cg
                    + V_cell * K_gs
                    + V_cell * K_gw
                )

                A[row, Ts_i] += (
                    -V_cell * K_gs
                )

                A[row, Tw_i] += (
                    -V_cell * K_gw
                )

                radiation_gas_sink = (
                    V_cell
                    * (
                        q_gs_rad[i]
                        + q_gw_rad[i]
                    )
                )

                if i == N - 1:

                    b[row] = (
                        Cg
                        * state.Tg_burning_in
                        + q_cell[i]
                        - radiation_gas_sink
                    )

                else:

                    Tg_up_i = i + 1

                    A[row, Tg_up_i] += -Cg

                    b[row] = (
                        q_cell[i]
                        - radiation_gas_sink
                    )

                row += 1

                # ==================================================
                # SOLID ENERGY BALANCE
                #
                # Cs*(Ts_i - Ts_up)
                #
                # - V*K_gs*(Tg_i - Ts_i)
                # + V*K_ws*(Ts_i - Tw_i)
                #
                # - V*q_gs_rad
                # + V*q_ws_rad
                #
                # = 0
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

                radiation_solid_source = (
                    V_cell
                    * (
                        q_gs_rad[i]
                        - q_ws_rad[i]
                    )
                )

                if i == 0:

                    b[row] = (
                        Cs
                        * state.Ts_burning_in
                        + radiation_solid_source
                    )

                else:

                    Ts_up_i = N + i - 1

                    A[row, Ts_up_i] += -Cs

                    b[row] = (
                        radiation_solid_source
                    )

                row += 1

                # ==================================================
                # WALL ENERGY BALANCE
                #
                # V*K_gw*(Tg_i - Tw_i)
                # + V*K_ws*(Ts_i - Tw_i)
                #
                # + V*q_gw_rad
                # + V*q_ws_rad
                #
                # - Q_loss = 0
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

                radiation_wall_source = (
                    V_cell
                    * (
                        q_gw_rad[i]
                        + q_ws_rad[i]
                    )
                )

                b[row] = (
                    -self.T_amb / R_total
                    - radiation_wall_source
                )

                row += 1

            # ==================================================
            # SOLVE
            # ==================================================

            x_solution = np.linalg.solve(
                A,
                b,
            )

            Tg_new = x_solution[:N]

            Ts_new = x_solution[N:2 * N]

            Tw_new = x_solution[2 * N:3 * N]

            # ==================================================
            # CONVERGENCE ERROR
            # ==================================================

            error = max(
                np.max(
                    np.abs(
                        Tg_new - Tg_iter
                    )
                ),
                np.max(
                    np.abs(
                        Ts_new - Ts_iter
                    )
                ),
                np.max(
                    np.abs(
                        Tw_new - Tw_iter
                    )
                ),
            )

            # ==================================================
            # UNDER-RELAXATION
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

            # ==================================================
            # CONVERGENCE CHECK
            # ==================================================

            if error < tol:

                converged = True

                break

        # ======================================================
        # CONVERGENCE WARNING
        # ======================================================

        if not converged:

            print(
                "WARNING: Burning radiation iteration "
                f"did not converge after {max_iter} iterations. "
                f"Final error = {error:.6e} K"
            )

        # ======================================================
        # FINAL STEADY-STATE TEMPERATURES
        # ======================================================

        Tg_ss = Tg_iter
        Ts_ss = Ts_iter
        Tw_ss = Tw_iter

        # ======================================================
        # FINAL RADIATION
        # ======================================================

        q_gs_rad = radiation(
            Tg_ss,
            Ts_ss,
            zone="burning",
            area=a_gs,
        )

        q_gw_rad = radiation(
            Tg_ss,
            Tw_ss,
            zone="burning",
            area=a_gw,
        )

        q_ws_rad = radiation(
            Ts_ss,
            Tw_ss,
            zone="burning",
            area=a_ws,
        )

        # ======================================================
        # CONVECTIVE HEAT TRANSFER
        # ======================================================

        q_gs_conv = (
            K_gs
            * (
                Tg_ss
                - Ts_ss
            )
        )

        q_gw_conv = (
            K_gw
            * (
                Tg_ss
                - Tw_ss
            )
        )

        q_ws_conv = (
            K_ws
            * (
                Ts_ss
                - Tw_ss
            )
        )

        # ======================================================
        # TOTAL HEAT TRANSFER
        #
        # CONVECTION + RADIATION
        # ======================================================

        q_gs_cell = (
            q_gs_conv
            + q_gs_rad
        )

        q_gw_cell = (
            q_gw_conv
            + q_gw_rad
        )

        q_ws_cell = (
            q_ws_conv
            + q_ws_rad
        )

        # ======================================================
        # WALL LOSS
        # ======================================================

        Q_loss_cell = (
            Tw_ss
            - self.T_amb
        ) / R_total

        # ======================================================
        # TOTAL HEAT TRANSFER
        # ======================================================

        Q_gs = (
            V_cell
            * np.sum(q_gs_cell)
        )

        Q_gw = (
            V_cell
            * np.sum(q_gw_cell)
        )

        Q_ws = (
            V_cell
            * np.sum(q_ws_cell)
        )

        Q_wall_loss = np.sum(
            Q_loss_cell
        )

        # ======================================================
        # RADIATION DIAGNOSTICS
        # ======================================================

        Q_gs_rad = (
            V_cell
            * np.sum(q_gs_rad)
        )

        Q_gw_rad = (
            V_cell
            * np.sum(q_gw_rad)
        )

        Q_ws_rad = (
            V_cell
            * np.sum(q_ws_rad)
        )

        # ======================================================
        # OUTLET TEMPERATURES
        # ======================================================

        Tg_out = Tg_ss[0]

        Ts_out = Ts_ss[-1]

        Tw_out = Tw_ss[-1]

        # ======================================================
        # INLET ENTHALPIES
        # ======================================================

        Tg_in = state.Tg_burning_in

        Ts_in = state.Ts_burning_in

        Hg_in = (
            m_dot_g
            * Cp_g
            * (
                Tg_in
                - T_ref
            )
        )

        Hs_in = (
            m_dot_s
            * Cp_s
            * (
                Ts_in
                - T_ref
            )
        )

        # ======================================================
        # OUTLET ENTHALPIES
        # ======================================================

        Hg_out = (
            m_dot_g
            * Cp_g
            * (
                Tg_out
                - T_ref
            )
        )

        Hs_out = (
            m_dot_s
            * Cp_s
            * (
                Ts_out
                - T_ref
            )
        )

        # ======================================================
        # ENERGY CHANGES
        # ======================================================

        gas_energy_change = (
            Hg_out
            - Hg_in
        )

        solid_energy_change = (
            Hs_out
            - Hs_in
        )
        
        # ======================================================
        # SOLID THERMAL CAPACITY DEBUG
        # ======================================================

        print(
            "\n--- SOLID THERMAL CAPACITY ---"
        )

        print(
            f"m_dot_s = {m_dot_s:.6f} kg/s"
        )

        print(
            f"Cp_s    = {Cp_s:.3f} J/(kg K)"
        )

        print(
            f"Cs      = {Cs:.3f} W/K"
        )

        print(
            f"Q_gs - Q_ws = "
            f"{(Q_gs - Q_ws) / 1e6:.6f} MW"
        )

        print(
            f"Delta_H_s = "
            f"{solid_energy_change / 1e6:.6f} MW"
        )

        print(
            f"Delta_Ts = "
            f"{(Ts_out - Ts_in):.3f} K"
        )

        print(
            f"Delta_Ts_expected = "
            f"{solid_energy_change / Cs:.3f} K"
        )

        # ======================================================
        # ENERGY BALANCES
        # ======================================================

        gas_expected = (
            Q_burning
            - Q_gs
            - Q_gw
        )

        solid_expected = (
            Q_gs
            - Q_ws
        )

        gas_energy_balance = (
            gas_energy_change
            - gas_expected
        )

        solid_energy_balance = (
            solid_energy_change
            - solid_expected
        )

        total_energy_balance = (
            Hg_in
            + Hs_in
            + Q_burning
            - Hg_out
            - Hs_out
            - Q_wall_loss
        )

        # ======================================================
        # DEBUG
        # ======================================================

        print(
            "\n========== BURNING STEADY STATE =========="
        )

        print(
            f"Tg_in       = {Tg_in:.3f} K"
        )

        print(
            f"Tg_out      = {Tg_out:.3f} K"
        )

        print(
            f"Ts_in       = {Ts_in:.3f} K"
        )

        print(
            f"Ts_out      = {Ts_out:.3f} K"
        )

        print(
            f"Tw_out      = {Tw_out:.3f} K"
        )

        print(
            f"Q_burning   = {Q_burning:.3f} W"
        )

        print(
            f"Q_gs        = {Q_gs:.3f} W"
        )

        print(
            f"Q_gw        = {Q_gw:.3f} W"
        )

        print(
            f"Q_ws        = {Q_ws:.3f} W"
        )

        print(
            f"Q_wall_loss = {Q_wall_loss:.3f} W"
        )

        print(
            "\n--- RADIATION ---"
        )

        print(
            f"Q_gs_rad    = {Q_gs_rad:.3f} W"
        )

        print(
            f"Q_gw_rad    = {Q_gw_rad:.3f} W"
        )

        print(
            f"Q_ws_rad    = {Q_ws_rad:.3f} W"
        )

        print(
            f"Iterations   = {iteration + 1}"
        )

        print(
            f"Rad. error   = {error:.6e} K"
        )

        print(
            "\n--- ENERGY BALANCE ---"
        )

        print(
            f"Gas balance   = "
            f"{gas_energy_balance:.6e} W"
        )

        print(
            f"Solid balance = "
            f"{solid_energy_balance:.6e} W"
        )

        print(
            f"Total balance = "
            f"{total_energy_balance:.6e} W"
        )

        # ======================================================
        # CELL DIAGNOSTICS
        # ======================================================

        print(
            "\n--- CELL TEMPERATURES ---"
        )

        for i in range(N):

            Qcomb_i = q_cell[i]

            Qgs_i = (
                V_cell
                * q_gs_cell[i]
            )

            Qgw_i = (
                V_cell
                * q_gw_cell[i]
            )

            Qws_i = (
                V_cell
                * q_ws_cell[i]
            )

            Qloss_i = Q_loss_cell[i]

            print(
                f"cell {i}: "
                f"Tg={Tg_ss[i]:.2f} K, "
                f"Ts={Ts_ss[i]:.2f} K, "
                f"Tw={Tw_ss[i]:.2f} K, "
                f"Qcomb={Qcomb_i / 1e6:.3f} MW, "
                f"Qgs={Qgs_i / 1e6:.3f} MW, "
                f"Qgw={Qgw_i / 1e6:.3f} MW, "
                f"Qws={Qws_i / 1e6:.3f} MW, "
                f"Qloss={Qloss_i / 1e6:.3f} MW"
            )

        print(
            "===========================================\n"
        )

        return (
            Tg_ss,
            Ts_ss,
            Tw_ss,
            Q_petcoke,
            Q_coal,
            Q_RDF,
            Q_H2,
            Q_burning,
        )

    
    # ======================================================
    # STATE UPDATE
    # ======================================================

    def apply(self, state, inputs, dt):

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
        # STEADY-STATE THERMAL SOLUTION
        # ======================================================
        (
            Tg,
            Ts,
            Tw,
            Q_petcoke,
            Q_coal,
            Q_RDF,
            Q_H2,
            Q_burning,
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
        state.Tg_burning = Tg
        state.Ts_burning = Ts
        state.Tw_burning = Tw

        # ======================================================
        # BURNING CHEMISTRY
        # ======================================================
        state = self.chemistry.apply_burning(state)

        # ======================================================
        # ENTHALPY STATES
        # ======================================================
        state.Hg_burning = (
            state.m_dot_g
            * self.Cp_g
            * (state.Tg_burning - self.T_ref)
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
        state.Q_coal = Q_coal
        state.Q_RDF = Q_RDF
        state.Q_H2 = Q_H2
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
        state.Burning_energy_balance = (
            state.Q_burning
            - state.Hgas_burning_out
            - state.Hsolid_burning_out
        )

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
