import json
import os
import pickle

import numpy as np
import yaml

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


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Burning:
    def __init__(self, N=5, L=60.0, chemistry=None):
        cfg = load_cfg("configs/twin_cfg.yaml")
        plant, motion, op = cfg.get("plant", {}), cfg.get("motion", {}), cfg.get("operational", {})
        
        self.zone = "burning"
        self.chemistry = chemistry or ChemistryModel()
        self.eps = 1e-9
        
        # Geometry
        self.N, self.L = plant.get("N", N), plant.get("length", L)
        
        self.dz, self.D = self.L / self.N, 4.2
        
        self.A_cross, self.V_total, self.V_cell = kiln_geometry(D=self.D, L=self.L, N=self.N)
        
        # Interfacial & Wall
        self.epsilon_bed, self.k_interfacial = 0.35, 1.0
        
        self.a_gs, self.a_ws = interfacial_areas(D=self.D, epsilon_bed=self.epsilon_bed, k_interfacial=self.k_interfacial)
        
        (self.wall_perimeter, self.A_wall_total, self.A_wall_cell,
         
         self.a_gw, self.V_wall) = wall_geometry(D=self.D, L=self.L, N=self.N, V_cell=self.V_cell)
        
        # Refractory, Props & Motion
        self.refractory_thickness, self.refractory_conductivity = op.get("refractory_thickness", 0.20), 1.8
        
        self.rho_g, self.rho_s, self.rho_wall = 0.30, 1100.0, 3000.0
        
        self.Cp_g, self.Cp_s, self.Cp_wall = 1150.0, 850.0, 1000.0
        
        
        self.rpm_default = motion.get("rpm_default", 1.5)
        
        self.rpm_min = motion.get("rpm_min", 1.0)
        
        self.rpm_max = motion.get("rpm_max", 3.0)
        
        self.slope_deg = motion.get("inclination_deg", 3.0)
        
        self.fill_fraction = op.get("kiln_load", 0.10)
        
        self.u_s = self.u_g = 0.0
        
        # Heat Transfer & Fuel
        ht = ZONE_HT_CONFIG[self.zone]
        self.hv_gs, self.hv_gw, self.hv_ws = ht["hv_gs"], ht["hv_gw"], ht["hv_ws"]
        
        self.h_ext, self.T_ref, self.T_amb = op.get("h_ext", 12.0), op.get("T_ref", 298.15), op.get("T_amb", 300.0)
        
        self.LHV = {"petcoke": 32e6, "coal": 18e6, "rdf": 20e6, "h2": 120e6}
        self.O2_opt, self.O2_sigma2 = 3.5, 25.0
        
        # Buffers & Cache
        self._dTg_dz, self._dTs_dz = np.zeros(self.N), np.zeros(self.N)
        self._rho_g_Vcell_Cp_g = self.rho_g * self.V_cell * self.Cp_g
        self._rho_s_Vcell_Cp_s = self.rho_s * self.V_cell * self.Cp_s
        self.V_wall_cell = self.V_wall / self.N
        self._rho_wall_Vwall_cell_Cp = self.rho_wall * self.V_wall_cell * self.Cp_wall
        

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

        # IMPORTANT:
        # Use the same reference temperature as the class.
        T_ref = self.T_ref

        # ======================================================
        # GRID / GEOMETRY
        # ======================================================

        N = len(Tg)

        A_cross = self.A_cross
        dz = self.dz

        V_cell = self.V_cell

        # ======================================================
        # MASS FLOW / THERMAL CAPACITY
        # ======================================================

        # These are calculated in apply()
        m_dot_g = state.m_dot_g
        m_dot_s = state.m_dot_s

        Cp_g = self.Cp_g
        Cp_s = self.Cp_s

        # Enthalpy flow capacities [W/K]
        Cg = m_dot_g * Cp_g
        Cs = m_dot_s * Cp_s

        # ======================================================
        # GAS-SOLID HEAT TRANSFER
        # ======================================================

        K = self.hv_gs * self.a_gs

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
        # TEMPORARY TEST
        # ======================================================

        # For the first steady-state closure test,
        # combustion heat is NOT distributed into cells yet.

        q_vol = np.zeros(N - 1)

        # ======================================================
        # WALL TRANSFER
        # ======================================================

        # Temporarily disabled.
        q_gw = np.zeros(N - 1)
        q_ws = np.zeros(N - 1)

        # ======================================================
        # BOUNDARY CONDITIONS
        # ======================================================

        # Gas:
        # inlet  -> N-1
        # outlet -> 0

        Tg_in = Tg[-1]

        # Solid:
        # inlet  -> 0
        # outlet -> N-1

        Ts_in = Ts[0]

        # ======================================================
        # INLET ENTHALPIES
        # ======================================================

        Hg_in = (
            m_dot_g
            * Cp_g
            * (Tg_in - T_ref)
        )

        Hs_in = (
            m_dot_s
            * Cp_s
            * (Ts_in - T_ref)
        )

        # ======================================================
        # STEADY-STATE LINEAR SYSTEM
        # ======================================================

        #
        # x =
        #
        # [ Tg[0], ..., Tg[N-1],
        #   Ts[0], ..., Ts[N-1] ]
        #

        A = np.zeros(
            (2 * N, 2 * N)
        )

        b = np.zeros(2 * N)

        row = 0

        # ======================================================
        # GAS INLET BOUNDARY
        # ======================================================

        A[row, N - 1] = 1.0
        b[row] = Tg_in

        row += 1

        # ======================================================
        # SOLID INLET BOUNDARY
        # ======================================================

        A[row, N] = 1.0
        b[row] = Ts_in

        row += 1

        # ======================================================
        # CELL EQUATIONS
        # ======================================================

        for j in range(N - 1):

            # --------------------------------------------------
            # GLOBAL INDICES
            # --------------------------------------------------

            Tg_j = j
            Tg_j1 = j + 1

            Ts_j = N + j
            Ts_j1 = N + j + 1

            # --------------------------------------------------
            # GAS ENERGY BALANCE
            # --------------------------------------------------

            #
            # Cg (Tg_j - Tg_j1)
            #
            #       = - V_cell q_gs,j
            #
            # q_gs,j =
            #
            # K/2 *
            # [
            #   Tg_j + Tg_j1
            #   - Ts_j - Ts_j1
            # ]
            #

            A[row, Tg_j] += (
                Cg
                + 0.5 * V_cell * K
            )

            A[row, Tg_j1] += (
                -Cg
                + 0.5 * V_cell * K
            )

            A[row, Ts_j] += (
                -0.5 * V_cell * K
            )

            A[row, Ts_j1] += (
                -0.5 * V_cell * K
            )

            b[row] = (
                -V_cell * q_vol[j]
            )

            row += 1

            # --------------------------------------------------
            # SOLID ENERGY BALANCE
            # --------------------------------------------------

            #
            # Cs (Ts_j1 - Ts_j)
            #
            #       = + V_cell q_gs,j
            #

            A[row, Ts_j1] += (
                Cs
                + 0.5 * V_cell * K
            )

            A[row, Ts_j] += (
                -Cs
                + 0.5 * V_cell * K
            )

            A[row, Tg_j] += (
                -0.5 * V_cell * K
            )

            A[row, Tg_j1] += (
                -0.5 * V_cell * K
            )

            b[row] = (
                V_cell * q_vol[j]
            )

            row += 1

        # ======================================================
        # SOLVE
        # ======================================================

        x = np.linalg.solve(A, b)

        Tg_ss = x[:N]
        Ts_ss = x[N:]

        # ======================================================
        # CELL CENTER TEMPERATURES
        # ======================================================

        Tg_cell = (
            Tg_ss[:-1]
            + Tg_ss[1:]
        ) * 0.5

        Ts_cell = (
            Ts_ss[:-1]
            + Ts_ss[1:]
        ) * 0.5

        # ======================================================
        # GAS-SOLID HEAT TRANSFER
        # ======================================================

        q_gs_cell = (
            K
            * (Tg_cell - Ts_cell)
        )

        Q_gs = (
            V_cell
            * np.sum(q_gs_cell)
        )

        # ======================================================
        # OUTLET TEMPERATURES
        # ======================================================

        Tg_out = Tg_ss[0]
        Ts_out = Ts_ss[-1]

        # ======================================================
        # OUTLET ENTHALPIES
        # ======================================================

        Hg_out = (
            m_dot_g
            * Cp_g
            * (Tg_out - T_ref)
        )

        Hs_out = (
            m_dot_s
            * Cp_s
            * (Ts_out - T_ref)
        )

        # ======================================================
        # ENERGY DIAGNOSTICS
        # ======================================================

        gas_energy_change = (
            Hg_out - Hg_in
        )

        solid_energy_change = (
            Hs_out - Hs_in
        )

        expected_gas_change = -Q_gs
        expected_solid_change = +Q_gs

        gas_energy_balance = (
            gas_energy_change
            - expected_gas_change
        )

        solid_energy_balance = (
            solid_energy_change
            - expected_solid_change
        )

        # ======================================================
        # WALL ENERGY
        # ======================================================

        Q_wall = (
            V_cell * np.sum(q_gw)
            + V_cell * np.sum(q_ws)
        )

        # ======================================================
        # TOTAL BURNING ZONE BALANCE
        # ======================================================

        total_energy_balance = (
            Hg_in
            + Hs_in
            + Q_burning
            - Hg_out
            - Hs_out
            - Q_wall
        )

        # ======================================================
        # DEBUG PRINT
        # ======================================================

        print("\n========== BURNING STEADY STATE ==========")

        print(f"Tg_in  = {Tg_in:.3f} K")
        print(f"Tg_out = {Tg_out:.3f} K")

        print(f"Ts_in  = {Ts_in:.3f} K")
        print(f"Ts_out = {Ts_out:.3f} K")

        print(f"Q_gs   = {Q_gs:.3f} W")

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

        print("===========================================\n")

        # ======================================================
        # RETURN
        # ======================================================

        return (
            Tg_ss,
            Ts_ss,
            Tw,
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

        if state.Tg_burning.shape != (5,):
            raise ValueError(
                f"Burning shape corrupted: {state.Tg_burning.shape}"
            )

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
            fuel_rate_total=inputs.get(
                "Fuel_rate_total",
                1.0,
            ),
            O2=inputs.get(
                "O2",
                3.5,
            ),
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
            * (
                state.Tg_burning
                - self.T_ref
            )
        )

        state.Hs_burning = (
            state.m_dot_s
            * self.Cp_s
            * (
                state.Ts_burning
                - self.T_ref
            )
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
