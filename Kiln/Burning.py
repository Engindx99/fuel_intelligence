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
        fuel_rate_total = inputs.get("Fuel_rate_total", 1.0)
        O2 = inputs.get("O2", 3.5)

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

        q_vol = Q_burning / (self.V_total + self.eps)
        
        if True:
            print("\n===== BURNING GEOMETRY DEBUG =====")
            print("N               =", self.N)
            print("L               =", self.L)
            print("dz              =", self.dz)
            print("L/N             =", self.L / self.N)
            print("A_cross         =", self.A_cross)
            print("V_total         =", self.V_total)
            print("A_cross * L     =", self.A_cross * self.L)

            print("q_vol           =", q_vol)
            print("q_vol * V_total =", q_vol * self.V_total)
            print(
                "q_vol*A*dz*N    =",
                q_vol * self.A_cross * self.dz * self.N
            )

        # ======================================================
        # MASS FLOW
        # ======================================================
        m_dot_g = max(float(state.m_dot_g), self.eps)
        m_dot_s = max(float(state.m_dot_s), self.eps)

        # ======================================================
        # BOUNDARY CONDITIONS
        # ======================================================
        Tg = np.asarray(Tg, dtype=float).copy()
        Ts = np.asarray(Ts, dtype=float).copy()
        Tw = np.asarray(Tw, dtype=float).copy()

        Hg_in = (
            m_dot_g
            * self.Cp_g
            * (Tg[-1] - self.T_ref)
        )

        Hs_in = (
            m_dot_s
            * self.Cp_s
            * (Ts[0] - self.T_ref)
        )

        # ======================================================
        # INITIAL GUESS
        # ======================================================
        Tg_ss = Tg.copy()
        Ts_ss = Ts.copy()
        Tw_ss = Tw.copy()

        # ======================================================
        # ITERATION
        # ======================================================
        max_iter = 1
        tolerance = 1e-5
        relaxation = 0.005

        error = np.inf

        for iteration in range(max_iter):

            if iteration < 5:
                print(f"\n######## ITERATION {iteration} ########")

            Tg_old = Tg_ss.copy()
            Ts_old = Ts_ss.copy()

            # ==================================================
            # HEAT TRANSFER
            # ==================================================
            q_gs = (
                self.hv_gs
                * self.a_gs
                * (Tg_ss - Ts_ss)
            )

            q_gw = np.zeros(self.N)
            q_ws = np.zeros(self.N)

            # ==================================================
            # GAS ENERGY SOURCE
            # ==================================================
            gas_source = (
                q_vol
                - q_gs
                - q_gw
            )

            # ==================================================
            # SOLID ENERGY SOURCE
            # ==================================================
            solid_source = (
                q_gs
                - q_ws
            )
            
            
            if iteration == 0:
                print("\n===== BURNING ENERGY DEBUG =====")
                print("Q_burning       =", Q_burning)
                print("q_vol           =", q_vol)
                print("q_gs min/max    =", np.min(q_gs), np.max(q_gs))
                print("gas_source min/max =", np.min(gas_source), np.max(gas_source))
                print("solid_source min/max =", np.min(solid_source), np.max(solid_source))
                print("A_cross         =", self.A_cross)
                print("dz              =", self.dz)
                print("cell gas energy =", self.A_cross * self.dz * np.max(np.abs(gas_source)))
                print("cell solid energy =", self.A_cross * self.dz * np.max(np.abs(solid_source)))
                print("Hg_in           =", Hg_in)
                print("Hs_in           =", Hs_in)

            # ==================================================
            # GAS ENTHALPY
            # ==================================================
            Hg_new = np.zeros(self.N, dtype=float)

            Hg_new[-1] = Hg_in

            for j in range(self.N - 2, -1, -1):

                Hg_new[j] = (
                    Hg_new[j + 1]
                    - self.A_cross
                    * self.dz
                    * gas_source[j + 1]
                )

            # ==================================================
            # GAS TEMPERATURE
            # ==================================================
            Tg_new = (
                self.T_ref
                + Hg_new
                / (
                    m_dot_g * self.Cp_g
                    + self.eps
                )
            )

            # ==================================================
            # SOLID ENTHALPY
            # ==================================================
            Hs_new = np.zeros(self.N, dtype=float)

            Hs_new[0] = Hs_in

            for j in range(1, self.N):

                Hs_new[j] = (
                    Hs_new[j - 1]
                    + self.A_cross
                    * self.dz
                    * solid_source[j - 1]
                )

            # ==================================================
            # SOLID TEMPERATURE
            # ==================================================
            Ts_new = (
                self.T_ref
                + Hs_new
                / (
                    m_dot_s * self.Cp_s
                    + self.eps
                )
            )

            # ==================================================
            # RELAXATION
            # ==================================================
            Tg_ss = Tg_new.copy()
            Ts_ss = Ts_new.copy()

            # Wall is temporarily frozen
            Tw_ss = Tw.copy()

            # ==================================================
            # CONVERGENCE
            # ==================================================
            error_Tg = np.max(
                np.abs(Tg_ss - Tg_old)
            )

            error_Ts = np.max(
                np.abs(Ts_ss - Ts_old)
            )

            error = max(
                error_Tg,
                error_Ts,
            )

            if error < tolerance:
                break

        # ======================================================
        # FINAL HEAT TRANSFER
        # ======================================================
        q_gs = (
            self.hv_gs
            * self.a_gs
            * (Tg_ss - Ts_ss)
        )

        q_gw = np.zeros(self.N)
        q_ws = np.zeros(self.N)

        # ======================================================
        # NO WALL LOSS FOR THIS TEST
        # ======================================================
        wall_loss = 0.0
        wall_debug = None

        # ======================================================
        # OUTLET ENTHALPIES
        # ======================================================
        Hg_out = Hg_new[0]
        Hs_out = Hs_new[-1]

        # ======================================================
        # ENERGY BALANCE
        # ======================================================
        gas_energy_change = (
            Hg_out - Hg_in
        )

        solid_energy_change = (
            Hs_out - Hs_in
        )

        gas_source_integral = (
            self.A_cross
            * self.dz
            * np.sum(gas_source)
        )

        solid_source_integral = (
            self.A_cross
            * self.dz
            * np.sum(solid_source)
        )

        gas_energy_balance = (
            gas_energy_change
            + gas_source_integral
        )

        solid_energy_balance = (
            solid_energy_change
            - solid_source_integral
        )

        # ======================================================
        # TOTAL THERMAL BALANCE
        # ======================================================
        total_in = (
            Q_burning
            + Hg_in
            + Hs_in
        )

        total_out = (
            Hg_out
            + Hs_out
            + wall_loss
        )

        thermal_energy_balance = (
            total_in - total_out
        )

        # ======================================================
        # DEBUG STATE
        # ======================================================
        state.Burning_thermal_iterations = iteration + 1

        state.Burning_thermal_converged = (
            error < tolerance
        )

        state.Burning_thermal_residual = float(error)

        state.Burning_wall_iterations = 0
        state.Burning_wall_residual = 0.0

        state.Burning_gas_energy_balance = float(
            gas_energy_balance
        )

        state.Burning_solid_energy_balance = float(
            solid_energy_balance
        )

        state.Burning_thermal_energy_balance = float(
            thermal_energy_balance
        )

        state.Hg_burning_in = float(Hg_in)
        state.Hg_burning_out = float(Hg_out)

        state.Hs_burning_in = float(Hs_in)
        state.Hs_burning_out = float(Hs_out)

        # ======================================================
        # RETURN
        # ======================================================
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
