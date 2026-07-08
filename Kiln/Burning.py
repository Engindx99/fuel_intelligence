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
        
    # ========================================================
    def thermal_step(self, Tg, Ts, Tw, state, inputs, dt, u_g, u_s):

        # ======================================================
        # GRADIENTS
        # ======================================================
        dTg_dz = self._dTg_dz
        dTs_dz = self._dTs_dz

        dTg_dz[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTs_dz[1:] = (Ts[1:] - Ts[:-1]) / self.dz

        dTg_dz[0] = dTg_dz[1]
        dTs_dz[0] = dTs_dz[1]
        

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

        # ======================================================
        # HEAT SOURCE
        # ======================================================
        q_vol = Q_burning / (self.V_total + self.eps)
        

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
            A_wall_total=self.A_wall_total,
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
            Q_petcoke,
            Q_coal,
            Q_RDF,
            Q_H2,
            Q_burning,
            wall_loss,
            wall_debug,
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
        # STORE OLD STATES
        # ======================================================
        state.Tg_burning_old = state.Tg_burning.copy()
        state.Ts_burning_old = state.Ts_burning.copy()
        state.Tw_burning_old = state.Tw_burning.copy()

        state.Hg_burning_old = state.Hg_burning.copy()
        state.Hs_burning_old = state.Hs_burning.copy()

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
        # GAS MASS FLOW + VELOCITY
        # ======================================================
        m_dot_g = gas_mass_balance(
            fuel_rate_total=inputs.get("Fuel_rate_total", 1.0),
            O2=inputs.get("O2", 3.5),
            eps=self.eps,
        )

        state.m_dot_g = float(m_dot_g)

        rho_g = getattr(self, "rho_g_avg", None)
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
        # THERMAL STEP
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
            wall_loss,
            wall_debug,
        ) = self.thermal_step(
            state.Tg_burning,
            state.Ts_burning,
            state.Tw_burning,
            state,
            inputs,
            dt,
            u_g,
            u_s,
        )

        # ======================================================
        # UPDATE STATES
        # ======================================================
        state.Tg_burning = Tg
        state.Ts_burning = Ts
        state.Tw_burning = Tw
        
        # ======================================================
        # BURNING CHEMISTRY
        # ======================================================
        
        state = self.chemistry.apply_burning(state)
        
        
        # ======================================================
        # UPDATE ENTHALPY STATES
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
        
        #print("\n========== BURNING ENTHALPY ==========")
        #print(f"Q_burning       : {state.Q_burning:.2f} W")
        #print(f"Hg_total        : {np.sum(state.Hg_burning):.2f} W")
        #print(f"Hs_total        : {np.sum(state.Hs_burning):.2f} W")
        

        state.Q_petcoke = Q_petcoke
        state.Q_coal = Q_coal
        state.Q_RDF = Q_RDF
        state.Q_H2 = Q_H2
        state.Q_burning = Q_burning

        state.Wall_loss_burning = float(wall_loss)

        # ======================================================
        # WALL DEBUG
        # ======================================================
        if wall_debug is None:
            wall_debug = {}

        state.wall_debug_burning = {
            "q_loss_mean": wall_debug.get("q_loss_mean", 0.0),
            "q_loss_total": wall_debug.get("wall_loss_total", 0.0),
            "A_wall": wall_debug.get("A_wall", 0.0),
            "V_cell": wall_debug.get("V_cell", 0.0),
            "N": wall_debug.get("N", 0),
        }

        state.q_loss_mean_burning = state.wall_debug_burning["q_loss_mean"]
        state.A_wall_burning = state.wall_debug_burning["A_wall"]
        state.V_cell_burning = state.wall_debug_burning["V_cell"]
        state.N_burning = state.wall_debug_burning["N"]

        # ======================================================
        # ENERGY OUT
        # ======================================================

        state.Hgas_burning_out = self.gas_enthalpy_out(
            state.Hg_burning
        )

        state.Hsolid_burning_out = self.solid_enthalpy_out(
            state.Hs_burning
        )
        #print(f"Hgas_out        : {state.Hgas_burning_out:.2f} W")
        #print(f"Hsolid_out      : {state.Hsolid_burning_out:.2f} W")

        # ======================================================
        # STORED ENERGY
        # ======================================================
        state.Burning_gas_stored = np.sum(
            (state.Hg_burning - state.Hg_burning_old)
            / dt
        )

        state.Burning_solid_stored = np.sum(
            (state.Hs_burning - state.Hs_burning_old)
            / dt
        )

        state.Burning_wall_stored = np.sum(
            self._rho_wall_Vwall_cell_Cp * (state.Tw_burning - state.Tw_burning_old) / dt
        )

        state.Burning_stored_energy_change = (
            state.Burning_gas_stored
            + state.Burning_solid_stored
            + state.Burning_wall_stored
        )

        # ======================================================
        # ENERGY BALANCE
        # ======================================================
        state.Burning_energy_balance = (
            state.Q_burning
            - state.Hgas_burning_out
            - state.Hsolid_burning_out
            - state.Burning_stored_energy_change
            - state.Wall_loss_burning
        )
    

        return state
    
    # ======================================================
    # GAS ENTHALPY TO NEXT ZONE
    # ======================================================
    def gas_enthalpy_out(self, Hg):

        return Hg[-1]



    # ======================================================
    # SOLID ENTHALPY TO NEXT ZONE
    # ======================================================
    def solid_enthalpy_out(self, Hs):

        return Hs[-1]
