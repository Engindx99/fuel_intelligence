from physics.physics import (
    kiln_geometry,
    wall_geometry,
    interfacial_areas,
    solid_axial_velocity,
    gas_axial_velocity,
    fuel_heat_release,
    heat_transfer,
    wall_losses,
    thermal_capacities,)

from bdb import effective
import json
import pickle
import os
import numpy as np
import yaml

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class Burning:

    def __init__(self, N=5, L=60.0):

        cfg = load_cfg("configs/twin_cfg.yaml")

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
            self.A_wall_total,
            self.A_wall_cell,
            self.a_gw,
            self.V_wall,
        ) = wall_geometry(
            D=self.D,
            L=self.L,
            N=self.N,
            V_cell=self.V_cell,
        )

        # ================= MATERIAL PROPERTIES =================
        self.rho_g = 0.30
        self.rho_s = 1100.0
        self.rho_wall = 3000.0

        self.Cp_g = 1150.0
        self.Cp_s = 850.0
        self.Cp_wall = 1000.0

        # ================= KILN MOTION =================
        self.rpm_default = cfg["motion"]["rpm_default"]
        self.rpm_min = cfg["motion"]["rpm_min"]
        self.rpm_max = cfg["motion"]["rpm_max"]

        self.slope_deg = cfg["motion"]["inclination_deg"]
        self.fill_fraction = 0.10

        self.u_s = solid_axial_velocity(
            L=self.L,
            D=self.D,
            slope_deg=self.slope_deg,
            fill_fraction=self.fill_fraction,
            rpm=self.rpm_default,
            eps=self.eps,
        )
        self.u_g = 0.0

        self.u_g = gas_axial_velocity(
            m_dot_g=inputs["Gas_mass_flow"],
            rho_g=self.rho_g,
            A_cross=self.A_cross,
            eps=self.eps,
        )

        # ================= HEAT TRANSFER =================
        self.hv_gs = 1300.0
        self.hv_gw = 250.0
        self.hv_ws = 300.0

        self.T_amb = 300.0

        # ================= FUEL =================
        self.LHV = {
            "petcoke": 32e6,
            "coal": 18e6,
            "rdf": 20e6,
            "h2": 120e6,
        }

        self.O2_opt = 3.5
        self.O2_sigma2 = 25.0

        # ================= PERFORMANCE BUFFERS =================
        self._dTg_dz = np.zeros(N)
        self._dTs_dz = np.zeros(N)

        # ================= CACHE CONSTANTS =================

        self._rho_g_Vcell_Cp_g = (
            self.rho_g * self.V_cell * self.Cp_g
        )

        self._rho_s_Vcell_Cp_s = (
            self.rho_s * self.V_cell * self.Cp_s
        )

        self.V_wall_cell = self.V_wall / self.N

        self._rho_wall_Vwall_cell_Cp = (
            self.rho_wall
            * self.V_wall_cell
            * self.Cp_wall
        )

    # ======================================================
    def thermal_step(self, Tg, Ts, Tw, inputs, dt):

        # ================= GRADIENTS (NO ALLOC) =================
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
            A_wall_total=self.A_wall_total,
            N=self.N,
            eps=self.eps,
        )

        # ======================================================
        # THERMAL CAPACITIES
        # ======================================================
        C_g, effective_C_s, C_w = heat_capacities(
            rho_g_Vcell_Cp_g=self._rho_g_Vcell_Cp_g,
            rho_s_Vcell_Cp_s=self._rho_s_Vcell_Cp_s,
            rho_wall_Vwall_cell_Cp=self._rho_wall_Vwall_cell_Cp,
            effective=0.01,
        )

        # ======================================================
        # DYNAMICS
        # ======================================================
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
            raise ValueError(f"Burning shape corrupted: {state.Tg_burning.shape}")

        # ================= STORE OLD STATES =================
        state.Tg_burning_old = state.Tg_burning.copy()
        state.Ts_burning_old = state.Ts_burning.copy()
        state.Tw_burning_old = state.Tw_burning.copy()

        # ======================================================
        # SOLID MOTION UPDATE
        # ======================================================

        rpm = np.clip(
            inputs.get("rpm", self.rpm_default),
            self.rpm_min,
            self.rpm_max,
        )

        tau = self.residence_time(rpm)

        self.u_s = self.solid_velocity(rpm)

        # Save to global state
        state.rpm = rpm
        state.residence_time = tau
        state.solid_velocity = self.u_s

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
            inputs,
            dt,
        )

        # ================= UPDATE TEMPERATURE FIELDS =================
        state.Tg_burning = Tg
        state.Ts_burning = Ts
        state.Tw_burning = Tw

        # ================= FUEL ENERGY =================
        state.Q_petcoke = Q_petcoke
        state.Q_coal = Q_coal
        state.Q_RDF = Q_RDF
        state.Q_H2 = Q_H2
        state.Q_burning = Q_burning

        # ================= WALL LOSS =================
        state.Wall_loss_burning = float(wall_loss)

        # ================= WALL DEBUG =================
        state.wall_debug_burning = wall_debug or {
            "q_loss_mean": 0.0,
            "q_loss_total": 0.0,
            "A_wall": 0.0,
            "V_cell": 0.0,
            "N": 0,
        }

        state.q_loss_mean_burning = state.wall_debug_burning["q_loss_mean"]
        state.A_wall_burning = state.wall_debug_burning["A_wall"]
        state.V_cell_burning = state.wall_debug_burning["V_cell"]
        state.N_burning = state.wall_debug_burning["N"]

        # ======================================================
        # ENERGY TO NEXT ZONE
        # ======================================================

        # Gas
        state.Hgas_burning_out = self.gas_enthalpy_out(
            state.Tg_burning
        )

        # Solid
        state.Hsolid_burning_out = self.solid_enthalpy_out(
            state.Ts_burning
        )

        # ======================================================
        # STORED ENERGY
        # ======================================================

        # Gas
        state.Burning_gas_stored = np.sum(
            self._rho_g_Vcell_Cp_g
            * (state.Tg_burning - state.Tg_burning_old)
            / dt
        )

        # Solid
        state.Burning_solid_stored = np.sum(
            self._rho_s_Vcell_Cp_s
            * (state.Ts_burning - state.Ts_burning_old)
            / dt
        )

        # Wall
        state.Burning_wall_stored = np.sum(
            self._rho_wall_Vwall_cell_Cp
            * (state.Tw_burning - state.Tw_burning_old)
            / dt
        )

        # Total
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
    def gas_enthalpy_out(self, Tg):

        m_dot_g = self.rho_g * self.u_g * self.A_cross

        H_gas_out = m_dot_g * self.Cp_g * Tg[-1]

        return H_gas_out
    
    # ======================================================
    def solid_enthalpy_out(self, Ts):

        m_dot_s = self.rho_s * self.u_s * self.A_cross

        H_solid_out = m_dot_s * self.Cp_s * Ts[-1]

        return H_solid_out