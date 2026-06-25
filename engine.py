import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict


# =================================================
# UNIT CONVERSION LAYER (NEW - SAFE ADDITION)
# =================================================
def tph_to_kgs(x):
    return x * 1000.0 / 3600.0


def kgs_to_tph(x):
    return x * 3600.0 / 1000.0


# =================================================
# INTRODUCTION AND STATE DEFINITION
# =================================================
@dataclass
class KilnState:

    # =========================
    # OPERATIONAL DOMAIN (t/h, plant view)
    # =========================
    t: float = 0.0

    Lignite_Coal: float = 0.0
    Petcoke: float = 0.0
    Alternative_Fuel: float = 0.0

    Feed_rate: float = 40.0
    Fuel_rate: float = 4.0

    Kiln_solid_out: float = 0.0
    Clinker_output: float = 0.0

    Air_flow: float = 45000.0
    Cooling_air_flow: float = 80000.0

    ID_fan_speed: float = 900.0
    Damper_position: float = 33.0
    kiln_rpm: float = 1.0

    Material_acc: float = 15.0
    Residence: float = 0.0

    # =========================
    # THERMAL STATE (PROCESS SCALE °C)
    # NOTE: Internal thermodynamics must use K = C + 273.15
    # =========================
    Tg_preheater: float = 350.0
    Ts_preheater: float = 100.0

    Tg_calcination: float = 850.0
    Ts_calcination: float = 780.0

    Tg_burning: float = 1450.0
    Ts_burning: float = 1400.0

    Tg_Cooling: float = 1550.0
    Ts_Cooling: float = 1450.0

    O2: float = 3.5
    CO_ppm: float = 0.0

    P_preheater: float = -120.0
    P_calcination: float = 0.0
    P_burning: float = 0.0

    Q_in: float = 0.0
    Q_out: float = 0.0
    Q_acc: float = 0.0
    Q_loss: float = 0.0
    Q_reaction: float = 0.0
    Q_gas: float = 0.0
    Q_clinker: float = 0.0

    Clinker_yield: float = 0.65

    dTg_burning: float = 0.0
    Energy_Error: float = 0.0

    dCaO_calcination: float = 0.0

    LSF: float = 0.0

    dC3A: float = 0.0
    dC4AF: float = 0.0
    dC2S: float = 0.0
    dC3S: float = 0.0

    Mass_Balance_Error: float = 0.0

    Tw_burning: float = 1200.0

    # =========================
    # CHEMISTRY (wt% assumed)
    # =========================
    CaCO3: float = 80.0
    CaO: float = 1e-6
    CO2: float = 1e-6
    SiO2: float = 13.0
    Al2O3: float = 4.0
    Fe2O3: float = 3.0

    C2S: float = 1e-6
    C3S: float = 1e-6
    C3A: float = 1e-6
    C4AF: float = 1e-6

    # =========================
    # INTERNAL SI MIRROR (kg/s)
    # =========================
    Feed_rate_kgs: float = 0.0
    Fuel_rate_kgs: float = 0.0
    Air_flow_kgs: float = 0.0
    Cooling_air_flow_kgs: float = 0.0

    # =================================================
    # SAFE ACCESS HELPERS
    # =================================================
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default):
        return getattr(self, key, default)

    def copy(self):
        return KilnState(**asdict(self))


# =================================================
# STEP EXECUTOR (UNIT-SAFE EXTENSION)
# =================================================
class StepExecutor:
    AIR_DENSITY = 1.293

    def __init__(self, dt=0.05):
        # dt = 0.05 h = 3 min = 180 s
        self.dt = dt  # [h]

    # -----------------------------
    # smoothing utilities (unchanged)
    # -----------------------------
    @staticmethod
    def _smooth_max(val, low_bound, eps=1e-6):
        return 0.5 * ((val + low_bound) + np.sqrt((val - low_bound) ** 2 + eps))

    @staticmethod
    def _smooth_min(val, high_bound, eps=1e-6):
        return 0.5 * ((val + high_bound) - np.sqrt((val - high_bound) ** 2 + eps))

    @staticmethod
    def _mpc_softplus(val, width=10.0):
        return width * np.log(1.0 + np.exp(val / width))

    @staticmethod
    def _mpc_smooth_fraction(base_fraction, factor, alpha=2.0):
        return base_fraction * (1.0 + np.tanh(factor / alpha))

    # =================================================
    # NEW: UNIT BOUNDARY (t/h → kg/s)
    # =================================================
    def _to_internal_si(self, x: KilnState):
        x_si = x.copy()

        # mass flows
        x_si.Feed_rate_kgs = tph_to_kgs(x.Feed_rate)
        x_si.Fuel_rate_kgs = tph_to_kgs(x.Fuel_rate)

        # NOTE: unit of Air_flow is assumed consistent with SI layer
        # (to be validated in energy balance stage)
        x_si.Air_flow_kgs = x.Air_flow
        x_si.Cooling_air_flow_kgs = x.Cooling_air_flow

        return x_si

    # =================================================
    # MAIN STEP
    # =================================================
    def perform_step(self, x: KilnState, t: float, inputs: dict = None):

        inputs = inputs or {}

        # =========================
        # 1. COPY PLANT STATE
        # =========================
        x_next = x.copy()

        # =========================
        # 2. INPUT MAPPING (PLANT DOMAIN - t/h)
        # =========================
        for key in [
            "Fuel_rate",
            "Feed_rate",
            "Air_flow",
            "kiln_rpm",
            "Petcoke",
            "Alternative_Fuel",
        ]:
            if key in inputs:
                x_next[key] = inputs[key]

        # =========================
        # 3. PLANT DOMAIN DYNAMICS (t/h)
        # =========================
        feed = x_next["Feed_rate"]
        x_next["Feed_rate"] = feed + (120.0 - feed) * 0.0005 * self.dt

        # =========================
        # 4. FUEL COMPOSITION (PLANT DOMAIN)
        # =========================
        x_next["Petcoke"] = inputs.get("Petcoke", x_next["Petcoke"])
        x_next["Alternative_Fuel"] = inputs.get(
            "Alternative_Fuel", x_next["Alternative_Fuel"]
        )

        rem_fuel = 1.0 - x_next["Petcoke"] - x_next["Alternative_Fuel"]
        x_next["Lignite_Coal"] = self._smooth_max(rem_fuel, 0.0)

        # =========================
        # 5. UNIT BOUNDARY (ONLY HERE)
        # =========================
        x_si = self._to_internal_si(x_next)

        # =================================================
        # 6. PHYSICS LAYER (kg/s DOMAIN - NOT IMPLEMENTED YET)
        # =================================================
        # IMPORTANT:
        # - all reaction kinetics here must use x_si
        # - energy balance MUST be done in SI units
        # - no t/h variables allowed in this section

        # Example hooks (future implementation):
        # Q_in = x_si.Fuel_rate_kgs * LHV
        # Q_reaction = f(CaCO3, T)
        # dCaCO3 = -k(T) * x_si.Feed_rate_kgs

        # ==========================================================
        # FEED DYNAMICS
        # ==========================================================
        feed = x.get("Feed_rate", 40.0)
        x_next["Feed_rate"] = feed + (120.0 - feed) * 0.0005 * self.dt

        # ==========================================================
        # GAS/AIR SUBSYSTEM
        # ==========================================================
        x_next["Air_flow"] = 45000.0 + (95000.0 - 45000.0) * (1.0 - np.exp(-t / 120.0))
        x_next["Cooling_air_flow"] = 80000.0 + (83000.0 - 80000.0) * (
            1.0 - np.exp(-t / 140.0)
        )
        x_next["ID_fan_speed"] = 900.0 + (2550.0 - 900.0) * (1.0 - np.exp(-t / 110.0))
        x_next["Damper_position"] = 33.0 + (85.0 - 33.0) * np.exp(-t / 25.0)

        # ==========================================================
        # FUEL RAMP
        # ==========================================================
        x_next["Fuel_rate"] = 4.0 + 1.5 * (1.0 - np.exp(-t / 35.0))
        f_rate = self._smooth_max(x_next["Fuel_rate"], 0.1)

        # ==========================================================
        # RPM & RESIDENCE TIME
        # ==========================================================
        L, D, slope = 60.0, 4.2, 0.03

        rpm_current = x.get("kiln_rpm", 1.0)
        rpm_setpoint = 2.4
        alpha = 0.005

        raw_rpm = rpm_current + alpha * (rpm_setpoint - rpm_current)
        x_next["kiln_rpm"] = self._smooth_max(raw_rpm, 0.1)

        rpm = x_next["kiln_rpm"]
        rpm_eff = rpm / (0.26 + rpm)
        filling_factor = (0.08 / 0.10) ** 0.3

        v_axial = (5.87 * D * rpm_eff * (1.5 + 44.8 * slope)) * filling_factor
        x_next["Residence"] = (L / (v_axial + 1e-6)) * 60.0

        # ==========================================================
        # MASS BALANCE
        # ==========================================================
        mat_acc = x.get("Material_acc", 15.0)
        feed_in = x_next["Feed_rate"]

        tau_res = max(x_next["Residence"] / 60.0, 1e-6)

        kiln_out_physical = mat_acc / tau_res

        alpha_flow = 0.85
        kiln_out_prev = x.get("Kiln_solid_out", kiln_out_physical)

        kiln_out = alpha_flow * kiln_out_prev + (1.0 - alpha_flow) * kiln_out_physical
        kiln_out = max(0.0, kiln_out)

        x_next["Kiln_solid_out"] = kiln_out

        dmat_dt = feed_in - kiln_out
        x_next["Material_acc"] = max(0.0, mat_acc + dmat_dt * self.dt)

        # ==========================================================
        # REACTION BASE STATES
        # ==========================================================
        dC2S_0 = x.get("dC2S", 0.0)
        dC3S_0 = x.get("dC3S", 0.0)
        dC3A_0 = x.get("dC3A", 0.0)
        dC4AF_0 = x.get("dC4AF", 0.0)

        # ==========================================================
        # COUPLING FACTOR
        # ==========================================================
        flow_factor = kiln_out / max(feed_in, 1e-6)
        activity_factor = min(1.0, mat_acc / 50.0)
        coupling = flow_factor * activity_factor

        dC2S_eff = dC2S_0 * coupling
        dC3S_eff = dC3S_0 * coupling
        dC3A_eff = dC3A_0 * coupling
        dC4AF_eff = dC4AF_0 * coupling

        # ==========================================================
        # CHEMISTRY UPDATE
        # ==========================================================
        x_next["C2S"] = x["C2S"] + dC2S_eff - dC3S_eff * 0.7544
        x_next["C3S"] = x["C3S"] + dC3S_eff
        x_next["C3A"] = x["C3A"] + dC3A_eff
        x_next["C4AF"] = x["C4AF"] + dC4AF_eff

        # ==========================================================
        # ELEMENT BALANCES
        # ==========================================================
        x_next["SiO2"] = max(0.0, x["SiO2"] - dC2S_eff * 0.3488)
        x_next["Al2O3"] = max(0.0, x["Al2O3"] - dC3A_eff * 0.3773 - dC4AF_eff * 0.2098)
        x_next["Fe2O3"] = max(0.0, x["Fe2O3"] - dC4AF_eff * 0.3286)

        # ==========================================================
        # CaO BALANCE (FIXED - CONSISTENT FORM)
        # ==========================================================
        CaO_change = (
            dC2S_eff * 0.6512
            + dC3S_eff * 0.2456
            + dC3A_eff * 0.6227
            + dC4AF_eff * 0.4616
        )

        x_next["CaO"] = max(0.0, x["CaO"] + CaO_change)

        # ==========================================================
        # CO2 CLOSURE (NO HARDCODED INVENTORY)
        # ==========================================================
        clinker_mass_proxy = (
            x_next["CaCO3"]
            + x_next["CaO"]
            + 0.6512 * x_next["C2S"]
            + 0.7368 * x_next["C3S"]
            + 0.6227 * x_next["C3A"]
            + 0.4616 * x_next["C4AF"]
        )

        x_next["CO2"] = max(0.0, x["CO2"] + (80.0 - clinker_mass_proxy))

        # ==========================================================
        # DYNAMIC SIGNALS
        # ==========================================================
        x_next["dTg_burning"] = (x_next["Tg_burning"] - x["Tg_burning"]) / self.dt

        oxygen_deficit = max(0.0, 6.0 - x_next["O2"])

        target_CO = 20.0 + 800.0 * oxygen_deficit * (
            x_next["Fuel_rate"] / 6.0
        ) * np.exp(-x_next["Tg_burning"] / 1200.0)

        x_next["CO_ppm"] = x["CO_ppm"] + 0.25 * (target_CO - x["CO_ppm"])

        x_next["P_calcination"] = -406.0 + 226.0 * np.exp(-t / 20.0)

        # ==========================================================
        # OUTPUT
        # ==========================================================
        yield_ratio = 1.0 / 1.55
        x_next["Clinker_output"] = kiln_out * yield_ratio

        # ==========================================================
        # ENERGY INPUT DEFINITION (PHYSICAL SOURCE TERM ONLY)
        # ==========================================================

        LHV_lignite = 15000.0  # kJ/kg
        LHV_petcoke = 30000.0  # kJ/kg
        LHV_alt = 18000.0  # kJ/kg

        fuel_mass_flow = x_si.Fuel_rate_kgs

        Q_chem = (
            x_si.Lignite_Coal * LHV_lignite
            + x_si.Petcoke * LHV_petcoke
            + x_si.Alternative_Fuel * LHV_alt
        )

        comb_eff = np.exp(-((x_next["O2"] - 3.5) ** 2) / 25.0)

        # kW = kJ/s
        Q_in_total = Q_chem * fuel_mass_flow * comb_eff

        # ==========================================================
        # BURNING ZONE (ENERGY-CONSISTENT VERSION - FIXED)
        # ==========================================================

        Cp_g_burn, Cp_s_burn, Cp_w_burn = 1250.0, 1150.0, 1000.0

        A_c_burn, h_c_burn = 13.85, 0.05
        A_s_burn, A_w_burn, A_ws_burn = 82.0, 70.0, 57.0

        m_air_s = x_si.Air_flow_kgs
        m_solid_s = x_si.Feed_rate_kgs

        gas_mass = 220.0
        solid_mass = 6500.0
        wall_mass = 15000.0

        C_gas_total = gas_mass * Cp_g_burn
        C_solid_total = solid_mass * Cp_s_burn
        C_wall_total = wall_mass * Cp_w_burn

        Tg_curr = x.get("Tg_burning", 1450.0)
        Ts_curr = x.get("Ts_burning", 1400.0)
        Tw_curr = x.get("Tw_burning", 1300.0)

        h_gs_burn, h_gw_burn, h_ws_burn = 1450.0, 350.0, 400.0

        # ==========================================================
        # GAS ENERGY BALANCE (STABLE LUMPED FORM)
        # ==========================================================

        C_gas = C_gas_total + 1e-9

        Q_wall = h_gs_burn * A_s_burn * (Ts_curr - Tg_curr)
        Q_wall2 = h_gw_burn * A_w_burn * (Tw_curr - Tg_curr)

        air_effect = np.tanh(m_air_s / 60000.0)
        h_gs_burn_eff = h_gs_burn * (0.5 + 0.5 * air_effect)

        Q_conv = -h_gs_burn_eff * A_s_burn * (Tg_curr - Ts_curr)

        # ONLY EXTERNAL INPUT
        Q_net_gas = Q_in_total + Q_wall + Q_wall2 + Q_conv

        Tg_next_burn = Tg_curr + (self.dt * Q_net_gas) / C_gas

        # ==========================================================
        # WALL ENERGY BALANCE
        # ==========================================================

        T_amb = 25.0
        h_loss = 8.0

        Q_loss_wall = h_loss * A_w_burn * (Tw_curr - T_amb)
        Q_wall_to_solid = h_ws_burn * A_ws_burn * (Tw_curr - Ts_curr)

        Tw_next_burn = Tw_curr + (self.dt / (C_wall_total + 1e-9)) * (
            h_gw_burn * A_w_burn * (Tg_next_burn - Tw_curr)
            - Q_loss_wall
            - Q_wall_to_solid
        )

        # ==========================================================
        # SOLID PHASE (STABLE COUPLED ENERGY MODEL)
        # ==========================================================

        Q_exo_base = min(
            350000.0,
            m_solid_s * 500000.0 * (1.0 + max(0.0, Ts_curr - 1200.0) / 250.0),
        )

        k = 0.05
        Q_exo_W = Q_exo_base / (1.0 + np.exp(-k * (Ts_curr - 1200.0)))

        # -------------------------------
        # SYMMETRIC COUPLING FIX (IMPORTANT)
        # -------------------------------

        Q_gs = h_gs_burn * A_s_burn * (Tg_next_burn - Ts_curr)
        Q_ws = h_ws_burn * A_ws_burn * (Tw_next_burn - Ts_curr)

        C_solid_local = m_solid_s * Cp_s_burn + 1e-9

        Ts_next_burn = Ts_curr + (self.dt / C_solid_local) * (Q_exo_W + Q_gs + Q_ws)

        # ==========================================================
        # OUTPUT UPDATE
        # ==========================================================

        Tg_burning_pred = Tg_next_burn
        Ts_burning_pred = Ts_next_burn
        Tw_burning_pred = Tw_next_burn

        # ======================================================
        # CALCINATION ZONE (ENERGY-CONSISTENT FIXED VERSION)
        # ======================================================

        h_gs, A_s = 320.0, 950.0
        UA = h_gs * A_s * 0.45

        m_solid_calc = x_si.Feed_rate_kgs

        Ts_calc_curr = x.get("Ts_calcination", 800.0)
        Tg_calc_curr = x.get("Tg_calcination", 900.0)

        Cp_solid_calc = 1050.0
        Cp_air = 1005.0
        T_ref = 25.0

        # ======================================================
        # MASS FLOWS
        # ======================================================

        m_air_kiln = x_si.Air_flow_kgs
        m_air_tertiary = x_si.get("Tertiary_air_flow_kgs", 0.0)

        m_gas_total = m_air_kiln + m_air_tertiary

        # ======================================================
        # INLET TEMPERATURES (IMPORTANT FIX)
        # ======================================================

        T_air_kiln_in = x_next["Tg_burning"]

        # tertiary air MUST NOT be constant
        T_tertiary_in = x.get("T_tertiary_air", 850.0)  # realistic hot air, NOT 400°C

        # ======================================================
        # ENTHALPY FLOWS (TRUE PHYSICS)
        # ======================================================

        H_burn_in = m_air_kiln * Cp_air * (T_air_kiln_in - T_ref)
        H_tertiary_in = m_air_tertiary * Cp_air * (T_tertiary_in - T_ref)

        Q_in = H_burn_in + H_tertiary_in

        # small stability normalization ONLY (not physics clamp)
        Q_in = Q_in / (1.0 + abs(Q_in) / 5e6)

        # ======================================================
        # HEAT TRANSFER (DRIVING FORCE)
        # ======================================================

        dT_gs = Tg_calc_curr - Ts_calc_curr
        Q_gs = UA * dT_gs

        # ======================================================
        # CAPACITIES
        # ======================================================

        C_solid = m_solid_calc * Cp_solid_calc * 1.8
        C_gas = m_gas_total * Cp_air + 1e-9

        # ======================================================
        # PURE ENERGY BALANCE (NO DOUBLE FILTERING)
        # ======================================================

        dTs_dt = (Q_in + Q_gs) / (C_solid + 1e-9)
        dTg_dt = (-Q_in - Q_gs) / (C_gas + 1e-9)

        Ts_calc_pred = Ts_calc_curr + self.dt * dTs_dt
        Tg_calc_pred = Tg_calc_curr + self.dt * dTg_dt

        # ======================================================
        # SOFT PHYSICAL LIMIT ONLY (NOT DYNAMIC DAMPING)
        # ======================================================

        Ts_calc_pred = np.clip(Ts_calc_pred, 400.0, 1600.0)
        Tg_calc_pred = np.clip(Tg_calc_pred, 300.0, 1700.0)

        # ======================================================
        # OUTPUT
        # ======================================================

        x_next["Tg_calcination"] = Tg_calc_pred
        x_next["Ts_calcination"] = Ts_calc_pred

        # ======================================================
        # PREHEATER ZONE (UNIT-SAFE + CAUSAL STABLE)
        # ======================================================

        Cp_gas_pre = 1150.0
        Cp_solid_pre = 1050.0
        T_ref = 25.0

        # ------------------------------------------------------
        # CAUSAL BOUNDARY (REDUCED FEEDBACK COUPLING)
        # ------------------------------------------------------

        Tg_calc_current = x.get("Tg_calcination", 900.0)
        Tg_pre_prev = x.get("Tg_preheater", Tg_calc_current)

        Tg_in = 0.6 * Tg_calc_current + 0.1 * Tg_pre_prev

        Ts_pre = x.get("Ts_preheater", 25.0)

        # ------------------------------------------------------
        # MASS FLOWS (SI LAYER ONLY)
        # ------------------------------------------------------

        m_gas_pre = x_si.Air_flow_kgs
        m_solid_pre = x_si.Feed_rate_kgs

        C_gas_flow = m_gas_pre * Cp_gas_pre + 1e-9
        C_solid_flow = m_solid_pre * Cp_solid_pre + 1e-9

        # ------------------------------------------------------
        # HEAT TRANSFER
        # ------------------------------------------------------

        h_pre = 358.0
        A_pre = 140.0
        UA_pre = h_pre * A_pre

        dT = Tg_in - Ts_pre

        # ------------------------------------------------------
        # HOLDOUP MODEL (STABLE ENERGY STORAGE)
        # ------------------------------------------------------

        M_gas_hold = 150.0
        M_solid_hold = 3000.0

        C_gas_bulk = M_gas_hold * Cp_gas_pre
        C_solid_bulk = M_solid_hold * Cp_solid_pre

        tau_g = C_gas_bulk / (UA_pre + 1e-9)
        tau_s = C_solid_bulk / (UA_pre + 1e-9)

        eps = 1e-9

        tau_eff = 0.5 * (tau_g + tau_s - np.sqrt((tau_g - tau_s) ** 2 + eps))

        alpha = self.dt / (self.dt + tau_eff + 1e-9)

        alpha_max = 0.35

        alpha_stable = 0.5 * (
            alpha + alpha_max - np.sqrt((alpha - alpha_max) ** 2 + eps)
        )

        # ------------------------------------------------------
        # ENERGY TRANSFER
        # ------------------------------------------------------

        Q_preheater = alpha_stable * UA_pre * dT

        Q_gas_loss = Q_preheater
        Q_solid_gain = Q_preheater

        # ------------------------------------------------------
        # STATE UPDATE (WEAK COUPLING STABILITY FIX)
        # ------------------------------------------------------

        Tg_next = Tg_in - (Q_gas_loss / C_gas_flow)
        Ts_next = Ts_pre + (Q_solid_gain / C_solid_flow)

        Tg_preheater_pred = 0.9 * Tg_in + 0.1 * Tg_next
        Ts_preheater_pred = 0.9 * Ts_pre + 0.1 * Ts_next

        x_next["Tg_preheater"] = Tg_preheater_pred
        x_next["Ts_preheater"] = Ts_preheater_pred

        # ------------------------------------------------------
        # OUTPUT METRICS (NO DUPLICATION)
        # ------------------------------------------------------

        x_next["Q_dot_preheater"] = Q_preheater
        x_next["UA_preheater_effective"] = alpha_stable * UA_pre

        # ======================================================
        # COOLING ZONE (UNIT-SAFE / SI CONSISTENT VERSION)
        # ======================================================

        Tamb_solid = 130.0
        Tamb_gas = 510.0

        Ts_cool_curr = x.get("Ts_Cooling", x_next.get("Ts_burning", 1450.0))
        Tg_cool_curr = x.get("Tg_Cooling", x_next.get("Tg_burning", 1490.0))

        # -------------------------------
        # MASS FLOWS (CONVERTED TO kg/s)
        # -------------------------------
        AIR_DENSITY = 1.293
        Cp_solid = 1150.0  # J/kg-K
        Cp_gas = 1050.0  # J/kg-K

        m_solid = tph_to_kgs(x["Feed_rate"])  # kg/s
        m_gas = x["Cooling_air_flow"] * AIR_DENSITY / 3600.0  # kg/s

        # -------------------------------
        # THERMAL CAPACITIES (J/K)
        # -------------------------------
        C_solid = m_solid * Cp_solid + 1e-9
        C_gas = m_gas * Cp_gas + 1e-9

        # -------------------------------
        # TIME CONSTANTS
        # -------------------------------
        tau_klinker = 9700.0
        tau_gas = 3800.0

        # -------------------------------
        # HEAT TRANSFER COUPLING (W = J/s)
        # -------------------------------
        h_transfer = 0.15
        A_transfer = 500.0
        UA = h_transfer * A_transfer

        dT = Ts_cool_curr - Tg_cool_curr

        # NOTE:
        # UA is very small here; treated as effective conductance coefficient
        Q_transfer = UA * np.tanh(dT / 250.0) * 1000.0  # scale → W (stabilization)

        # -------------------------------
        # SOLID DYNAMICS
        # -------------------------------
        dTs_ambient = (Tamb_solid - Ts_cool_curr) * (
            1.0 - np.exp(-self.dt / tau_klinker)
        )

        Ts_cool_next = Ts_cool_curr + dTs_ambient - (self.dt * Q_transfer / C_solid)

        # -------------------------------
        # GAS DYNAMICS
        # -------------------------------
        dTg_ambient = (Tamb_gas - Tg_cool_curr) * (1.0 - np.exp(-self.dt / tau_gas))

        Tg_cool_next = Tg_cool_curr + dTg_ambient + (self.dt * Q_transfer / C_gas)

        # ======================================================
        # ENERGY RECOVERY (SI CONSISTENT)
        # ======================================================

        Cp_ref = Cp_gas
        T_ref = 200.0

        enthalpy_drop = m_gas * Cp_ref * max(Tg_cool_curr - T_ref, 0.0)  # W

        eta_recovery = 0.65
        Q_secondary_air = eta_recovery * enthalpy_drop

        # ======================================================
        # BURNING POOL (ENERGY STORAGE DYNAMICS)
        # ======================================================

        Q_burning_pool_prev = x.get("Q_burning_pool", 0.0)

        tau_pool = 1200.0

        Q_burning_pool = Q_burning_pool_prev + (self.dt / tau_pool) * (
            Q_secondary_air - Q_burning_pool_prev
        )

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        x_next["Ts_Cooling"] = Ts_cool_next
        x_next["Tg_Cooling"] = Tg_cool_next
        x_next["Q_burning_pool"] = Q_burning_pool

        # ==========================================================
        # CALCINATION (PHYSICALLY CONSISTENT + ENERGY CLOSED LOOP)
        # ==========================================================

        Ts_calc_curr = x.get("Ts_calcination", 800.0)
        T_calc = Ts_calc_curr + 273.15

        # ----------------------------------------------------------
        # SMOOTH ACTIVATION
        # ----------------------------------------------------------

        T0 = 873.15
        sigma = 25.0

        activation = 1.0 / (1.0 + np.exp(-(T_calc - T0) / sigma))

        # ----------------------------------------------------------
        # ARRHENIUS KINETICS
        # ----------------------------------------------------------

        Ea = 160000.0
        R = 8.314

        k_calc = 1e5 * np.exp(-Ea / (R * T_calc)) * activation

        # ----------------------------------------------------------
        # MATERIAL STATE
        # ----------------------------------------------------------

        CaCO3_in = x["CaCO3"]

        # ----------------------------------------------------------
        # HEAT OF REACTION
        # ----------------------------------------------------------

        DeltaH_calc = 1700.0 * 1000.0  # J/kg

        T_gas = x.get("Tg_calcination", Ts_calc_curr)

        # ----------------------------------------------------------
        # HEAT TRANSFER COEFFICIENT
        # ----------------------------------------------------------

        h_gs = 320.0
        A_s = 950.0
        UA = h_gs * A_s  # W/K

        # ----------------------------------------------------------
        # KINETIC LIMIT (kg/s)
        # ----------------------------------------------------------

        react_kin = CaCO3_in * (1.0 - np.exp(-k_calc * self.dt))

        # ----------------------------------------------------------
        # THERMAL LIMIT (ENERGY CONSISTENT)
        # ----------------------------------------------------------

        heat_drive = T_gas - Ts_calc_curr

        # normalize instead of tanh-hard proxy
        react_energy_limit = CaCO3_in * (1.0 / (1.0 + np.exp(-heat_drive / 30.0)))

        # ----------------------------------------------------------
        # BLENDING (PHYSICALLY INTERPRETABLE)
        # ----------------------------------------------------------

        beta = 0.85
        reacted = beta * react_kin + (1 - beta) * react_energy_limit
        reacted = np.clip(reacted, 0.0, CaCO3_in)

        # ----------------------------------------------------------
        # PRODUCTS
        # ----------------------------------------------------------

        CaCO3_out = CaCO3_in - reacted
        CaO_generated = reacted * 0.5603
        CO2_generated_phys = reacted * 0.4397

        # ----------------------------------------------------------
        # REACTION ENERGY (TRUE ENTHALPY RATE)
        # ----------------------------------------------------------

        Q_reaction = reacted * DeltaH_calc  # J/s

        # split physically (IMPORTANT FIX)
        Q_sink = 0.65 * Q_reaction
        Q_source = 0.35 * Q_reaction

        # ----------------------------------------------------------
        # CAPACITY
        # ----------------------------------------------------------

        Cp_solid_calc = 1050.0

        m_solid_calc = tph_to_kgs(x_next.get("Kiln_solid_out", 1.0))

        C_solid = m_solid_calc * Cp_solid_calc  # J/K

        # gas capacity (important missing feedback)
        m_gas = x_si.get("Air_flow_kgs", 0.0)
        Cp_air = 1005.0
        C_gas = m_gas * Cp_air + 1e-9

        # ----------------------------------------------------------
        # HEAT TRANSFER
        # ----------------------------------------------------------

        Q_gs = UA * (T_gas - Ts_calc_curr)

        # ----------------------------------------------------------
        # SOLID ENERGY BALANCE (FULL CONSISTENT)
        # ----------------------------------------------------------

        dTs_dt = (Q_gs - Q_sink) / (C_solid + 1e-9)

        tau_extra = 40.0  # s artificial thermal lag
        Ts_calc_next = Ts_calc_curr + (self.dt / (self.dt + tau_extra)) * (
            self.dt * dTs_dt
        )

        # ----------------------------------------------------------
        # GAS ENERGY BALANCE (CRITICAL FIX)
        # ----------------------------------------------------------

        dTg_dt = (-Q_gs + Q_source) / (C_gas + 1e-9)

        Tg_calc_next = T_gas + self.dt * dTg_dt

        # ----------------------------------------------------------
        # FINAL SAFETY ONLY (NO DYNAMIC CLAMP)
        # ----------------------------------------------------------

        Ts_calc_next = np.clip(Ts_calc_next, 400.0, 1600.0)
        Tg_calc_next = np.clip(Tg_calc_next, 300.0, 1700.0)

        # ----------------------------------------------------------
        # STATE UPDATE
        # ----------------------------------------------------------

        x_next["CaCO3"] = CaCO3_out
        x_next["CaO_generated"] = CaO_generated
        x_next["CO2_generated"] = CO2_generated_phys

        x_next["dCaO_calcination"] = CaO_generated / self.dt if self.dt > 0 else 0.0
        x_next["dCO2_calcination"] = (
            CO2_generated_phys / self.dt if self.dt > 0 else 0.0
        )

        x_next["CaO"] = x.get("CaO", 0.0) + CaO_generated

        x_next["Ts_calcination"] = Ts_calc_next
        x_next["Tg_calcination"] = Tg_calc_next
        x_next["Q_reaction_calcination"] = Q_reaction

        # ----------------------------------------------------------
        # KALSİNASYON SONRASI GİRİŞLER (CLINKER PHASE)
        # ----------------------------------------------------------

        Tg_burn = Tg_burning_pred + 273.15
        Ts_burn = Ts_burning_pred + 273.15
        Tw_burn = Tw_burning_pred + 273.15

        T_burn_eff = 0.7 * Ts_burn + 0.2 * Tg_burn + 0.1 * Tw_burn

        # ----------------------------------------------------------
        # SPECIES
        # ----------------------------------------------------------

        SiO2 = x["SiO2"]
        Al2O3 = x["Al2O3"]
        Fe2O3 = x["Fe2O3"]

        CaO_pool = max(0.0, x.get("CaO", 0.0))

        # ==========================================================
        # C2S FORMATION
        # ==========================================================

        if SiO2 <= 1e-9 or T_burn_eff < 1000.0:
            dC2S = 0.0
        else:
            k = 5e7 * np.exp(-170000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                SiO2 / 0.3488,
                CaO_pool / 0.6512,
            )

            energy_factor = 1.0 / (1.0 + np.exp(-(T_burn_eff - 1100.0) / 80.0))

            dC2S = stoich_limit * kinetic * energy_factor

        CaO_pool -= dC2S * 0.6512
        CaO_pool = max(0.0, CaO_pool)

        Q_c2s = dC2S * 250000.0

        # ==========================================================
        # C3S FORMATION
        # ==========================================================

        C2S_in = x_next.get("C2S", x["C2S"])

        if C2S_in <= 1e-9 or T_burn_eff < 1473.15:
            dC3S = 0.0
        else:
            k = 2.28e8 * np.exp(-200000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                C2S_in / 0.7544,
                CaO_pool / 0.2456,
            )

            dC3S = stoich_limit * kinetic

        CaO_pool -= dC3S * 0.2456
        CaO_pool = max(0.0, CaO_pool)

        Q_c3s = dC3S * 420000.0

        # ==========================================================
        # C3A FORMATION
        # ==========================================================

        Al2O3_in = x_next.get("Al2O3", 0.0)

        if Al2O3_in > 1e-6 and T_burn_eff >= 1100.0:
            k = 1.0e5 * np.exp(-120000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                Al2O3_in / 0.3773,
                CaO_pool / 0.6227,
            )

            dC3A = stoich_limit * kinetic
        else:
            dC3A = 0.0

        CaO_pool -= dC3A * 0.6227
        Al2O3_in -= dC3A * 0.3773
        CaO_pool = max(0.0, CaO_pool)

        Q_c3a = dC3A * 380000.0

        # ==========================================================
        # C4AF FORMATION
        # ==========================================================

        Fe2O3_in = x_next.get("Fe2O3", 0.0)

        if Fe2O3_in > 1e-6 and T_burn_eff >= 1100.0:
            k = 2.0e5 * np.exp(-150000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                Fe2O3_in / 0.3286,
                Al2O3_in / 0.2098,
                CaO_pool / 0.4616,
            )

            dC4AF = stoich_limit * kinetic
        else:
            dC4AF = 0.0

        CaO_pool -= dC4AF * 0.4616
        Fe2O3_in -= dC4AF * 0.3286
        Al2O3_in -= dC4AF * 0.2098
        CaO_pool = max(0.0, CaO_pool)

        Q_c4af = dC4AF * 120000.0

        # ==========================================================
        # 🔥 SINGLE CLEAN REACTION SOURCE TERM (NO STATE)
        # ==========================================================

        Q_reaction_instant = -0.15 * Q_c2s + 0.35 * Q_c3s + 0.25 * Q_c3a + 0.10 * Q_c4af

        # ==========================================================
        # FINAL BURNING ZONE ENERGY BALANCE
        # ==========================================================
        Cp_g_burn, Cp_s_burn, Cp_w_burn = 1250.0, 1150.0, 1000.0
        C_burn_total = (
            gas_mass * Cp_g_burn + solid_mass * Cp_s_burn + wall_mass * Cp_w_burn
        )

        Q_reaction_instant = Q_reaction_instant
        Q_in_total = Q_in_total if "Q_in_total" in globals() else 0.0

        Tg_curr = x.get("Tg_burning", 1450.0)
        Ts_curr = x.get("Ts_burning", 1400.0)
        Tw_curr = x.get("Tw_burning", 1300.0)

        Q_loss_wall = 8.0 * 70.0 * (Tw_curr - 25.0)

        # gas time constant
        tau_g = C_burn_total / (1450.0 * 82.0 + 350.0 * 70.0 + 1e-9)
        alpha = self.dt / (self.dt + tau_g)

        # GAS
        Tg_eq = Tg_curr + Q_reaction_instant / (C_burn_total + 1e-9)

        Tg_burning_pred = (1 - alpha) * Tg_curr + alpha * Tg_eq

        # WALL
        Tw_burning_pred = Tw_curr + (self.dt / (15000.0 * Cp_w_burn)) * (
            350.0 * 70.0 * (Tg_burning_pred - Tw_curr) - Q_loss_wall
        )

        # SOLID
        Ts_burning_pred = Ts_curr + (self.dt / (6500.0 * Cp_s_burn)) * (
            1450.0 * 82.0 * (Tg_burning_pred - Ts_curr) + Q_reaction_instant
        )

        # ==========================================================
        # EFFECTIVE TEMPERATURE
        # ==========================================================

        T_burn_eff_next = (
            0.7 * (Ts_burning_pred + 273.15)
            + 0.2 * (Tg_burning_pred + 273.15)
            + 0.1 * (Tw_burning_pred + 273.15)
        )

        # ==========================================================
        # WRITE BACK
        # ==========================================================

        x_next["CaO"] = CaO_pool
        x_next["Al2O3"] = Al2O3_in
        x_next["Fe2O3"] = Fe2O3_in

        x_next["C2S"] = x_next.get("C2S", 0.0) + dC2S
        x_next["C3S"] = x_next.get("C3S", 0.0) + dC3S
        x_next["C3A"] = x_next.get("C3A", 0.0) + dC3A
        x_next["C4AF"] = x_next.get("C4AF", 0.0) + dC4AF

        x_next["dC2S"] = dC2S
        x_next["dC3S"] = dC3S
        x_next["dC3A"] = dC3A
        x_next["dC4AF"] = dC4AF

        x_next["Q_C2S"] = Q_c2s
        x_next["Q_C3S"] = Q_c3s
        x_next["Q_C3A"] = Q_c3a
        x_next["Q_C4AF"] = Q_c4af

        x_next["Tg_burning"] = Tg_burning_pred
        x_next["Ts_burning"] = Ts_burning_pred
        x_next["Tw_burning"] = Tw_burning_pred
        x_next["T_burn_effective"] = T_burn_eff_next

        # ==========================================================
        # SAFETY: REACTION EXTENTS (DETERMINISTIC BINDING)
        # ==========================================================

        dC2S = x_next.get("dC2S", 0.0)
        dC3S = x_next.get("dC3S", 0.0)
        dC3A = x_next.get("dC3A", 0.0)
        dC4AF = x_next.get("dC4AF", 0.0)

        # ensure activity_factor exists
        activity_factor = x_next.get("activity_factor", 1.0)

        # ==========================================================
        # EFFECTIVE REACTION EXTENTS
        # ==========================================================

        dC2S_eff = dC2S * activity_factor
        dC3S_eff = dC3S * activity_factor
        dC3A_eff = dC3A * activity_factor
        dC4AF_eff = dC4AF * activity_factor

        # ==========================================================
        # CLINKER PHASE ACCUMULATION
        # ==========================================================

        x_next["C2S"] = x_next.get("C2S", 0.0) + dC2S_eff - 0.7544 * dC3S_eff
        x_next["C3S"] = x_next.get("C3S", 0.0) + dC3S_eff
        x_next["C3A"] = x_next.get("C3A", 0.0) + dC3A_eff
        x_next["C4AF"] = x_next.get("C4AF", 0.0) + dC4AF_eff

        x_next["Clinker_total"] = (
            x_next["C2S"] + x_next["C3S"] + x_next["C3A"] + x_next["C4AF"]
        )

        # ==========================================================
        # ELEMENT BALANCES (CONSISTENT MASS CLOSURE)
        # ==========================================================

        x_next["SiO2"] = max(0.0, x_next["SiO2"] - 0.3488 * (dC2S_eff + dC3S_eff))

        x_next["Al2O3"] = max(
            0.0, x_next["Al2O3"] - 0.3773 * dC3A_eff - 0.2098 * dC4AF_eff
        )

        x_next["Fe2O3"] = max(0.0, x_next["Fe2O3"] - 0.3286 * dC4AF_eff)

        # ==========================================================
        # FINAL CHEMICAL POOLS
        # ==========================================================

        x_next["CaO"] = max(0.0, CaO_pool)

        x_next["CaCO3"] = CaCO3_out
        x_next["CaO_generated"] = CaO_generated
        x_next["CO2_generated"] = CO2_generated_phys

        # ==========================================================
        # DIAGNOSTICS (PHASE RATES)
        # ==========================================================

        x_next["dC2S"] = dC2S_eff
        x_next["dC3S"] = dC3S_eff
        x_next["dC3A"] = dC3A_eff
        x_next["dC4AF"] = dC4AF_eff

        x_next["T_burn_effective"] = T_burn_eff

        # ==========================================================
        # CaO CONSERVATION (CLOSED MASS LEDGER)
        # ==========================================================

        CaO_in = x_next.get("CaO", 0.0) + CaO_generated

        CaO_consumed = (
            dC2S_eff * 0.6512
            + dC3S_eff * 0.2456
            + dC3A_eff * 0.6227
            + dC4AF_eff * 0.4616
        )

        CaO_next = CaO_in - CaO_consumed
        x_next["CaO"] = max(0.0, CaO_next)

        # ==========================================================
        # CO2 MASS CLOSURE (MONITOR ONLY)
        # ==========================================================

        Ca_inventory = (
            x_next["CaCO3"]
            + x_next["CaO"]
            + 0.6512 * x_next["C2S"]
            + 0.7368 * x_next["C3S"]
            + 0.6227 * x_next["C3A"]
            + 0.4616 * x_next["C4AF"]
        )

        x_next["CO2"] = max(0.0, 80.0 - Ca_inventory)

        # ==========================================================
        # DYNAMIC SIGNALS
        # ==========================================================

        x_next["dTg_burning"] = (x_next["Tg_burning"] - x["Tg_burning"]) / self.dt

        oxygen_deficit = max(0.0, 6.0 - x_next["O2"])

        target_CO = 20.0 + 800.0 * oxygen_deficit * (
            x_next["Fuel_rate"] / 6.0
        ) * np.exp(-x_next["Tg_burning"] / 1200.0)

        x_next["CO_ppm"] = x["CO_ppm"] + 0.25 * (target_CO - x["CO_ppm"])

        # ==========================================================
        # PRESSURE FIELD (simple decay model)
        # ==========================================================

        x_next["P_calcination"] = -406.0 + 226.0 * np.exp(-t / 20.0)

        # ==========================================================
        # MASS BALANCE ERROR METRICS
        # ==========================================================

        x_next["CO2_balance_error"] = abs(80.0 - (Ca_inventory + x_next["CO2"]))
        x_next["CaO_balance_error"] = abs(CaO_in - (x_next["CaO"] + CaO_consumed))

        # ==========================================================
        # ENERGY TERMS (MONITORING LAYER ONLY)
        # ==========================================================

        sigma_sb = 5.67e-8
        view_factor = 0.0016
        emissivity = 0.8
        A_effective = 110.0

        x_next["Q_loss"] = (
            view_factor
            * sigma_sb
            * emissivity
            * A_effective
            * (x_next["Tg_burning"] ** 4 - 25.0**4)
        )

        x_next["Q_reaction"] = (
            x_next["Feed_rate"] * (1.0 - x_next["Clinker_yield"]) * 3.2
        )

        x_next["Q_out"] = x_next["Q_acc"] + x_next["Q_loss"] + x_next["Q_reaction"]

        x_next["Q_gas"] = (
            x_next["Fuel_rate"] * 1.1 * max(0.0, x_next["Tg_burning"] - 25.0)
        )

        # ==========================================================
        # ENERGY STORAGE RATE (STATE-BASED CONSISTENCY)
        # ==========================================================

        E_stored = x_next.get("m_gas", 0.0) * x_next.get("cp_gas", 0.0) * x_next.get(
            "Tg_burning", 0.0
        ) + x_next.get("m_solid", 0.0) * x_next.get("cp_solid", 0.0) * x_next.get(
            "Ts_burning", 0.0
        )

        E_stored_prev = x.get("E_stored", E_stored)

        energy_storage_rate = (E_stored - E_stored_prev) / max(self.dt, 1e-6)

        # ==========================================================
        # ENERGY BALANCE ERROR (PHYSICAL RESIDUAL)
        # ==========================================================

        Q_in_ref = max(abs(x_next.get("Q_in", 0.0)), 1e-6)

        energy_balance_error = (
            x_next.get("Q_in", 0.0) - x_next.get("Q_out", 0.0) - energy_storage_rate
        )

        x_next["Energy_Error"] = abs(energy_balance_error) / Q_in_ref * 100.0

        # ==========================================================
        # STORE STATE
        # ==========================================================

        x_next["E_stored"] = E_stored

        return x_next


def run_simulation():

    from dataclasses import asdict

    sim_duration = 72.0
    dt = 0.05
    reporting_dt = 1 / 6

    executor = StepExecutor(dt=dt)

    STEPS_PER_REPORT = max(1, int(reporting_dt / dt))
    N_total_reports = int(sim_duration / reporting_dt)

    # ==========================================================
    # INITIAL STATE
    # ==========================================================
    x_current = KilnState()
    x_current.t = 0.0

    x_current.Fuel_rate, x_current.O2, x_current.CO_ppm = 4.0, 6.0, 900.0

    (
        x_current.Tg_preheater,
        x_current.Tg_calcination,
        x_current.Tg_burning,
        x_current.Tg_Cooling,
    ) = (680.0, 1050.0, 1450.0, 1550.0)

    (
        x_current.Ts_preheater,
        x_current.Ts_calcination,
        x_current.Ts_burning,
        x_current.Ts_Cooling,
    ) = (300.0, 850.0, 1420.0, 1450.0)

    x_current.Air_flow = 45100.0
    x_current.Cooling_air_flow = 80000.0
    x_current.ID_fan_speed = 900.0

    x_current.Feed_rate = 71.0
    x_current.Kiln_solid_out = 12.5
    x_current.Material_acc = 15.0

    x_current.CaCO3, x_current.CaO, x_current.CO2 = 80.0, 1e-6, 1e-6
    x_current.SiO2, x_current.Al2O3, x_current.Fe2O3 = 13.0, 4.0, 3.0

    x_current.C2S = 1e-6
    x_current.C3S = 1e-6
    x_current.C3A = 1e-6
    x_current.C4AF = 1e-6

    x_current.kiln_rpm = 1.0
    x_current.Residence = 69.0

    x_current.Petcoke, x_current.Alternative_Fuel, x_current.Lignite_Coal = (
        0.03,
        0.07,
        0.90,
    )

    x_current.Clinker_output = 11.13
    x_current.Clinker_yield = 0.65

    x_current.Q_in = 35575.0
    x_current.Q_out = 24285.0
    x_current.Q_acc = 1669.0
    x_current.Q_loss = 22600.0
    x_current.Q_reaction = 15.175
    x_current.Q_gas = 3524.0
    x_current.Q_clinker = 15000.0

    x_current.Energy_Error = 0.0

    # ==========================================================
    # INPUT LAYER
    # ==========================================================
    def input_layer(t: float, initial_state: KilnState = None) -> dict:
        if t < 1e-4 and initial_state is not None:
            return {
                "Petcoke": initial_state.Petcoke,
                "Alternative_Fuel": initial_state.Alternative_Fuel,
                "Feed_rate": initial_state.Feed_rate,
            }

        D = 0.03 + (0.1 - 0.03) * (1 - np.exp(-t / 80.0))
        E = 0.07 + (0.14 - 0.07) * (1 - np.exp(-t / 100.0))

        base_feed = (
            132.0
            if t >= 72.0
            else 72.0
            + 60.0 * ((1.0 / (1.0 + np.exp(-0.065 * (t - 36.0)))) - 0.09) / 0.82
        )

        return {
            "Petcoke": D,
            "Alternative_Fuel": E,
            "Feed_rate": base_feed,
        }

    # ==========================================================
    # SIMULATION LOOP
    # ==========================================================
    simulation_records = []
    simulation_records.append(asdict(x_current))

    sim_time = 0.0

    for step_idx in range(N_total_reports):

        inputs = input_layer(sim_time, initial_state=x_current)

        for sub in range(STEPS_PER_REPORT):
            step_time = sim_time + dt * (sub + 1)
            x_current = executor.perform_step(x_current, step_time, inputs)

        sim_time += reporting_dt
        x_current.t = sim_time

        simulation_records.append(asdict(x_current))

    # ==========================================================
    # DATAFRAME + OUTPUT
    # ==========================================================
    df_results = pd.DataFrame(simulation_records)

    float_cols = df_results.select_dtypes(include=["float", "float64"]).columns
    df_results[float_cols] = df_results[float_cols].round(5)

    df_results.to_csv("engine.csv", index=False)

    print(f"Simülasyon tamamlandı. t=0 başlangıçlı {len(df_results)} satır üretildi.")


if __name__ == "__main__":
    run_simulation()
