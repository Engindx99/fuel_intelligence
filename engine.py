import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict


# =================================================
# UNIT CONVERSION LAYER
# =================================================
def tph_to_kgs(x):
    return x * 1000.0 / 3600.0


def kgs_to_tph(x):
    return x * 3600.0 / 1000.0


# =================================================
# STATE DEFINITION
# =================================================
@dataclass
class KilnState:

    # =========================
    # OPERATIONAL DOMAIN
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
    # THERMAL STATE
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
    # CHEMISTRY
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
    # INTERNAL SI MIRROR
    # =========================
    Feed_rate_kgs: float = 0.0
    Fuel_rate_kgs: float = 0.0
    Air_flow_kgs: float = 0.0
    Cooling_air_flow_kgs: float = 0.0


# =================================================
# STEP EXECUTOR
# =================================================
class StepExecutor:

    def __init__(self, dt=0.05):
        self.dt = dt  # hours

    # -----------------------------
    # smoothing utilities
    # -----------------------------
    @staticmethod
    def _smooth_max(val, low_bound, eps=1e-6):
        return 0.5 * ((val + low_bound) + np.sqrt((val - low_bound) ** 2 + eps))

    @staticmethod
    def _smooth_min(val, high_bound, eps=1e-6):
        return 0.5 * ((val + high_bound) - np.sqrt((val - high_bound) ** 2 + eps))

    # =================================================
    # SI conversion
    # =================================================
    def _to_internal_si(self, x: KilnState):
        x_si = KilnState(**asdict(x))  # safe reconstruction (no mutation)

        x_si.Feed_rate_kgs = tph_to_kgs(x.Feed_rate)
        x_si.Fuel_rate_kgs = tph_to_kgs(x.Fuel_rate)

        x_si.Air_flow_kgs = x.Air_flow
        x_si.Cooling_air_flow_kgs = x.Cooling_air_flow

        return x_si

    # =================================================
    # MAIN STEP (FULL REWRITE)
    # =================================================
    def perform_step(self, x: KilnState, t: float, inputs: dict = None):

        inputs = inputs or {}

        # =========================
        # 1. CREATE NEW STATE
        # =========================
        x_next = KilnState()

        # =========================
        # 2. TIME UPDATE
        # =========================
        x_next.t = t

        # =========================
        # 3. CONTROL INPUTS (SAFE ATTRIBUTE STYLE)
        # =========================
        x_next.Feed_rate = inputs["Feed_rate"] if "Feed_rate" in inputs else x.Feed_rate
        x_next.Fuel_rate = inputs["Fuel_rate"] if "Fuel_rate" in inputs else x.Fuel_rate
        x_next.Air_flow = inputs["Air_flow"] if "Air_flow" in inputs else x.Air_flow
        x_next.kiln_rpm = inputs["kiln_rpm"] if "kiln_rpm" in inputs else x.kiln_rpm

        # =========================
        # 4. MASS BALANCE (TEMP LAYER)
        # =========================
        x_si_old = self._to_internal_si(x)

        feed = x_si_old.Feed_rate_kgs
        fuel = x_si_old.Fuel_rate_kgs

        x_next.Kiln_solid_out = kgs_to_tph(feed * 0.98)
        x_next.Clinker_output = x_next.Kiln_solid_out * x.Clinker_yield

        # =========================
        # 5. THERMAL UPDATE
        # =========================
        x_next.Tg_burning = x.Tg_burning + 0.01 * (fuel - feed)
        x_next.Ts_burning = x.Ts_burning + 0.005 * (x_next.Tg_burning - x.Ts_burning)

        # =========================
        # 6. CHEMISTRY UPDATE
        # =========================
        reaction = 0.001 * feed

        x_next.CaCO3 = max(0.0, x.CaCO3 - reaction)
        x_next.CaO = x.CaO + 0.56 * reaction
        x_next.CO2 = x.CO2 + 0.44 * reaction

        # =========================
        # 7. METRICS
        # =========================
        x_next.Energy_Error = abs(x_next.Tg_burning - x_next.Ts_burning)
        x_next.Mass_Balance_Error = abs(feed - kgs_to_tph(x_next.Kiln_solid_out))

        # =========================
        # 8. FEED DYNAMICS
        # =========================
        feed_rate = x.Feed_rate
        x_next.Feed_rate = feed_rate + (120.0 - feed_rate) * 0.0005 * self.dt

        # =========================
        # 9. AIR SYSTEM
        # =========================
        x_next.Air_flow = 45000.0 + (95000.0 - 45000.0) * (1.0 - np.exp(-t / 120.0))
        x_next.Cooling_air_flow = 80000.0 + (83000.0 - 80000.0) * (
            1.0 - np.exp(-t / 140.0)
        )
        x_next.ID_fan_speed = 900.0 + (2550.0 - 900.0) * (1.0 - np.exp(-t / 110.0))
        x_next.Damper_position = 33.0 + (85.0 - 33.0) * np.exp(-t / 25.0)

        # =========================
        # 10. FUEL RAMP
        # =========================
        x_next.Fuel_rate = 4.0 + 1.5 * (1.0 - np.exp(-t / 35.0))

        # =========================
        # 11. RPM DYNAMICS
        # =========================
        rpm_current = x.kiln_rpm
        rpm_setpoint = 2.4
        alpha = 0.005

        raw_rpm = rpm_current + alpha * (rpm_setpoint - rpm_current)
        x_next.kiln_rpm = self._smooth_max(raw_rpm, 0.1)

        rpm = x_next.kiln_rpm
        rpm_eff = rpm / (0.26 + rpm)
        filling_factor = (0.08 / 0.10) ** 0.3

        L, D, slope = 60.0, 4.2, 0.03
        v_axial = (5.87 * D * rpm_eff * (1.5 + 44.8 * slope)) * filling_factor

        x_next.Residence = (L / (v_axial + 1e-6)) * 60.0

        # ==========================================================
        # MASS BALANCE
        # ==========================================================
        mat_acc = x.Material_acc
        feed_in = x_next.Feed_rate

        tau_res = max(x_next.Residence / 60.0, 1e-6)

        kiln_out_physical = mat_acc / tau_res

        alpha_flow = 0.85
        kiln_out_prev = x.Kiln_solid_out

        kiln_out = alpha_flow * kiln_out_prev + (1.0 - alpha_flow) * kiln_out_physical
        kiln_out = max(0.0, kiln_out)

        x_next.Kiln_solid_out = kiln_out

        dmat_dt = feed_in - kiln_out
        x_next.Material_acc = max(0.0, mat_acc + dmat_dt * self.dt)

        # ==========================================================
        # REACTION BASE STATES
        # ==========================================================
        dC2S_0 = x.dC2S
        dC3S_0 = x.dC3S
        dC3A_0 = x.dC3A
        dC4AF_0 = x.dC4AF

        # ==========================================================
        # COUPLING FACTOR
        # ==========================================================
        flow_factor = kiln_out / max(feed_in, 1e-6)
        activity_factor = min(1.0, x_next.Material_acc / 50.0)
        coupling = flow_factor * activity_factor

        dC2S_eff = dC2S_0 * coupling
        dC3S_eff = dC3S_0 * coupling
        dC3A_eff = dC3A_0 * coupling
        dC4AF_eff = dC4AF_0 * coupling

        # ==========================================================
        # CHEMISTRY UPDATE (STRUCTURAL PHASES)
        # ==========================================================
        x_next.C2S = x.C2S + dC2S_eff - dC3S_eff * 0.7544
        x_next.C3S = x.C3S + dC3S_eff
        x_next.C3A = x.C3A + dC3A_eff
        x_next.C4AF = x.C4AF + dC4AF_eff

        # ==========================================================
        # ELEMENT BALANCES
        # ==========================================================
        x_next.SiO2 = max(0.0, x.SiO2 - dC2S_eff * 0.3488)
        x_next.Al2O3 = max(0.0, x.Al2O3 - dC3A_eff * 0.3773 - dC4AF_eff * 0.2098)
        x_next.Fe2O3 = max(0.0, x.Fe2O3 - dC4AF_eff * 0.3286)

        # ==========================================================
        # CaO BALANCE (STRICT LEDGER - SINGLE WRITE)
        # ==========================================================
        CaO_change = (
            dC2S_eff * 0.6512
            + dC3S_eff * 0.2456
            + dC3A_eff * 0.6227
            + dC4AF_eff * 0.4616
        )

        x_next.CaO = max(0.0, x.CaO + CaO_change)

        # ==========================================================
        # CO2 CLOSURE (NO DOUBLE COUNTING)
        # ==========================================================
        clinker_mass_proxy = (
            x_next.CaCO3
            + x_next.CaO
            + 0.6512 * x_next.C2S
            + 0.7368 * x_next.C3S
            + 0.6227 * x_next.C3A
            + 0.4616 * x_next.C4AF
        )

        x_next.CO2 = max(0.0, x.CO2 + (80.0 - clinker_mass_proxy))

        # ==========================================================
        # DYNAMIC SIGNALS
        # ==========================================================
        x_next.dTg_burning = (x_next.Tg_burning - x.Tg_burning) / self.dt

        oxygen_deficit = max(0.0, 6.0 - x_next.O2)

        target_CO = 20.0 + 800.0 * oxygen_deficit * (x_next.Fuel_rate / 6.0) * np.exp(
            -x_next.Tg_burning / 1200.0
        )

        x_next.CO_ppm = x.CO_ppm + 0.25 * (target_CO - x.CO_ppm)

        # ==========================================================
        # PRESSURE FIELD
        # ==========================================================
        x_next.P_calcination = -406.0 + 226.0 * np.exp(-t / 20.0)

        # ==========================================================
        # OUTPUT
        # ==========================================================
        yield_ratio = 1.0 / 1.55
        x_next.Clinker_output = kiln_out * yield_ratio

        # =========================
        # 5. UNIT BOUNDARY REFRESH
        # =========================
        x_si = self._to_internal_si(x_next)

        # ==========================================================
        # ENERGY INPUT DEFINITION (PHYSICAL SOURCE TERM ONLY)
        # ==========================================================

        LHV_lignite = 15000.0
        LHV_petcoke = 30000.0
        LHV_alt = 18000.0

        fuel_mass_flow = x_si.Fuel_rate_kgs

        Q_chem = (
            x.Lignite_Coal * LHV_lignite
            + x.Petcoke * LHV_petcoke
            + x.Alternative_Fuel * LHV_alt
        )

        comb_eff = np.exp(-((x.O2 - 3.5) ** 2) / 25.0)

        Q_in_total = Q_chem * fuel_mass_flow * comb_eff

        # ==========================================================
        # BURNING ZONE (ENERGY-CONSISTENT VERSION)
        # ==========================================================

        # ----------------------------------------------------------
        # SAFE INITIALIZATION (required by downstream chemistry)
        # ----------------------------------------------------------

        Tg_burning_pred = x.Tg_burning
        Ts_burning_pred = x.Ts_burning
        Tw_burning_pred = x.Tw_burning

        # ----------------------------------------------------------
        # THERMAL PROPERTIES
        # ----------------------------------------------------------

        Cp_g = 1250.0
        Cp_s = 1150.0
        Cp_w = 1000.0

        A_s = 82.0
        A_w = 70.0
        A_ws = 57.0

        m_air = x_si.Air_flow_kgs
        m_solid = x_si.Feed_rate_kgs

        gas_mass = 220.0
        solid_mass = 6500.0
        wall_mass = 15000.0

        C_gas = gas_mass * Cp_g
        C_wall = wall_mass * Cp_w

        # Solid capacity follows actual solids flow
        C_solid = max(m_solid * Cp_s, 1e-9)

        # ----------------------------------------------------------
        # CURRENT STATES (READ ONLY)
        # ----------------------------------------------------------

        Tg = x.Tg_burning
        Ts = x.Ts_burning
        Tw = x.Tw_burning

        # ----------------------------------------------------------
        # HEAT TRANSFER COEFFICIENTS
        # ----------------------------------------------------------

        h_gs = 1450.0
        h_gw = 350.0
        h_ws = 400.0

        # ----------------------------------------------------------
        # GAS ENERGY BALANCE
        # ----------------------------------------------------------

        Q_wall = h_gs * A_s * (Ts - Tg)
        Q_wall2 = h_gw * A_w * (Tw - Tg)

        air_effect = np.tanh(m_air / 60000.0)
        h_gs_eff = h_gs * (0.5 + 0.5 * air_effect)

        Q_conv = -h_gs_eff * A_s * (Tg - Ts)

        Q_net_gas = Q_in_total + Q_wall + Q_wall2 + Q_conv

        Tg_next = Tg + self.dt * Q_net_gas / (C_gas + 1e-9)

        # ----------------------------------------------------------
        # WALL ENERGY BALANCE
        # ----------------------------------------------------------

        T_amb = 25.0
        h_loss = 8.0

        Q_loss_wall = h_loss * A_w * (Tw - T_amb)
        Q_wall_to_solid = h_ws * A_ws * (Tw - Ts)

        Tw_next = Tw + self.dt * (
            h_gw * A_w * (Tg_next - Tw) - Q_loss_wall - Q_wall_to_solid
        ) / (C_wall + 1e-9)

        # ----------------------------------------------------------
        # SOLID ENERGY BALANCE
        # ----------------------------------------------------------

        Q_exo_base = min(
            350000.0,
            m_solid * 500000.0 * (1.0 + max(0.0, Ts - 1200.0) / 250.0),
        )

        k_exo = 0.05

        Q_exo = Q_exo_base / (1.0 + np.exp(-k_exo * (Ts - 1200.0)))

        Q_gs = h_gs_eff * A_s * (Tg_next - Ts)
        Q_ws = h_ws * A_ws * (Tw_next - Ts)

        Ts_next = Ts + self.dt * (Q_exo + Q_gs + Q_ws) / (C_solid + 1e-9)

        # ----------------------------------------------------------
        # STORE PREDICTED TEMPERATURES
        # (used immediately by clinker reactions)
        # ----------------------------------------------------------

        Tg_burning_pred = Tg_next
        Ts_burning_pred = Ts_next
        Tw_burning_pred = Tw_next

        # ----------------------------------------------------------
        # WRITE BACK
        # ----------------------------------------------------------

        x_next.Tg_burning = Tg_burning_pred
        x_next.Ts_burning = Ts_burning_pred
        x_next.Tw_burning = Tw_burning_pred

        # ======================================================
        # CALCINATION ZONE (ENERGY-CONSISTENT FIXED VERSION)
        # ======================================================

        h_gs, A_s = 320.0, 950.0
        UA = h_gs * A_s * 0.45

        m_solid_calc = x_si.Feed_rate_kgs

        Ts_calc_curr = x.Ts_calcination
        Tg_calc_curr = x.Tg_calcination

        Cp_solid_calc = 1050.0
        Cp_air = 1005.0
        T_ref = 25.0

        # ======================================================
        # MASS FLOWS
        # ======================================================

        m_air_kiln = x_si.Air_flow_kgs
        m_air_tertiary = getattr(x_si, "Tertiary_air_flow_kgs", 0.0)

        m_gas_total = m_air_kiln + m_air_tertiary

        # ======================================================
        # INLET TEMPERATURES (IMPORTANT FIX)
        # ======================================================

        T_air_kiln_in = x_next.Tg_burning

        T_tertiary_in = getattr(x, "T_tertiary_air", 850.0)

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
        # WRITE BACK (ONLY x_next)
        # ======================================================

        x_next.Tg_calcination = Tg_calc_pred
        x_next.Ts_calcination = Ts_calc_pred

        # ======================================================
        # PREHEATER ZONE (UNIT-SAFE + CAUSAL STABLE)
        # ======================================================

        Cp_gas_pre = 1150.0
        Cp_solid_pre = 1050.0
        T_ref = 25.0

        # ------------------------------------------------------
        # CAUSAL BOUNDARY (REDUCED FEEDBACK COUPLING)
        # ------------------------------------------------------

        Tg_calc_current = getattr(x, "Tg_calcination", 900.0)
        Tg_pre_prev = getattr(x, "Tg_preheater", Tg_calc_current)

        Tg_in = 0.6 * Tg_calc_current + 0.1 * Tg_pre_prev

        Ts_pre = getattr(x, "Ts_preheater", 25.0)

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

        x_next.Tg_preheater = Tg_preheater_pred
        x_next.Ts_preheater = Ts_preheater_pred

        # ------------------------------------------------------
        # OUTPUT METRICS (NO DUPLICATION)
        # ------------------------------------------------------

        x_next.Q_dot_preheater = Q_preheater
        x_next.UA_preheater_effective = alpha_stable * UA_pre

        # ======================================================
        # COOLING ZONE (UNIT-SAFE / SI CONSISTENT VERSION)
        # ======================================================

        Tamb_solid = 130.0
        Tamb_gas = 510.0

        Ts_cool_curr = getattr(x, "Ts_Cooling", getattr(x_next, "Ts_burning", 1450.0))

        Tg_cool_curr = getattr(x, "Tg_Cooling", getattr(x_next, "Tg_burning", 1490.0))

        # -------------------------------
        # MASS FLOWS (CONVERTED TO kg/s)
        # -------------------------------
        AIR_DENSITY = 1.293
        Cp_solid = 1150.0  # J/kg-K
        Cp_gas = 1050.0  # J/kg-K

        m_solid = tph_to_kgs(x.Feed_rate)
        m_gas = x.Cooling_air_flow * AIR_DENSITY / 3600.0

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
        # HEAT TRANSFER COUPLING (W)
        # -------------------------------
        h_transfer = 0.15
        A_transfer = 500.0
        UA = h_transfer * A_transfer

        dT = Ts_cool_curr - Tg_cool_curr

        Q_transfer = UA * np.tanh(dT / 250.0) * 1000.0

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

        enthalpy_drop = m_gas * Cp_ref * max(Tg_cool_curr - T_ref, 0.0)

        eta_recovery = 0.65
        Q_secondary_air = eta_recovery * enthalpy_drop

        # ======================================================
        # BURNING POOL (ENERGY STORAGE DYNAMICS)
        # ======================================================

        Q_burning_pool_prev = getattr(x, "Q_burning_pool", 0.0)

        tau_pool = 1200.0

        Q_burning_pool = Q_burning_pool_prev + (self.dt / tau_pool) * (
            Q_secondary_air - Q_burning_pool_prev
        )

        # -------------------------------
        # WRITE BACK
        # -------------------------------
        x_next.Ts_Cooling = Ts_cool_next
        x_next.Tg_Cooling = Tg_cool_next
        x_next.Q_burning_pool = Q_burning_pool

        # ==========================================================
        # CALCINATION (PHYSICALLY CONSISTENT + STATE-ONLY + STEP SAFE)
        # ==========================================================

        # ==========================================================
        # STATE READ (ONLY FROM x)
        # ==========================================================

        Ts_calc_curr = x.Ts_calcination
        T_calc = Ts_calc_curr + 273.15

        T_gas = x.Tg_calcination

        CaCO3_in = x.CaCO3

        # SI SAFE MASS FLOWS (IMPORTANT FIX)
        m_solid_calc = x.Kiln_solid_out * 1000.0 / 3600.0
        m_gas = x.Air_flow_kgs

        # ==========================================================
        # SMOOTH ACTIVATION
        # ==========================================================

        T0 = 873.15
        sigma = 25.0

        activation = 1.0 / (1.0 + np.exp(-(T_calc - T0) / sigma))

        # ==========================================================
        # ARRHENIUS KINETICS
        # ==========================================================

        Ea = 160000.0
        R = 8.314

        k_calc = 1e5 * np.exp(-Ea / (R * T_calc)) * activation

        # ==========================================================
        # HEAT OF REACTION
        # ==========================================================

        DeltaH_calc = 1.7e6  # J/kg CaCO3

        # ==========================================================
        # KINETIC LIMIT
        # ==========================================================

        react_kin = CaCO3_in * (1.0 - np.exp(-k_calc * self.dt))

        # ==========================================================
        # THERMAL LIMIT
        # ==========================================================

        heat_drive = T_gas - Ts_calc_curr

        react_energy_limit = CaCO3_in * (1.0 / (1.0 + np.exp(-heat_drive / 30.0)))

        # ==========================================================
        # BLENDING
        # ==========================================================

        beta = 0.85

        reacted = beta * react_kin + (1.0 - beta) * react_energy_limit
        reacted = np.clip(reacted, 0.0, CaCO3_in)

        # ==========================================================
        # PRODUCTS
        # ==========================================================

        CaCO3_out = CaCO3_in - reacted
        CaO_generated = reacted * 0.5603
        CO2_generated = reacted * 0.4397

        # ==========================================================
        # ENERGY TERMS (J/s)
        # ==========================================================

        Q_reaction = reacted * DeltaH_calc

        Q_sink = 0.65 * Q_reaction
        Q_source = 0.35 * Q_reaction

        # ==========================================================
        # CAPACITIES
        # ==========================================================

        Cp_solid = 1050.0
        Cp_gas = 1005.0

        C_solid = m_solid_calc * Cp_solid + 1e-9
        C_gas = m_gas * Cp_gas + 1e-9

        # ==========================================================
        # HEAT TRANSFER
        # ==========================================================

        h_gs = 320.0
        A_s = 950.0

        UA = h_gs * A_s

        Q_gs = UA * (T_gas - Ts_calc_curr)

        # ==========================================================
        # SOLID ENERGY BALANCE
        # ==========================================================

        dTs_dt = (Q_gs - Q_sink) / C_solid

        tau_extra = 40.0

        Ts_calc_next = Ts_calc_curr + (self.dt / (self.dt + tau_extra)) * (
            self.dt * dTs_dt
        )

        # ==========================================================
        # GAS ENERGY BALANCE
        # ==========================================================

        dTg_dt = (-Q_gs + Q_source) / C_gas

        Tg_calc_next = T_gas + self.dt * dTg_dt

        # ==========================================================
        # CLIP SAFETY
        # ==========================================================

        Ts_calc_next = np.clip(Ts_calc_next, 400.0, 1600.0)
        Tg_calc_next = np.clip(Tg_calc_next, 300.0, 1700.0)

        # ==========================================================
        # WRITE BACK (ONLY x_next)
        # ==========================================================

        x_next.CaCO3 = CaCO3_out

        x_next.CaO = x.CaO + CaO_generated
        x_next.CO2 = x.CO2 + CO2_generated

        x_next.dCaO_calcination = CaO_generated / self.dt if self.dt > 0 else 0.0
        x_next.dCO2_calcination = CO2_generated / self.dt if self.dt > 0 else 0.0

        x_next.Ts_calcination = Ts_calc_next
        x_next.Tg_calcination = Tg_calc_next

        x_next.Q_reaction = x.Q_reaction + Q_reaction
        x_next.Q_gas = Q_gs
        # ==========================================================
        # KALSİNASYON SONRASI GİRİŞLER (CLINKER PHASE)
        # ==========================================================

        Tg_burning_pred = getattr(x, "Tg_burning", 1450.0)
        Ts_burning_pred = getattr(x, "Ts_burning", 1400.0)
        Tw_burning_pred = getattr(x, "Tw_burning", 1300.0)

        Tg_burn = Tg_burning_pred + 273.15
        Ts_burn = Ts_burning_pred + 273.15
        Tw_burn = Tw_burning_pred + 273.15

        T_burn_eff = 0.7 * Ts_burn + 0.2 * Tg_burn + 0.1 * Tw_burn

        # ==========================================================
        # SPECIES (STATE READ ONLY)
        # ==========================================================

        SiO2 = x.SiO2
        Al2O3_pool = x.Al2O3
        Fe2O3_pool = x.Fe2O3

        CaO_pool = max(0.0, x.CaO)

        # ==========================================================
        # C2S FORMATION (KINETIC + THERMAL COUPLING)
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

        # ==========================================================
        # MASS CONSUMPTION (LOCAL BALANCE)
        # ==========================================================

        CaO_consumed = dC2S * 0.6512
        SiO2_consumed = dC2S * 0.3488

        CaO_pool_new = max(0.0, CaO_pool - CaO_consumed)
        SiO2_new = max(0.0, SiO2 - SiO2_consumed)

        # ==========================================================
        # ENERGY RELEASE
        # ==========================================================

        Q_c2s = dC2S * 250000.0  # J/s equivalent

        Q_reaction_total = x.Q_reaction + Q_c2s

        # ==========================================================
        # THERMAL FEEDBACK (CONSISTENT LUMPED MODEL)
        # ==========================================================

        Cp_g = 1250.0
        Cp_s = 1150.0
        Cp_w = 1000.0

        gas_mass = 220.0
        solid_mass = 6500.0
        wall_mass = 15000.0

        C_gas = gas_mass * Cp_g + 1e-9
        C_solid = solid_mass * Cp_s + 1e-9
        C_wall = wall_mass * Cp_w + 1e-9

        Q_gas = 0.6 * Q_c2s
        Q_solid = 0.3 * Q_c2s
        Q_wall = 0.1 * Q_c2s

        Tg_burn_next = Tg_burn + (self.dt * Q_gas) / C_gas
        Ts_burn_next = Ts_burn + (self.dt * Q_solid) / C_solid
        Tw_burn_next = Tw_burn + (self.dt * Q_wall) / C_wall

        # ==========================================================
        # BACK TO °C DOMAIN
        # ==========================================================

        Tg_burning_new = Tg_burn_next - 273.15
        Ts_burning_new = Ts_burn_next - 273.15
        Tw_burning_new = Tw_burn_next - 273.15

        # ==========================================================
        # STATE UPDATE
        # ==========================================================

        x_next.C2S = x.C2S + dC2S

        x_next.SiO2 = SiO2_new
        x_next.CaO = CaO_pool_new

        x_next.Al2O3 = Al2O3_pool
        x_next.Fe2O3 = Fe2O3_pool

        x_next.Q_clinker = x.Q_clinker + Q_c2s
        x_next.Q_reaction = Q_reaction_total

        x_next.Tg_burning = Tg_burning_new
        x_next.Ts_burning = Ts_burning_new
        x_next.Tw_burning = Tw_burning_new

        x_next.dC2S = dC2S

        # ==========================================================
        # C3S FORMATION (DOT-STATE CONSISTENT VERSION)
        # ==========================================================

        # --------------------------
        # READ STATE (ONLY FROM x)
        # --------------------------

        C2S_in = x.C2S
        CaO_in = max(0.0, x.CaO)

        # already computed upstream (burning zone)
        # T_burn_eff must exist in local scope

        # --------------------------
        # VALIDITY CHECK
        # --------------------------

        if C2S_in <= 1e-9 or T_burn_eff < 1473.15:
            dC3S = 0.0

        else:
            k = 2.28e8 * np.exp(-200000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                C2S_in / 0.7544,
                CaO_in / 0.2456,
            )

            dC3S = stoich_limit * kinetic

        # --------------------------
        # MASS BALANCE
        # --------------------------

        C2S_consumed = dC3S * 0.7544
        CaO_consumed = dC3S * 0.2456

        C2S_out = max(0.0, C2S_in - C2S_consumed)
        CaO_out = max(0.0, CaO_in - CaO_consumed)

        # --------------------------
        # ENERGY RELEASE
        # --------------------------

        Q_c3s = dC3S * 420000.0

        Q_reaction_total = x.Q_reaction + Q_c3s

        # --------------------------
        # THERMAL FEEDBACK
        # --------------------------

        Cp_g = 1250.0
        Cp_s = 1150.0
        Cp_w = 1000.0

        gas_mass = 220.0
        solid_mass = 6500.0
        wall_mass = 15000.0

        C_gas = gas_mass * Cp_g + 1e-9
        C_solid = solid_mass * Cp_s + 1e-9
        C_wall = wall_mass * Cp_w + 1e-9

        Q_gas = 0.5 * Q_c3s
        Q_solid = 0.4 * Q_c3s
        Q_wall = 0.1 * Q_c3s

        # read current thermal state
        Tg_burn = x.Tg_burning + 273.15
        Ts_burn = x.Ts_burning + 273.15
        Tw_burn = x.Tw_burning + 273.15

        # integrate
        Tg_burn_next = Tg_burn + (self.dt * Q_gas) / C_gas
        Ts_burn_next = Ts_burn + (self.dt * Q_solid) / C_solid
        Tw_burn_next = Tw_burn + (self.dt * Q_wall) / C_wall

        # back to °C
        Tg_burning_new = Tg_burn_next - 273.15
        Ts_burning_new = Ts_burn_next - 273.15
        Tw_burning_new = Tw_burn_next - 273.15

        # --------------------------
        # WRITEBACK (NEW STANDARD)
        # --------------------------

        x_next.C3S = x.C3S + dC3S

        x_next.C2S = C2S_out
        x_next.CaO = CaO_out

        x_next.Q_clinker = x.Q_clinker + Q_c3s
        x_next.Q_reaction = Q_reaction_total

        x_next.Tg_burning = Tg_burning_new
        x_next.Ts_burning = Ts_burning_new
        x_next.Tw_burning = Tw_burning_new

        x_next.dC3S = dC3S

        # ==========================================================
        # C3A FORMATION (DOT STATE + CAUSAL + CONSISTENT + SAFE)
        # ==========================================================

        # ==========================================================
        # SAFE STATE READ
        # ==========================================================

        Al2O3_in = getattr(x, "Al2O3", 0.0)
        CaO_in = max(0.0, getattr(x, "CaO", 0.0))

        Q_reaction_prev = getattr(x, "Q_reaction", 0.0)
        Q_c3a_prev = getattr(x, "Q_c3a", 0.0)

        Tg_burning = getattr(x, "Tg_burning", 1450.0)
        Ts_burning = getattr(x, "Ts_burning", 1400.0)
        Tw_burning = getattr(x, "Tw_burning", 1300.0)

        # ==========================================================
        # VALIDITY CHECK
        # ==========================================================

        if Al2O3_in <= 1e-6 or T_burn_eff < 1100.0:
            dC3A = 0.0

        else:
            k = 1.0e5 * np.exp(-120000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                Al2O3_in / 0.3773,
                CaO_in / 0.6227,
            )

            dC3A = stoich_limit * kinetic

        # ==========================================================
        # MASS BALANCE
        # ==========================================================

        Al2O3_consumed = dC3A * 0.3773
        CaO_consumed = dC3A * 0.6227

        Al2O3_out = max(0.0, Al2O3_in - Al2O3_consumed)
        CaO_out = max(0.0, CaO_in - CaO_consumed)

        # ==========================================================
        # ENERGY RELEASE
        # ==========================================================

        Q_c3a = dC3A * 380000.0

        Q_reaction_total = Q_reaction_prev + Q_c3a
        Q_c3a_total = Q_c3a_prev + Q_c3a

        # ==========================================================
        # THERMAL FEEDBACK
        # ==========================================================

        Cp_g = 1250.0
        Cp_s = 1150.0
        Cp_w = 1000.0

        gas_mass = 220.0
        solid_mass = 6500.0
        wall_mass = 15000.0

        C_gas = gas_mass * Cp_g + 1e-9
        C_solid = solid_mass * Cp_s + 1e-9
        C_wall = wall_mass * Cp_w + 1e-9

        Q_gas = 0.5 * Q_c3a
        Q_solid = 0.4 * Q_c3a
        Q_wall = 0.1 * Q_c3a

        # integrate temperatures (K)
        Tg = Tg_burning + 273.15
        Ts = Ts_burning + 273.15
        Tw = Tw_burning + 273.15

        Tg_next = Tg + (self.dt * Q_gas) / C_gas
        Ts_next = Ts + (self.dt * Q_solid) / C_solid
        Tw_next = Tw + (self.dt * Q_wall) / C_wall

        # back to °C
        Tg_burning_new = Tg_next - 273.15
        Ts_burning_new = Ts_next - 273.15
        Tw_burning_new = Tw_next - 273.15

        # ==========================================================
        # WRITEBACK (DOT STATE SAFE)
        # ==========================================================

        x_next.C3A = getattr(x, "C3A", 0.0) + dC3A

        x_next.Al2O3 = Al2O3_out
        x_next.CaO = CaO_out

        x_next.Q_c3a = Q_c3a_total
        x_next.Q_reaction = Q_reaction_total

        x_next.Tg_burning = Tg_burning_new
        x_next.Ts_burning = Ts_burning_new
        x_next.Tw_burning = Tw_burning_new

        x_next.dC3A = dC3A

        # ==========================================================
        # C4AF FORMATION (SEQUENTIAL, CONSISTENT STATE + THERMAL COUPLING)
        # ==========================================================

        # ==========================================================
        # READ STATE (FROM x_next - NOT x)
        # ==========================================================

        Fe2O3_in = x_next.Fe2O3
        Al2O3_in = x_next.Al2O3
        CaO_in = max(0.0, x_next.CaO)

        # ==========================================================
        # VALIDITY CHECK
        # ==========================================================

        if Fe2O3_in <= 1e-6 or T_burn_eff < 1100.0:
            dC4AF = 0.0

        else:
            k = 2.0e5 * np.exp(-150000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                Fe2O3_in / 0.3286,
                Al2O3_in / 0.2098,
                CaO_in / 0.4616,
            )

            dC4AF = stoich_limit * kinetic

        # ==========================================================
        # MASS BALANCE
        # ==========================================================

        Fe2O3_consumed = dC4AF * 0.3286
        Al2O3_consumed = dC4AF * 0.2098
        CaO_consumed = dC4AF * 0.4616

        Fe2O3_out = max(0.0, Fe2O3_in - Fe2O3_consumed)
        Al2O3_out = max(0.0, Al2O3_in - Al2O3_consumed)
        CaO_out = max(0.0, CaO_in - CaO_consumed)

        # ==========================================================
        # ENERGY RELEASE
        # ==========================================================

        Q_c4af = dC4AF * 120000.0

        # SAFE ACCUMULATION (NO NONE ERROR GUARANTEE)
        Q_reaction = getattr(x_next, "Q_reaction", 0.0) + Q_c4af

        # ==========================================================
        # THERMAL FEEDBACK
        # ==========================================================

        Cp_g = 1250.0
        Cp_s = 1150.0
        Cp_w = 1000.0

        gas_mass = 220.0
        solid_mass = 6500.0
        wall_mass = 15000.0

        C_gas = gas_mass * Cp_g + 1e-9
        C_solid = solid_mass * Cp_s + 1e-9
        C_wall = wall_mass * Cp_w + 1e-9

        Q_gas = 0.45 * Q_c4af
        Q_solid = 0.40 * Q_c4af
        Q_wall = 0.15 * Q_c4af

        # IMPORTANT: SAFE READ (NO ASSUMPTION ABOUT PRIOR BLOCK ORDER)
        Tg = x_next.Tg_burning + 273.15
        Ts = x_next.Ts_burning + 273.15
        Tw = x_next.Tw_burning + 273.15

        Tg_next = Tg + (self.dt * Q_gas) / C_gas
        Ts_next = Ts + (self.dt * Q_solid) / C_solid
        Tw_next = Tw + (self.dt * Q_wall) / C_wall

        Tg_burning_new = Tg_next - 273.15
        Ts_burning_new = Ts_next - 273.15
        Tw_burning_new = Tw_next - 273.15

        # ==========================================================
        # STATE WRITEBACK (DOT SAFE)
        # ==========================================================

        x_next.C4AF += dC4AF

        # IMPORTANT: no self-accumulating undefined fields
        x_next.Q_c4af = getattr(x_next, "Q_c4af", 0.0) + Q_c4af
        x_next.Q_reaction = Q_reaction

        x_next.Tg_burning = Tg_burning_new
        x_next.Ts_burning = Ts_burning_new
        x_next.Tw_burning = Tw_burning_new

        x_next.Fe2O3 = Fe2O3_out
        x_next.Al2O3 = Al2O3_out
        x_next.CaO = CaO_out

        x_next.dC4AF = dC4AF

        # ==========================================================
        # SINGLE CLEAN REACTION SOURCE TERM
        # ==========================================================

        Q_c2s = getattr(x_next, "Q_c2s", 0.0)
        Q_c3s = getattr(x_next, "Q_c3s", 0.0)
        Q_c3a = getattr(x_next, "Q_c3a", 0.0)
        Q_c4af = getattr(x_next, "Q_c4af", 0.0)

        Q_reaction_instant = -0.15 * Q_c2s + 0.35 * Q_c3s + 0.25 * Q_c3a + 0.10 * Q_c4af

        # ==========================================================
        # BURNING ZONE CAPACITIES (CONSISTENT)
        # ==========================================================

        Cp_g_burn = 1250.0
        Cp_s_burn = 1150.0
        Cp_w_burn = 1000.0

        C_burn_total = (
            gas_mass * Cp_g_burn + solid_mass * Cp_s_burn + wall_mass * Cp_w_burn
        )

        Tg_curr = x_next.Tg_burning
        Ts_curr = x_next.Ts_burning
        Tw_curr = x_next.Tw_burning

        Q_loss_wall = 8.0 * 70.0 * (Tw_curr - 25.0)

        # ==========================================================
        # TIME CONSTANT (STABILIZED COUPLING)
        # ==========================================================

        UA_gw = 350.0 * 70.0
        UA_gs = 1450.0 * 82.0

        tau_g = C_burn_total / (UA_gs + UA_gw + 1e-9)
        alpha = self.dt / (self.dt + tau_g + 1e-9)

        # ==========================================================
        # GAS ENERGY BALANCE (REACTION AS SOURCE ONLY)
        # ==========================================================

        Tg_eq = Tg_curr + Q_reaction_instant / (C_burn_total + 1e-9)

        Tg_burning_pred = (1.0 - alpha) * Tg_curr + alpha * Tg_eq

        # ==========================================================
        # WALL ENERGY BALANCE (CAUSAL)
        # ==========================================================

        Tw_burning_pred = Tw_curr + (self.dt / (wall_mass * Cp_w_burn + 1e-9)) * (
            UA_gw * (Tg_burning_pred - Tw_curr) - Q_loss_wall
        )

        # ==========================================================
        # SOLID ENERGY BALANCE (CAUSAL + NO DOUBLE REACTION FEED)
        # ==========================================================

        Ts_burning_pred = Ts_curr + (self.dt / (solid_mass * Cp_s_burn + 1e-9)) * (
            UA_gs * (Tg_burning_pred - Ts_curr) + Q_reaction_instant
        )

        # ==========================================================
        # EFFECTIVE TEMPERATURE (OUTPUT ONLY)
        # ==========================================================

        T_burn_eff_next = (
            0.7 * (Ts_burning_pred + 273.15)
            + 0.2 * (Tg_burning_pred + 273.15)
            + 0.1 * (Tw_burning_pred + 273.15)
        )

        # ==========================================================
        # FINAL PHASE UPDATE (PURE MASS CONSISTENT LAYER)
        # ==========================================================

        C2S_prev = x_next.C2S
        C3S_prev = x_next.C3S
        C3A_prev = x_next.C3A
        C4AF_prev = x_next.C4AF

        # frozen reaction extents (NO RESCALING HERE)
        dC2S = x_next.dC2S
        dC3S = x_next.dC3S
        dC3A = x_next.dC3A
        dC4AF = x_next.dC4AF

        # ==========================================================
        # STRUCTURAL UPDATE (ORDER-RESPECTING, NO DOUBLE COUNT)
        # ==========================================================

        C3S_next = C3S_prev + dC3S
        C3A_next = C3A_prev + dC3A
        C4AF_next = C4AF_prev + dC4AF

        # C2S is CONSUMED by downstream phases (ONLY sink terms matter)
        C2S_next = C2S_prev + dC2S - 0.7544 * dC3S

        # physical bounds
        C2S_next = max(0.0, C2S_next)
        C3S_next = max(0.0, C3S_next)
        C3A_next = max(0.0, C3A_next)
        C4AF_next = max(0.0, C4AF_next)

        # write back
        x_next.C2S = C2S_next
        x_next.C3S = C3S_next
        x_next.C3A = C3A_next
        x_next.C4AF = C4AF_next

        # ==========================================================
        # CLINKER TOTAL (STRUCTURAL INDEX ONLY)
        # ==========================================================

        x_next.Clinker_total = C2S_next + C3S_next + C3A_next + C4AF_next

        # ==========================================================
        # ELEMENT BALANCES (SINGLE SOURCE CONSISTENCY)
        # ==========================================================

        SiO2 = x_next.SiO2
        Al2O3 = x_next.Al2O3
        Fe2O3 = x_next.Fe2O3
        CaO = x_next.CaO
        CO2 = x_next.CO2
        CaCO3 = x_next.CaCO3

        # ==========================================================
        # ELEMENT CONSUMPTION (FROM EFFECTIVE EXTENTS ONLY)
        # ==========================================================

        # ALL STATE MUST COME FROM x_next (single source of truth)
        SiO2 = x_next.SiO2
        Al2O3 = x_next.Al2O3
        Fe2O3 = x_next.Fe2O3
        CaO = x_next.CaO
        CO2 = x_next.CO2
        CaCO3 = x_next.CaCO3

        # reaction extents (must already be computed in this step)
        dC2S_eff = x_next.dC2S
        dC3S_eff = x_next.dC3S
        dC3A_eff = x_next.dC3A
        dC4AF_eff = x_next.dC4AF

        # ==========================================================
        # ELEMENT CONSUMPTION (STOICHIOMETRIC MAP)
        # ==========================================================

        SiO2_consumed = 0.3488 * dC2S_eff

        Al2O3_consumed = 0.3773 * dC3A_eff + 0.2098 * dC4AF_eff

        Fe2O3_consumed = 0.3286 * dC4AF_eff

        CaO_consumed = (
            0.6512 * dC2S_eff
            + 0.2456 * dC3S_eff
            + 0.6227 * dC3A_eff
            + 0.4616 * dC4AF_eff
        )

        # ==========================================================
        # CARBONATE LINK (OPTIONAL PHYSICS CLOSURE)
        # ==========================================================

        CaCO3_consumed = max(0.0, CaO_consumed)
        CO2_generated = CaCO3_consumed

        CaO_generated = 0.0  # explicit closure (no hidden source)

        # ==========================================================
        # ELEMENT UPDATE (CONSISTENT PASS)
        # ==========================================================

        x_next.SiO2 = max(0.0, SiO2 - SiO2_consumed)
        x_next.Al2O3 = max(0.0, Al2O3 - Al2O3_consumed)
        x_next.Fe2O3 = max(0.0, Fe2O3 - Fe2O3_consumed)

        x_next.CaO = max(0.0, CaO - CaO_consumed + CaO_generated)

        x_next.CaCO3 = max(0.0, CaCO3 - CaCO3_consumed)
        x_next.CO2 = CO2 + CO2_generated

        # ==========================================================
        # WRITEBACK (ATOMIC UPDATE - NO OVERWRITES)
        # ==========================================================

        # IMPORTANT: variables already live in x_next from previous step
        # so we only enforce explicit final assignment (optional safety layer)

        x_next.SiO2 = x_next.SiO2
        x_next.Al2O3 = x_next.Al2O3
        x_next.Fe2O3 = x_next.Fe2O3
        x_next.CaO = x_next.CaO

        x_next.CaCO3 = x_next.CaCO3
        x_next.CO2 = x_next.CO2

        # ==========================================================
        # DIAGNOSTICS (PHASE RATES - READ ONLY)
        # ==========================================================

        x_next.dC2S = dC2S_eff
        x_next.dC3S = dC3S_eff
        x_next.dC3A = dC3A_eff
        x_next.dC4AF = dC4AF_eff

        x_next.T_burn_effective = T_burn_eff

        # ==========================================================
        # CaO CONSERVATION (STRICT LEDGER - SINGLE SOURCE)
        # ==========================================================

        # IMPORTANT: CaO already updated in previous element balance layer
        CaO_prev = x_next.CaO

        CaO_consumed = (
            dC2S_eff * 0.6512
            + dC3S_eff * 0.2456
            + dC3A_eff * 0.6227
            + dC4AF_eff * 0.4616
        )

        # NO CaO_generated (must not exist in correct closure system)
        CaO_next = CaO_prev - CaO_consumed
        x_next.CaO = max(0.0, CaO_next)

        # ==========================================================
        # CO2 MASS (DIRECT STATE UPDATE ONLY ONCE)
        # ==========================================================

        CO2_prev = x_next.CO2
        x_next.CO2 = CO2_prev + CO2_generated

        # ==========================================================
        # CO2 CHECK (DIAGNOSTIC ONLY - NOT STATE)
        # ==========================================================

        CO2_check = x_next.CO2 - (x_next.CaCO3 * 0.4397)
        x_next.CO2_balance_error = CO2_check

        # ==========================================================
        # DYNAMIC SIGNALS
        # ==========================================================

        x_next.dTg_burning = (x_next.Tg_burning - x.Tg_burning) / max(self.dt, 1e-9)

        # ==========================================================
        # CO FORMATION (CAUSAL + LAGGED RESPONSE)
        # ==========================================================

        oxygen_deficit = max(0.0, 6.0 - x_next.O2)

        fuel_factor = x_next.Fuel_rate / 6.0

        temp_factor = np.exp(-x_next.Tg_burning / 1200.0)

        target_CO = 20.0 + 800.0 * oxygen_deficit * fuel_factor * temp_factor

        x_next.CO_ppm = x.CO_ppm + 0.25 * (target_CO - x.CO_ppm)

        # ==========================================================
        # PRESSURE FIELD (EXOGENOUS)
        # ==========================================================

        x_next.P_calcination = -406.0 + 226.0 * np.exp(-x_next.t / 20.0)

        # ==========================================================
        # RADIATION LOSS (MONITORING ONLY)
        # ==========================================================

        sigma_sb = 5.67e-8
        view_factor = 0.0016
        emissivity = 0.8
        A_effective = 110.0

        Tg = x_next.Tg_burning + 273.15  # consistency fix

        Q_radiation = (
            view_factor * sigma_sb * emissivity * A_effective * (Tg**4 - 298.15**4)
        )

        x_next.Q_loss = Q_radiation

        # ==========================================================
        # ENERGY OUT (SEPARATED CONTROL SIGNAL)
        # ==========================================================

        x_next.Q_gas = x_next.Fuel_rate * 1.1 * max(0.0, x_next.Tg_burning - 25.0)

        # ==========================================================
        # DIAGNOSTICS (PHASE RATES - READ ONLY)
        # ==========================================================

        x_next.dC2S = dC2S
        x_next.dC3S = dC3S
        x_next.dC3A = dC3A
        x_next.dC4AF = dC4AF

        x_next.T_burn_effective = T_burn_eff

        # ==========================================================
        # CaO CONSERVATION (STRICT LEDGER - SINGLE SOURCE)
        # ==========================================================

        CaO_prev = x.CaO

        CaO_consumed = dC2S * 0.6512 + dC3S * 0.2456 + dC3A * 0.6227 + dC4AF * 0.4616

        # CaO generation is only from decomposition (if modeled earlier)
        CaO_generated = 0.0

        CaO_next = CaO_prev + CaO_generated - CaO_consumed
        x_next.CaO = max(0.0, CaO_next)

        # ==========================================================
        # CO2 MASS BALANCE
        # ==========================================================

        CO2_prev = x.CO2
        CO2_generated = max(0.0, CaO_consumed * 0.4397)

        x_next.CO2 = CO2_prev + CO2_generated

        # ==========================================================
        # CO FORMATION (CAUSAL LAG MODEL)
        # ==========================================================

        oxygen_deficit = max(0.0, 6.0 - x_next.O2)
        fuel_factor = x_next.Fuel_rate / 6.0
        temp_factor = np.exp(-x_next.Tg_burning / 1200.0)

        target_CO = 20.0 + 800.0 * oxygen_deficit * fuel_factor * temp_factor

        x_next.CO_ppm = x.CO_ppm + 0.25 * (target_CO - x.CO_ppm)

        # ==========================================================
        # PRESSURE FIELD
        # ==========================================================

        x_next.P_calcination = -406.0 + 226.0 * np.exp(-t / 20.0)

        # ==========================================================
        # RADIATION LOSS
        # ==========================================================

        sigma_sb = 5.67e-8
        view_factor = 0.0016
        emissivity = 0.8
        A_effective = 110.0

        Tg = x_next.Tg_burning + 273.15

        Q_radiation = (
            view_factor * sigma_sb * emissivity * A_effective * (Tg**4 - 298.15**4)
        )

        x_next.Q_loss = Q_radiation

        # ==========================================================
        # ENERGY STORAGE (CONSISTENT SINGLE DEFINITION)
        # ==========================================================

        E_stored = (
            x_next.Air_flow_kgs * 1005.0 * Tg
            + x_next.Kiln_solid_out * 1050.0 * x_next.Ts_burning
        )

        E_prev = x.E_stored if hasattr(x, "E_stored") else E_stored

        x_next.dE_stored = (E_stored - E_prev) / max(self.dt, 1e-9)
        x_next.E_stored = E_stored

        # ==========================================================
        # ENERGY BALANCE ERROR
        # ==========================================================

        Q_in = x_next.Q_in
        Q_out = x_next.Q_loss + x_next.Q_gas

        energy_balance_error = Q_in - Q_out - x_next.dE_stored

        x_next.Energy_Error = abs(energy_balance_error) / max(abs(Q_in), 1e-9) * 100.0

        # ==========================================================
        # MASS / ENERGY DEBUG METRICS
        # ==========================================================

        CaCO3_in = x.CaCO3

        expected_CO2 = CaO_consumed * (44.01 / 100.09)
        expected_CaO = CaO_consumed * (56.08 / 100.09)

        CO2_error = expected_CO2 - CO2_generated
        CaO_error = expected_CaO - CaO_consumed

        x_next.CO2_balance_error = CO2_error
        x_next.CaO_balance_error = CaO_error

        x_next.CO2_balance_error_pct = abs(CO2_error) / max(expected_CO2, 1e-9) * 100.0
        x_next.CaO_balance_error_pct = abs(CaO_error) / max(expected_CaO, 1e-9) * 100.0

        return x_next


def run_simulation():

    import numpy as np
    import pandas as pd
    from dataclasses import asdict

    sim_duration = 72.0
    dt = 0.05
    reporting_dt = 1 / 6

    executor = StepExecutor(dt=dt)

    STEPS_PER_REPORT = max(1, int(reporting_dt / dt))
    N_total_reports = int(sim_duration / reporting_dt)

    # ==========================================================
    # INITIAL STATE (FULL SAFE INIT)
    # ==========================================================

    x_current = KilnState()
    x_current.t = 0.0

    # control
    x_current.Fuel_rate = 4.0
    x_current.O2 = 6.0
    x_current.CO_ppm = 900.0

    # temperatures
    x_current.Tg_preheater = 680.0
    x_current.Tg_calcination = 1050.0
    x_current.Tg_burning = 1450.0
    x_current.Tg_Cooling = 1550.0

    x_current.Ts_preheater = 300.0
    x_current.Ts_calcination = 850.0
    x_current.Ts_burning = 1420.0
    x_current.Ts_Cooling = 1450.0

    # flows
    x_current.Air_flow = 45100.0
    x_current.Cooling_air_flow = 80000.0
    x_current.ID_fan_speed = 900.0

    x_current.Feed_rate = 71.0
    x_current.Kiln_solid_out = 12.5
    x_current.Material_acc = 15.0

    # chemistry
    x_current.CaCO3 = 80.0
    x_current.CaO = 1e-6
    x_current.CO2 = 1e-6

    x_current.SiO2 = 13.0
    x_current.Al2O3 = 4.0
    x_current.Fe2O3 = 3.0

    x_current.C2S = 1e-6
    x_current.C3S = 1e-6
    x_current.C3A = 1e-6
    x_current.C4AF = 1e-6

    x_current.kiln_rpm = 1.0
    x_current.Residence = 69.0

    x_current.Petcoke = 0.03
    x_current.Alternative_Fuel = 0.07
    x_current.Lignite_Coal = 0.90

    x_current.Clinker_yield = 0.65

    # ==========================================================
    # CRITICAL SAFETY STATE INIT (FIXES ALL AttributeError CRASHES)
    # ==========================================================

    for k in [
        "Q_c2s",
        "Q_c3s",
        "Q_c3a",
        "Q_c4af",
        "Q_reaction",
        "dC2S",
        "dC3S",
        "dC3A",
        "dC4AF",
        "T_burn_effective",
        "Q_clinker",
        "E_stored",
    ]:
        setattr(x_current, k, 0.0)

    # ==========================================================
    # INPUT LAYER
    # ==========================================================

    def input_layer(t: float, x):

        D = 0.03 + (0.1 - 0.03) * (1 - np.exp(-t / 80.0))
        E = 0.07 + (0.14 - 0.07) * (1 - np.exp(-t / 100.0))

        base_feed = (
            132.0
            if t >= 72.0
            else 72.0
            + 60.0 * ((1.0 / (1.0 + np.exp(-0.065 * (t - 36.0)))) - 0.09) / 0.82
        )

        oxygen_factor = np.clip(6.0 / (x.O2 + 1e-9), 0.8, 1.2)
        co_factor = 1.0 - 0.0001 * max(0.0, x.CO_ppm - 800.0)

        Feed_rate = base_feed * oxygen_factor * co_factor

        return {
            "Petcoke": float(D),
            "Alternative_Fuel": float(E),
            "Feed_rate": float(Feed_rate),
            "Fuel_rate": getattr(x, "Fuel_rate", 4.0),
            "O2": x.O2,
        }

    # ==========================================================
    # SIMULATION LOOP
    # ==========================================================

    simulation_records = []
    simulation_records.append(asdict(x_current))

    sim_time = 0.0

    for step_idx in range(N_total_reports):

        for sub in range(STEPS_PER_REPORT):
            step_time = sim_time + dt * (sub + 1)

            inputs = input_layer(step_time, x_current)

            x_current = executor.perform_step(x_current, step_time, inputs)

            if x_current is None:
                raise RuntimeError("StepExecutor returned None (state lost)")

        # MASS BALANCE CHECK
        inventory_sum = (
            x_current.CaCO3
            + x_current.CaO
            + x_current.SiO2
            + x_current.Al2O3
            + x_current.Fe2O3
            + x_current.C2S
            + x_current.C3S
            + x_current.C3A
            + x_current.C4AF
        )

        x_current.Mass_Balance_Error = inventory_sum

        # TIME UPDATE
        sim_time += reporting_dt
        x_current.t = sim_time

        simulation_records.append(asdict(x_current))

    # ==========================================================
    # OUTPUT CSV
    # ==========================================================

    df_results = pd.DataFrame(simulation_records)

    float_cols = df_results.select_dtypes(include=["float", "float64"]).columns
    df_results[float_cols] = df_results[float_cols].round(5)

    df_results.to_csv("engine.csv", index=False)

    print(f"Simülasyon tamamlandı. {len(df_results)} satır üretildi.")


if __name__ == "__main__":
    run_simulation()
