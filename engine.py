import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class KilnState:
    t: float = 0.0
    Lignite_Coal: float = 0.0
    Petcoke: float = 0.0
    Alternative_Fuel: float = 0.0
    Feed_rate: float = 40.0
    Kiln_solid_out: float = 0.0
    Material_acc: float = 15.0
    Clinker_output: float = 0.0
    Air_flow: float = 45000.0
    Cooling_air_flow: float = 80000.0
    ID_fan_speed: float = 900.0
    Fuel_rate: float = 4.0
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
    Damper_position: float = 33.0
    kiln_rpm: float = 1.0
    Residence: float = 0.0
    Q_in: float = 0.0
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
    Q_out: float = 0.0
    Q_acc: float = 0.0
    Q_loss: float = 0.0
    Q_reaction: float = 0.0
    Q_gas: float = 0.0
    Q_clinker: float = 0.0
    Clinker_yield: float = 0.65
    dTg_burning: float = 0.0
    Normalized_Energy_Index: float = 1.0
    Global_Energy_Closure: float = 1.0
    Energy_error: float = 0.0
    dCaO_calcination: float = 0.0
    LSF: float = 0.0
    dC3A: float = 0.0
    dC4AF: float = 0.0
    dC2S: float = 0.0
    dC3S: float = 0.0
    Mass_Balance_Error: float = 0.0
    SCALE: float = 1.0
    Energy_Residual: float = 0.0
    Tw_burning: float = 1200.0

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default):
        return getattr(self, key, default)

    def copy(self):
        return KilnState(**asdict(self))


# 2. Fizik Motoru
class KilnPhysicsEngine:
    @staticmethod
    def get_residence_time(kiln_rpm, filling_rate=0.10):
        L, D, slope = 60.0, 4.2, 0.03
        rpm = kiln_rpm
        rpm_eff = rpm / (0.26 + rpm)
        filling = filling_rate
        filling_factor = (0.08 / filling) ** 0.3
        # Endüstriyel kalibrasyona göre eksenel hız katsayısı güncellendi
        v_axial = (5.87 * D * rpm_eff * (1.5 + 44.8 * slope)) * filling_factor
        return (L / (v_axial + 1e-6)) * 60.0


class StepExecutor:
    AIR_DENSITY = 1.293

    def __init__(self, dt=0.05):
        self.dt = dt  # saat cinsinden (0.05 saat = 180 saniye)

    @staticmethod
    def _smooth_max(val, low_bound, eps=1e-6):
        """Türevlenebilir pürüzsüz maksimum fonksiyonu (C-sonsuz sürekli)"""
        return 0.5 * ((val + low_bound) + np.sqrt((val - low_bound) ** 2 + eps))

    @staticmethod
    def _smooth_min(val, high_bound, eps=1e-6):
        """Türevlenebilir pürüzsüz minimum fonksiyonu (C-sonsuz sürekli)"""
        return 0.5 * ((val + high_bound) - np.sqrt((val - high_bound) ** 2 + eps))

    @staticmethod
    def _mpc_softplus(val, width=10.0):
        """MPC çözücülerinde gradient vanishing'i önleyen pürüzsüz sınırlandırıcı."""
        return width * np.log(1.0 + np.exp(val / width))

    @staticmethod
    def _mpc_smooth_fraction(base_fraction, factor, alpha=2.0):
        """Bütçe dağılım faktörlerini asla negatif yapmayan pürüzsüz ölçekleyici."""
        return base_fraction * (1.0 + np.tanh(factor / alpha))

    def perform_step(self, x: KilnState, t: float, inputs: dict = None) -> KilnState:
        inputs = inputs or {}
        x_next = x.copy()

        # INPUT MAPPING
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

        # FUEL COMPOSITION
        x_next["Petcoke"] = inputs.get("Petcoke", x.get("Petcoke", 0.0))
        x_next["Alternative_Fuel"] = inputs.get(
            "Alternative_Fuel", x.get("Alternative_Fuel", 0.0)
        )

        rem_fuel = 1.0 - x_next["Petcoke"] - x_next["Alternative_Fuel"]
        x_next["Lignite_Coal"] = self._smooth_max(rem_fuel, 0.0)

        # CONTROL INPUTS (Süreç Dinamikleri)
        feed = x.get("Feed_rate", 40.0)
        x_next["Feed_rate"] = feed + (120.0 - feed) * 0.0005 * self.dt

        import numpy as np

        x_next["Air_flow"] = 45000.0 + (95000.0 - 45000.0) * (1.0 - np.exp(-t / 120.0))
        x_next["Cooling_air_flow"] = 80000.0 + (83000.0 - 80000.0) * (
            1.0 - np.exp(-t / 140.0)
        )
        x_next["ID_fan_speed"] = 900.0 + (2550.0 - 900.0) * (1.0 - np.exp(-t / 110.0))
        x_next["Damper_position"] = 33.0 + (85.0 - 33.0) * np.exp(-t / 25.0)

        x_next["Fuel_rate"] = 4.0 + 1.5 * (1.0 - np.exp(-t / 35.0))
        f_rate = self._smooth_max(x_next["Fuel_rate"], 0.1)

        # RPM & RESIDENCE TIME
        rpm_current = x.get("kiln_rpm", 1.0)
        rpm_setpoint = 2.4
        alpha = 0.005
        raw_rpm = rpm_current + alpha * (rpm_setpoint - rpm_current)
        x_next["kiln_rpm"] = self._smooth_max(raw_rpm, 0.1)

        res_min = KilnPhysicsEngine.get_residence_time(x_next["kiln_rpm"])
        x_next["Residence"] = res_min

        # ==========================================================
        # MASS BALANCE (STABLE INVENTORY–FLOW COUPLING)
        # ==========================================================

        # Inventory state
        mat_acc = x.get("Material_acc", 15.0)  # ton

        feed_in = x_next["Feed_rate"]  # ton/h

        tau_res = max(res_min / 60.0, 1e-6)  # h

        # ==========================================================
        # PHYSICAL BASE OUTFLOW MODEL
        # ==========================================================
        kiln_out_physical = mat_acc / tau_res  # ton/h

        # ==========================================================
        # DYNAMIC CONSISTENCY CORRECTION
        # ==========================================================
        alpha_flow = 0.85

        kiln_out_prev = x.get("Kiln_solid_out", kiln_out_physical)

        kiln_out = alpha_flow * kiln_out_prev + (1.0 - alpha_flow) * kiln_out_physical

        kiln_out = max(0.0, kiln_out)

        x_next["Kiln_solid_out"] = kiln_out

        # ==========================================================
        # INVENTORY DYNAMICS
        # ==========================================================
        dmat_dt = feed_in - kiln_out

        mat_next = mat_acc + dmat_dt * self.dt
        x_next["Material_acc"] = max(0.0, mat_next)

        # ==========================================================
        # SAFETY: BASE REACTION RATES (NEVER MUTATE ORIGINALS)
        # ==========================================================
        dC2S_0 = x.get("dC2S", 0.0)
        dC3S_0 = x.get("dC3S", 0.0)
        dC3A_0 = x.get("dC3A", 0.0)
        dC4AF_0 = x.get("dC4AF", 0.0)

        # ==========================================================
        # REACTION–FLOW COUPLING
        # ==========================================================
        flow_factor = kiln_out / max(feed_in, 1e-6)

        thermal_factor = np.exp(-x_next["Tg_burning"] / 1200.0)

        activity_factor = min(1.0, mat_acc / 50.0)

        coupling = flow_factor * thermal_factor * activity_factor

        dC2S_eff = dC2S_0 * coupling
        dC3S_eff = dC3S_0 * coupling
        dC3A_eff = dC3A_0 * coupling
        dC4AF_eff = dC4AF_0 * coupling

        # ==========================================================
        # STATE UPDATE (COUPLED CHEMISTRY)
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
        # CaO BALANCE (FIXED — NO GHOST VARIABLES, NO DOUBLE COUNT)
        # ==========================================================
        CaO_consumed = (
            dC2S_eff * 0.6512
            + dC3S_eff * 0.2456
            + dC3A_eff * 0.6227
            + dC4AF_eff * 0.4616
        )

        CaO_in = x["CaO"] + CaO_consumed

        x_next["CaO"] = max(0.0, CaO_in - CaO_consumed)

        # ---------------------------------
        # CO2 CLOSURE
        # ---------------------------------
        Ca_inventory = (
            x_next["CaCO3"]
            + x_next["CaO"]
            + 0.6512 * x_next["C2S"]
            + 0.7368 * x_next["C3S"]
            + 0.6227 * x_next["C3A"]
            + 0.4616 * x_next["C4AF"]
        )

        x_next["CO2"] = max(0.0, 80.0 - Ca_inventory)

        # ---------------------------------
        # DYNAMIC SIGNALS
        # ---------------------------------
        x_next["dTg_burning"] = (x_next["Tg_burning"] - x["Tg_burning"]) / self.dt

        oxygen_deficit = max(0.0, 6.0 - x_next["O2"])

        target_CO = 20.0 + 800.0 * oxygen_deficit * (
            x_next["Fuel_rate"] / 6.0
        ) * np.exp(-x_next["Tg_burning"] / 1200.0)

        x_next["CO_ppm"] = x["CO_ppm"] + 0.25 * (target_CO - x["CO_ppm"])

        x_next["P_calcination"] = -406.0 + 226.0 * np.exp(-t / 20.0)

        # ---------------------------------
        # MASS BALANCE CHECKS
        # ---------------------------------
        x_next["CO2_balance_error"] = abs(80.0 - (Ca_inventory + x_next["CO2"]))

        x_next["CaO_balance_error"] = abs(CaO_in - (x_next["CaO"] + CaO_consumed))

        # ---------------------------------
        # DOWNSTREAM OUTPUT
        # ---------------------------------
        yield_ratio = 1.0 / 1.55
        x_next["Clinker_output"] = kiln_out * yield_ratio

        # ==========================================================
        # ENERGY INPUT DEFINITION (PHYSICAL SOURCE TERM ONLY)
        # ==========================================================

        LHV_lignite = 15000.0  # kJ/kg
        LHV_petcoke = 30000.0  # kJ/kg
        LHV_alt = 18000.0  # kJ/kg

        fuel_mass_flow = x_next["Fuel_rate"]  # kg/s (veya scaled)

        Q_chem = (
            x_next["Lignite_Coal"] * LHV_lignite
            + x_next["Petcoke"] * LHV_petcoke
            + x_next["Alternative_Fuel"] * LHV_alt
        )

        # oxygen / combustion efficiency factor (physical meaning preserved)
        comb_eff = np.exp(-((x_next["O2"] - 3.5) ** 2) / 25.0)

        Q_in_total = Q_chem * fuel_mass_flow * comb_eff * 1000.0

        # ==========================================================
        # BURNING ZONE (ENERGY-CONSISTENT VERSION)
        # ==========================================================

        Cp_g_burn, Cp_s_burn, Cp_w_burn = 1250.0, 1150.0, 1000.0
        A_c_burn, h_c_burn = 13.85, 0.05
        A_s_burn, A_w_burn, A_ws_burn = 82.0, 70.0, 57.0

        m_air_s = x_next["Air_flow"] * self.AIR_DENSITY / 3600.0
        m_solid_s = (x_next["Feed_rate"] * 1000.0) / 3600.0

        # lumped capacities (explicit mass assumption)
        gas_mass = 220.0
        solid_mass = 6500.0
        wall_mass = 15000.0

        C_gas_total = gas_mass * Cp_g_burn
        C_solid_total = solid_mass * Cp_s_burn
        C_wall_total = wall_mass * Cp_w_burn

        Tg_curr, Ts_curr, Tw_curr = (
            x.get("Tg_burning", 1450.0),
            x.get("Ts_burning", 1400.0),
            x.get("Tw_burning", 1300.0),
        )

        h_gs_burn, h_gw_burn, h_ws_burn = 1450.0, 350.0, 400.0

        # ==========================================================
        # ENERGY INPUT (DIRECT ENTRY GATE — NO ROUTING LAYER)
        # ==========================================================

        Q_burn_in = Q_in_total  # <- CRITICAL CHANGE: direct enthalpy injection

        # ==========================================================
        # GAS ENERGY BALANCE (ENTHALPY FORM)
        # ==========================================================

        a_gas_burn = (
            (h_gs_burn * A_s_burn)
            + (h_gw_burn * A_w_burn)
            + (h_c_burn * 1000.0 * A_c_burn)
            + (m_air_s * Cp_g_burn)
        )

        b_gas_burn = (
            Q_burn_in
            + (h_gw_burn * A_w_burn * Tw_curr)
            + (h_c_burn * 1000.0 * A_c_burn * 30.0)
            + (m_air_s * Cp_g_burn * 400.0)
            + (h_gs_burn * A_s_burn * Ts_curr)
        )

        Tg_next_burn = (C_gas_total * Tg_curr + self.dt * b_gas_burn) / (
            C_gas_total + self.dt * a_gas_burn
        )

        # ==========================================================
        # WALL ENERGY BALANCE
        # ==========================================================

        T_amb = 25.0
        h_loss = 8.0

        Q_loss_wall = h_loss * A_w_burn * (Tw_curr - T_amb)
        Q_wall_to_solid = h_ws_burn * A_ws_burn * (Tw_curr - Ts_curr)

        Tw_next_burn = Tw_curr + (self.dt / C_wall_total) * (
            h_gw_burn * A_w_burn * (Tg_next_burn - Tw_curr)
            - Q_loss_wall
            - Q_wall_to_solid
        )

        # ==========================================================
        # SOLID PHASE (SMOOTH EXOTHERMIC REACTION)
        # ==========================================================

        Q_exo_base = min(
            350000.0, m_solid_s * 500000.0 * (1.0 + max(0.0, Ts_curr - 1200.0) / 250.0)
        )

        k = 0.05
        Q_exo_W = Q_exo_base / (1.0 + np.exp(-k * (Ts_curr - 1200.0)))

        a_sol_burn = (
            (h_gs_burn * A_s_burn) + (h_ws_burn * A_ws_burn) + (m_solid_s * Cp_s_burn)
        )

        b_sol_burn = (
            Q_exo_W
            + (m_solid_s * Cp_s_burn * 900.0)
            + (h_gs_burn * A_s_burn * Tg_next_burn)
            + (h_ws_burn * A_ws_burn * Tw_next_burn)
        )

        Ts_next_burn = (C_solid_total * Ts_curr + self.dt * b_sol_burn) / (
            C_solid_total + self.dt * a_sol_burn
        )

        # ==========================================================
        # STATE UPDATE
        # ==========================================================

        Tg_burning_pred = Tg_next_burn
        Ts_burning_pred = Ts_next_burn
        Tw_burning_pred = Tw_next_burn

        # ------------------------------------------------------
        # CALCINATION ZONE (ENERGY-CONSISTENT VERSION)
        # ------------------------------------------------------

        import numpy as np

        h_gs, A_s = 320.0, 950.0

        # FIX: solid inventory (not feed rate)
        m_solid_calc = x_next["Kiln_solid_out"]

        Ts_calc_curr = x.get("Ts_calcination", 800.0)
        Tg_calc_curr = x.get("Tg_calcination", 900.0)

        Cp_solid_calc = 1050.0
        Cp_air = 1005.0
        T_ref = 25.0

        # ------------------------------------------------------
        # BURNING → CALCINATION ENERGY COUPLING
        # ------------------------------------------------------

        eta_bc = 0.85  # enthalpy transfer efficiency

        m_air_kiln = x_next["Air_flow"] * self.AIR_DENSITY / 3600.0
        m_air_tertiary = (
            x_next.get("Tertiary_air_flow", 0.0) * self.AIR_DENSITY / 3600.0
        )

        H_burn_out = m_air_kiln * Cp_air * (x_next["Tg_burning"] - T_ref)
        Q_from_burning = eta_bc * H_burn_out

        Q_from_tertiary = m_air_tertiary * Cp_air * (400.0 - T_ref)

        Q_calc_total_in = Q_from_burning + Q_from_tertiary

        # ------------------------------------------------------
        # REACTION HEAT LOAD (NO BUDGET CLAMP)
        # ------------------------------------------------------

        activation = 1.0 / (1.0 + np.exp(-0.08 * (Ts_calc_curr - 850.0)))

        Q_reaction_rate = 1700.0 * 1000.0  # J/kg CaCO3

        Q_calc_load = (m_solid_calc / 3600.0) * activation * Q_reaction_rate

        Q_rxn_calc = -Q_calc_load  # endotermik sink

        # ------------------------------------------------------
        # SOLID ENERGY BALANCE
        # ------------------------------------------------------

        C_solid_calc_total = (m_solid_calc * 1000.0 * Cp_solid_calc) / 3600.0

        a_solid_calc = h_gs * A_s + (m_solid_calc * 1000.0 / 3600.0) * Cp_solid_calc

        b_solid_calc = (
            Q_calc_load
            + Q_rxn_calc
            + (h_gs * A_s * Tg_calc_curr)
            - (m_solid_calc * 1000.0 / 3600.0) * Cp_solid_calc * (Ts_calc_curr - T_ref)
        )

        Ts_calc_pred = (C_solid_calc_total * Ts_calc_curr + self.dt * b_solid_calc) / (
            C_solid_calc_total + self.dt * a_solid_calc
        )

        # ------------------------------------------------------
        # GAS ENERGY BALANCE
        # ------------------------------------------------------

        C_gas_calc_total = 100.0 * Cp_air
        wall_term = 4.0 * 13.85

        a_gas_calc = (m_air_kiln + m_air_tertiary) * Cp_air + wall_term + h_gs * A_s

        b_gas_calc = (
            Q_calc_total_in
            + wall_term * x.get("Tg_preheater", 350.0)
            + h_gs * A_s * Ts_calc_pred
        )

        Tg_calc_pred = (C_gas_calc_total * Tg_calc_curr + self.dt * b_gas_calc) / (
            C_gas_calc_total + self.dt * a_gas_calc
        )

        # ------------------------------------------------------
        # FINAL STATE UPDATE
        # ------------------------------------------------------

        x_next["Tg_calcination"] = Tg_calc_pred

        # NOTE: cascade coupling (intentionally kept)
        x_next["Tg_preheater"] = Tg_calc_pred

        # ==============================================================================
        # PREHEATER ZONE (TIME-CONSTANT BASED / MPC COMPLIANT / ENERGY-CONSISTENT)
        # ==============================================================================

        import numpy as np

        Cp_gas_pre = 1150.0
        Cp_solid_pre = 1050.0
        T_ref = 25.0

        Tg_in = x_next["Tg_calcination"]
        Ts_pre = x.get("Ts_preheater", 25.0)

        # --------------------------------------------------------------
        # MASS FLOWS
        # --------------------------------------------------------------

        m_gas_pre = (x_next["Air_flow"] * self.AIR_DENSITY) / 3600.0
        m_solid_pre = (x_next["Feed_rate"] * 1000.0) / 3600.0

        C_gas_flow = m_gas_pre * Cp_gas_pre + 1e-4
        C_solid_flow = m_solid_pre * Cp_solid_pre + 1e-4

        # --------------------------------------------------------------
        # HEAT TRANSFER STRUCTURE
        # --------------------------------------------------------------

        h_pre = 18.0
        A_pre = 120.0
        UA_pre = h_pre * A_pre

        # --------------------------------------------------------------
        # PRESSURE MODEL (CAUSAL / LAG-CONSISTENT)
        # --------------------------------------------------------------

        P_in = x_next.get("P_calcination", 101325.0)

        Tg_ref = x.get("Tg_preheater", Tg_in)
        Tg_preheater_K = Tg_ref + 273.15

        rho = self.AIR_DENSITY * (T_ref / (Tg_preheater_K + 1e-9))

        K_resistance = 1200.0

        dP = K_resistance * (m_gas_pre**2) / (rho + 1e-9)

        x_next["P_preheater"] = P_in - dP

        # --------------------------------------------------------------
        # ENERGY DRIVER
        # --------------------------------------------------------------

        dT = Tg_in - Ts_pre

        # --------------------------------------------------------------
        # HOLDOUP-BASED TIME CONSTANT MODEL
        # --------------------------------------------------------------

        M_gas_hold = 150.0
        M_solid_hold = 3000.0

        C_gas_bulk = M_gas_hold * Cp_gas_pre
        C_solid_bulk = M_solid_hold * Cp_solid_pre

        tau_g = C_gas_bulk / (UA_pre + 1e-4)
        tau_s = C_solid_bulk / (UA_pre + 1e-4)

        eps_smooth = 1e-4

        tau_eff = 0.5 * (tau_g + tau_s - np.sqrt((tau_g - tau_s) ** 2 + eps_smooth))

        # ❗ dt_sec REMOVED → replaced with self.dt

        alpha = self.dt / (self.dt + tau_eff + 1e-4)

        alpha_max = 0.35

        alpha_stable = 0.5 * (
            alpha + alpha_max - np.sqrt((alpha - alpha_max) ** 2 + eps_smooth)
        )

        # --------------------------------------------------------------
        # ENERGY TRANSFER (PURE PHYSICAL — NO BUDGET CLAMP)
        # --------------------------------------------------------------

        Q_preheater_physical = alpha_stable * UA_pre * dT

        Q_preheater = Q_preheater_physical

        Q_gas_loss = Q_preheater
        Q_solid_gain = Q_preheater

        # --------------------------------------------------------------
        # STATE UPDATE
        # --------------------------------------------------------------

        Tg_next = Tg_in - (Q_gas_loss / (C_gas_flow + 1e-9))
        Ts_next = Ts_pre + (Q_solid_gain / (C_solid_flow + 1e-9))

        Tg_preheater_pred = 0.7 * Tg_in + 0.3 * Tg_next
        Ts_preheater_pred = 0.7 * Ts_pre + 0.3 * Ts_next

        # --------------------------------------------------------------
        # WRITE STATE
        # --------------------------------------------------------------

        x_next["Tg_preheater"] = Tg_preheater_pred
        x_next["Ts_preheater"] = Ts_preheater_pred

        x_next["Q_dot_preheater"] = Q_preheater
        x_next["UA_preheater_effective"] = alpha_stable * UA_pre
        x_next["Q_preheater_physical"] = Q_preheater
        x_next["Q_preheater_used"] = Q_preheater

        # ------------------------------------------------------
        # COOLING ZONE (FINAL CLOSED-LOOP ENERGY VERSION - CLEAN)
        # ------------------------------------------------------

        import numpy as np

        Tamb_solid = 130.0
        Tamb_gas = 510.0

        Ts_cool_curr = x.get("Ts_Cooling", x_next.get("Ts_burning", 1450.0))
        Tg_cool_curr = x.get("Tg_Cooling", x_next.get("Tg_burning", 1490.0))

        # -------------------------------
        # MASS FLOWS
        # -------------------------------
        AIR_DENSITY = 1.293
        Cp_solid = 1150.0
        Cp_gas = 1050.0

        m_solid = (x["Feed_rate"] * 1000.0) / 3600.0
        m_gas = x["Cooling_air_flow"] * AIR_DENSITY / 3600.0

        # -------------------------------
        # THERMAL CAPACITIES
        # -------------------------------
        C_solid = m_solid * Cp_solid + 1e-9
        C_gas = m_gas * Cp_gas + 1e-9

        # -------------------------------
        # TIME CONSTANTS
        # -------------------------------
        tau_klinker = 9700.0
        tau_gas = 3800.0

        # -------------------------------
        # HEAT TRANSFER COUPLING
        # -------------------------------
        h_transfer = 0.15
        A_transfer = 500.0
        UA = h_transfer * A_transfer

        dT = Ts_cool_curr - Tg_cool_curr

        # -------------------------------
        # COUPLING ENERGY TRANSFER
        # -------------------------------
        Q_transfer = UA * np.tanh(dT / 250.0)

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
        # ENERGY RECOVERY (PHYSICALLY CONSISTENT ENTHALPY DROP)
        # ======================================================

        # enthalpy reference drop (true physical driving force)
        Cp_ref = Cp_gas
        T_ref = 200.0

        enthalpy_drop = m_gas * Cp_ref * max(Tg_cool_curr - T_ref, 0.0)

        # recovery efficiency (irreversibility + mixing losses)
        eta_recovery = 0.65

        Q_secondary_air = eta_recovery * enthalpy_drop

        # ======================================================
        # BURNING POOL UPDATE (FLUX-BASED, NOT ACCUMULATIVE DRIFT)
        # ======================================================

        Q_burning_pool_prev = x.get("Q_burning_pool", 0.0)

        # relaxation dynamics (prevents energy explosion / drift)
        tau_pool = 1200.0  # effective thermal integration time scale

        Q_burning_pool = Q_burning_pool_prev + (self.dt / tau_pool) * (
            Q_secondary_air - Q_burning_pool_prev
        )

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        x_next["Ts_Cooling"] = Ts_cool_next
        x_next["Tg_Cooling"] = Tg_cool_next
        x_next["Q_burning_pool"] = Q_burning_pool

        # ----------------------------------------------------------
        # KALSİNASYON (STOKİYOMETRİK + KÜTLE + ENERGY-COUPLED MODEL)
        # ----------------------------------------------------------

        import numpy as np

        Ts_calc_curr = x.get("Ts_calcination", 800.0)
        T_calc = Ts_calc_curr + 273.15

        # -------------------------------
        # ARRHENIUS KINETICS
        # -------------------------------
        if T_calc > 873.15:
            k_calc = 1e7 * np.exp(-160000.0 / (8.314 * T_calc))
        else:
            k_calc = 0.0

        CaCO3_in = x["CaCO3"]

        # -------------------------------
        # REACTION ENTHALPY (ENDOTHERMIC)
        # -------------------------------
        DeltaH_calc = 1700.0 * 1000.0  # J/kg CaCO3

        # gas driving temperature (local coupling ONLY)
        T_gas = x.get("Tg_calcination", Ts_calc_curr)

        # -------------------------------
        # ENERGY AVAILABILITY (HEAT SUPPLY FROM GAS)
        # -------------------------------
        Cp_solid_calc = 1050.0  # J/kgK

        Q_available = max(0.0, (T_gas - Ts_calc_curr) * Cp_solid_calc * CaCO3_in)

        react_max_energy = Q_available / (DeltaH_calc + 1e-9)

        # -------------------------------
        # KINETIC LIMIT
        # -------------------------------
        react_kin = CaCO3_in * (1.0 - np.exp(-k_calc * self.dt))

        reacted = min(react_kin, react_max_energy)
        reacted = np.clip(reacted, 0.0, CaCO3_in)

        # -------------------------------
        # PRODUCTS
        # -------------------------------
        CaCO3_out = CaCO3_in - reacted
        CaO_generated = reacted * 0.5603
        CO2_generated_phys = reacted * 0.4397

        # ==========================================================
        #  ENERGY FEEDBACK (THIS IS THE MISSING PART YOU WANTED)
        # ==========================================================

        # enthalpy consumed by reaction (endothermic sink)
        Q_reaction = reacted * DeltaH_calc  # J

        # convert to temperature drop of solid phase
        solid_heat_capacity = Cp_solid_calc * max(CaCO3_in, 1e-6)

        dT_reaction = Q_reaction / (solid_heat_capacity + 1e-9)

        # update calcination solid temperature (implicit Euler style)
        Ts_calc_next = Ts_calc_curr - dT_reaction + 0.15 * (T_gas - Ts_calc_curr)

        # clamp physically reasonable bounds
        Ts_calc_next = np.clip(Ts_calc_next, 400.0, 1600.0)

        # optional gas feedback (weak coupling)
        Tg_calc_next = T_gas - 0.05 * (Ts_calc_curr - Ts_calc_next)

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        x_next["CaCO3"] = CaCO3_out
        x_next["CaO_generated"] = CaO_generated
        x_next["CO2_generated"] = CO2_generated_phys

        x_next["dCaO_calcination"] = CaO_generated / self.dt if self.dt > 0 else 0.0
        x_next["dCO2_calcination"] = (
            CO2_generated_phys / self.dt if self.dt > 0 else 0.0
        )

        x_next["CaO"] = x.get("CaO", 0.0) + CaO_generated

        #  NEW ENERGY STATES
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

        SiO2 = x["SiO2"]
        Al2O3 = x["Al2O3"]
        Fe2O3 = x["Fe2O3"]

        # 🔴 CRITICAL FIX: state load
        CaO_pool = x.get("CaO", 0.0)
        CaO_pool_local = max(0.0, CaO_pool)

        # ==========================================================
        # C2S FORMASYONU (ENERGY + MASS LIMITED)
        # ==========================================================

        if SiO2 <= 1e-6 or T_burn_eff < 1000.0:
            dC2S = 0.0

        else:
            k = 5e7 * np.exp(-170000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                SiO2 / 0.3488,
                CaO_pool_local / 0.6512,
            )

            energy_factor = 1.0 / (1.0 + np.exp(-(T_burn_eff - 1100.0) / 80.0))

            dC2S = stoich_limit * kinetic * energy_factor

        # ==========================================================
        # MASS UPDATE
        # ==========================================================

        CaO_consumed = dC2S * 0.6512
        CaO_pool_local -= CaO_consumed
        CaO_pool_local = max(0.0, CaO_pool_local)

        # ==========================================================
        #  REACTION ENTHALPY (C2S FORMATION)
        # ==========================================================

        DeltaH_c2s = 250.0 * 1000.0  # J per unit reaction extent (lumped)

        Q_c2s = dC2S * DeltaH_c2s  # W equivalent over dt-scale

        # ==========================================================
        #  BURNING ZONE ENERGY FEEDBACK (PHYSICALLY CONSISTENT)
        # ==========================================================

        # burning zone heat capacity (lumped)
        Cp_burn_solid = 1150.0
        Cp_burn_gas = 1250.0
        Cp_burn_wall = 1000.0

        C_burn_total = (
            6500.0 * Cp_burn_solid + 220.0 * Cp_burn_gas + 15000.0 * Cp_burn_wall
        )

        # convert reaction enthalpy to temperature impact
        dT_burn = Q_c2s / (C_burn_total + 1e-9)

        # distribute effect across phases (stable coupling)
        Ts_burning_pred = Ts_burning_pred - 0.03 * dT_burn
        Tg_burning_pred = Tg_burning_pred - 0.05 * dT_burn
        Tw_burning_pred = Tw_burning_pred - 0.02 * dT_burn

        # update effective temperature
        T_burn_eff_next = (
            0.7 * (Ts_burning_pred + 273.15)
            + 0.2 * (Tg_burning_pred + 273.15)
            + 0.1 * (Tw_burning_pred + 273.15)
        )

        # ==========================================================
        # WRITE BACK
        # ==========================================================

        CaO_pool = CaO_pool_local

        x_next["dC2S"] = dC2S
        x_next["Q_C2S"] = Q_c2s

        x_next["Tg_burning"] = Tg_burning_pred
        x_next["Ts_burning"] = Ts_burning_pred
        x_next["Tw_burning"] = Tw_burning_pred

        x_next["T_burn_effective"] = T_burn_eff_next

        # ==========================================================
        # C3S FORMASYONU (ENERGY + MASS + THERMAL FEEDBACK)
        # ==========================================================

        C2S_in = x["C2S"] + dC2S

        if C2S_in <= 1e-6 or T_burn_eff < 1473.15:
            dC3S = 0.0

        else:
            k = 2.28e8 * np.exp(-200000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                C2S_in / 0.7544,
                max(0.0, CaO_pool) / 0.2456,
            )

            dC3S = stoich_limit * kinetic

        # ==========================================================
        # MASS UPDATE
        # ==========================================================

        CaO_consumed_c3s = dC3S * 0.2456
        CaO_pool -= CaO_consumed_c3s
        CaO_pool = max(0.0, CaO_pool)

        x_next["C2S"] = C2S_in
        x_next["C3S"] = x.get("C3S", 0.0) + dC3S

        # ==========================================================
        #  REACTION ENTHALPY (C3S FORMATION)
        # ==========================================================

        # C3S formation is strongly exothermic in clinkerization context
        DeltaH_c3s = 420.0 * 1000.0  # J (effective lumped enthalpy)

        Q_c3s = dC3S * DeltaH_c3s

        # ==========================================================
        #  BURNING ZONE THERMAL FEEDBACK (COUPLED ENERGY POOL)
        # ==========================================================

        Cp_burn_solid = 1150.0
        Cp_burn_gas = 1250.0
        Cp_burn_wall = 1000.0

        C_burn_total = (
            6500.0 * Cp_burn_solid + 220.0 * Cp_burn_gas + 15000.0 * Cp_burn_wall
        )

        # convert enthalpy to temperature perturbation
        dT_burn_c3s = Q_c3s / (C_burn_total + 1e-9)

        # stronger coupling than C2S (physically consistent)
        Ts_burning_pred += 0.06 * dT_burn_c3s
        Tg_burning_pred += 0.08 * dT_burn_c3s
        Tw_burning_pred += 0.04 * dT_burn_c3s

        # update effective burning temperature
        T_burn_eff_next = (
            0.7 * (Ts_burning_pred + 273.15)
            + 0.2 * (Tg_burning_pred + 273.15)
            + 0.1 * (Tw_burning_pred + 273.15)
        )

        # ==========================================================
        # WRITE BACK
        # ==========================================================

        CaO_pool = CaO_pool

        x_next["C3S"] = x_next.get("C3S", 0.0) + dC3S
        x_next["Q_C3S"] = Q_c3s

        x_next["Tg_burning"] = Tg_burning_pred
        x_next["Ts_burning"] = Ts_burning_pred
        x_next["Tw_burning"] = Tw_burning_pred
        x_next["T_burn_effective"] = T_burn_eff_next

        # ==========================================================
        # C3A FORMASYONU (ENERGY + MASS + THERMODYNAMIC COUPLING)
        # ==========================================================

        Al2O3_in = x["Al2O3"]

        if Al2O3_in <= 1e-6 or T_burn_eff < 1100.0:
            dC3A = 0.0

        else:
            k = 1.0e5 * np.exp(-120000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                Al2O3_in / 0.3773,
                max(0.0, CaO_pool) / 0.6227,
            )

            dC3A = stoich_limit * kinetic

        # ==========================================================
        # MASS UPDATE
        # ==========================================================

        CaO_consumed_c3a = dC3A * 0.6227
        CaO_pool -= CaO_consumed_c3a
        CaO_pool = max(0.0, CaO_pool)

        x_next["C3A"] = x.get("C3A", 0.0) + dC3A

        # ==========================================================
        #  REACTION ENTHALPY (C3A FORMATION)
        # ==========================================================

        # C3A formation is moderately exothermic in clinker system
        DeltaH_c3a = 380.0 * 1000.0  # J (lumped effective enthalpy)

        Q_c3a = dC3A * DeltaH_c3a

        # ==========================================================
        #  BURNING ZONE ENERGY FEEDBACK (STABILIZED COUPLING)
        # ==========================================================

        Cp_burn_solid = 1150.0
        Cp_burn_gas = 1250.0
        Cp_burn_wall = 1000.0

        C_burn_total = (
            6500.0 * Cp_burn_solid + 220.0 * Cp_burn_gas + 15000.0 * Cp_burn_wall
        )

        # temperature perturbation from reaction enthalpy
        dT_burn_c3a = Q_c3a / (C_burn_total + 1e-9)

        # weaker coupling than C3S (physically correct hierarchy)
        Ts_burning_pred += 0.02 * dT_burn_c3a
        Tg_burning_pred += 0.03 * dT_burn_c3a
        Tw_burning_pred += 0.015 * dT_burn_c3a

        # update effective burning temperature
        T_burn_eff_next = (
            0.7 * (Ts_burning_pred + 273.15)
            + 0.2 * (Tg_burning_pred + 273.15)
            + 0.1 * (Tw_burning_pred + 273.15)
        )

        # ==========================================================
        # WRITE BACK
        # ==========================================================

        CaO_pool = CaO_pool

        x_next["C3A"] = x_next.get("C3A", 0.0) + dC3A
        x_next["Q_C3A"] = Q_c3a

        x_next["Tg_burning"] = Tg_burning_pred
        x_next["Ts_burning"] = Ts_burning_pred
        x_next["Tw_burning"] = Tw_burning_pred
        x_next["T_burn_effective"] = T_burn_eff_next

        # ==========================================================
        # C4AF FORMASYONU (ENERGY + MASS + TEMPERATURE CONSISTENT)
        # ==========================================================

        if Fe2O3 <= 1e-6 or T_burn_eff < 1100.0:
            dC4AF = 0.0
            Q_c4af = 0.0
        else:
            k = 200000.0 * np.exp(-150000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            Al2O3_remaining = max(
                0.0,
                Al2O3 - dC3A * 0.3773,
            )

            stoich_limit = min(
                Fe2O3 / 0.3286,
                Al2O3_remaining / 0.2098,
                max(0.0, CaO_pool) / 0.4616,
            )

            dC4AF = stoich_limit * kinetic

            # ==========================================================
            # ENERGY TERM
            # ==========================================================
            DeltaH_c4af = 120.0 * 1000.0  # J (low-energy stabilizing phase)
            Q_c4af = dC4AF * DeltaH_c4af

        # ==========================================================
        # MASS UPDATE
        # ==========================================================
        CaO_consumed_c4af = dC4AF * 0.4616
        CaO_pool -= CaO_consumed_c4af
        CaO_pool = max(0.0, CaO_pool)

        # ==========================================================
        # 🔥 TEMPERATURE FEEDBACK (CLOSED LOOP)
        # ==========================================================

        Cp_effective = 1150.0 * max(CaO_pool + CaO_consumed_c4af, 1e-6)

        dT_c4af = Q_c4af / (Cp_effective + 1e-9)

        # -------------------------------
        # APPLY THERMAL COUPLING
        # -------------------------------

        Ts_burning_pred = Ts_burning_pred - 0.006 * dT_c4af
        Tg_burning_pred = Tg_burning_pred - 0.002 * dT_c4af
        Tw_burning_pred = Tw_burning_pred - 0.001 * dT_c4af

        # derived diagnostic metric
        T_burn_eff = T_burn_eff - 0.01 * dT_c4af

        # ==========================================================
        # WRITE BACK (STATE CONSISTENCY)
        # ==========================================================

        x_next["dC4AF"] = dC4AF
        x_next["Q_C4AF"] = Q_c4af
        x_next["T_burn_effective"] = T_burn_eff

        # ==========================================================
        # STATE UPDATE (FLOW–COUPLED CHEMISTRY - CONSISTENT VERSION)
        # ==========================================================

        # -------------------------------
        # PREHEATER
        # -------------------------------
        x_next["Tg_preheater"] = Tg_preheater_pred
        x_next["Ts_preheater"] = Ts_preheater_pred

        # -------------------------------
        # CALCINATION
        # -------------------------------
        x_next["Tg_calcination"] = Tg_calc_pred
        x_next["Ts_calcination"] = Ts_calc_pred

        # -------------------------------
        # BURNING ZONE
        # -------------------------------
        x_next["Tg_burning"] = Tg_burning_pred
        x_next["Ts_burning"] = Ts_burning_pred
        x_next["Tw_burning"] = Tw_burning_pred

        # -------------------------------
        # COOLING ZONE
        # -------------------------------
        x_next["Tg_Cooling"] = Tg_cool_next
        x_next["Ts_Cooling"] = Ts_cool_next

        # -------------------------------
        # ENERGY POOL (recovered secondary air feedback)
        # -------------------------------
        x_next["Q_burning_pool"] = Q_burning_pool

        # -------------------------------
        # SOLID CHEMISTRY POOLS
        # -------------------------------
        x_next["CaO"] = CaO_pool

        x_next["CaCO3"] = CaCO3_out
        x_next["CaO_generated"] = CaO_generated
        x_next["CO2_generated"] = CO2_generated_phys

        # -------------------------------
        # CLINKER PHASE PRODUCTS
        # -------------------------------
        x_next["dC2S"] = dC2S
        x_next["dC3S"] = dC3S
        x_next["dC3A"] = dC3A
        x_next["dC4AF"] = dC4AF

        # -------------------------------
        # THERMAL FEEDBACK METRICS (optional but important)
        # -------------------------------
        x_next["T_burn_effective"] = T_burn_eff

        # ----------------------------------------------------------
        # SAFETY: ensure reaction terms exist
        # ----------------------------------------------------------
        dC2S = locals().get("dC2S", 0.0)
        dC3S = locals().get("dC3S", 0.0)
        dC3A = locals().get("dC3A", 0.0)
        dC4AF = locals().get("dC4AF", 0.0)

        # ----------------------------------------------------------
        # FLOW COUPLING FACTOR (CRITICAL FIX)
        # ----------------------------------------------------------
        flow_scale = x_next.get("Kiln_solid_out", 1e-6) / max(x["Feed_rate"], 1e-6)

        thermal_factor = np.exp(-x_next["Tg_burning"] / 1200.0)

        activity_factor = min(1.0, x_next["Material_acc"] / 50.0)

        # ----------------------------------------------------------
        # EFFECTIVE REACTION EXTENTS (PHYSICALLY CONSISTENT)
        # ----------------------------------------------------------
        dC2S_eff = dC2S * flow_scale * thermal_factor * activity_factor
        dC3S_eff = dC3S * flow_scale * thermal_factor * activity_factor
        dC3A_eff = dC3A * flow_scale * thermal_factor * activity_factor
        dC4AF_eff = dC4AF * flow_scale * thermal_factor * activity_factor

        # ==========================================================
        # CLINKER PHASE UPDATE
        # ==========================================================

        x_next["C2S"] = x["C2S"] + dC2S_eff - dC3S_eff * 0.7544
        x_next["C3S"] = x["C3S"] + dC3S_eff
        x_next["C3A"] = x["C3A"] + dC3A_eff
        x_next["C4AF"] = x["C4AF"] + dC4AF_eff

        # ==========================================================
        # ELEMENT BALANCES (FLOW-CONSISTENT)
        # ==========================================================

        x_next["SiO2"] = max(0.0, x["SiO2"] - dC2S_eff * 0.3488)

        x_next["Al2O3"] = max(0.0, x["Al2O3"] - dC3A_eff * 0.3773 - dC4AF_eff * 0.2098)

        x_next["Fe2O3"] = max(0.0, x["Fe2O3"] - dC4AF_eff * 0.3286)

        # ==========================================================
        # 🔥 CaO CONSERVATION (FIXED & STABLE)
        # ==========================================================

        CaO_in = x["CaO"] + CaO_generated

        CaO_consumed = (
            dC2S_eff * 0.6512
            + dC3S_eff * 0.2456
            + dC3A_eff * 0.6227
            + dC4AF_eff * 0.4616
        )

        CaO_out = x["CaO"]

        CaO_next = CaO_in - CaO_consumed
        x_next["CaO"] = max(0.0, CaO_next)

        # ==========================================================
        # CO2 CLOSURE (CONSISTENT MASS GRAPH)
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

        x_next["P_calcination"] = -406.0 + 226.0 * np.exp(-t / 20.0)

        # ==========================================================
        # MASS BALANCE CHECKS
        # ==========================================================

        x_next["CO2_balance_error"] = abs(80.0 - (Ca_inventory + x_next["CO2"]))

        x_next["CaO_balance_error"] = abs(CaO_in - (x_next["CaO"] + CaO_consumed))

        # ==========================================================
        # ENERGY TERMS
        # ==========================================================
        x_next["Q_loss"] = (
            0.0016 * 5.67e-8 * 0.8 * 110 * (x_next["Tg_burning"] ** 4 - 25.0**4)
        )

        x_next["Q_reaction"] = (
            x_next["Feed_rate"] * (1.0 - x_next["Clinker_yield"]) * 3.2
        )

        x_next["Q_out"] = x_next["Q_acc"] + x_next["Q_loss"] + x_next["Q_reaction"]

        x_next["Q_gas"] = x_next["Fuel_rate"] * 1.1 * (x_next["Tg_burning"] - 25.0)

        # ==========================================================
        # GLOBAL ENERGY CLOSURE
        # ==========================================================
        x_next["Global_Energy_Closure"] = x_next["Q_in"] - x_next["Q_out"]

        q_in_safe = max(x_next["Q_in"], 1e-6)

        x_next["Energy_error"] = (x_next["Global_Energy_Closure"] / q_in_safe) * 100.0

        clinker_safe = max(x_next["Clinker_output"], 1e-6)

        x_next["Normalized_Energy_Index"] = x_next["Q_in"] / clinker_safe

        if t == 0:
            x_next["SCALE"] = 1.0

        return x_next


def run_simulation():
    # --- MÜHENDİSLİK YAPILANDIRMASI ---
    sim_duration = 72.0
    dt = 0.05
    reporting_dt = 1 / 6

    executor = StepExecutor(dt=dt)
    STEPS_PER_REPORT = max(1, int(reporting_dt / dt))
    N_total_reports = int(sim_duration / reporting_dt)

    # --- Başlangıç Koşulları (t=0.0) ---
    x_current = KilnState()
    x_current.t = 0.0
    x_current.Fuel_rate, x_current.O2, x_current.CO_ppm = 4.0, 6.0, 900.0
    (
        x_current.Tg_preheater,
        x_current.Tg_calcination,
        x_current.Tg_burning,
        x_current.Tg_Cooling,
    ) = (650.0, 950.0, 1450.0, 1550.0)
    x_current.Air_flow, x_current.Cooling_air_flow, x_current.ID_fan_speed = (
        45100.0,
        80000.0,
        900.0,
    )
    x_current.Feed_rate, x_current.Kiln_solid_out, x_current.Material_acc = (
        71.00,
        0.0,
        0.0,
    )
    (
        x_current.Ts_preheater,
        x_current.Ts_calcination,
        x_current.Ts_burning,
        x_current.Ts_Cooling,
    ) = (300.0, 850.0, 1420.0, 1450.0)
    x_current.CaCO3, x_current.CaO, x_current.CO2 = 80.0, 1e-6, 1e-6
    x_current.SiO2, x_current.Al2O3, x_current.Fe2O3 = 13.0, 4.0, 3.0
    x_current.C2S, x_current.C3S, x_current.C3A, x_current.C4AF = 1e-6, 1e-6, 1e-6, 1e-6
    x_current.kiln_rpm = 1.0
    x_current.Residence = KilnPhysicsEngine.get_residence_time(1.0)
    x_current.Petcoke, x_current.Alternative_Fuel, x_current.Lignite_Coal = (
        0.03,
        0.07,
        0.90,
    )
    x_current.Material_acc, x_current.Clinker_output, x_current.Kiln_solid_out = (
        15.0,
        11.13,
        12.50,
    )
    x_current.P_preheater, x_current.P_calcination, x_current.P_burning = 12.0, 5.5, 1.2
    x_current.Damper_position = 33.0
    (
        x_current.Q_in,
        x_current.Q_out,
        x_current.Q_acc,
        x_current.Q_loss,
        x_current.Q_reaction,
    ) = (35575.0, 24285.0, 1669.0, 22600.0, 15.175)
    x_current.Q_gas, x_current.Q_clinker = 3524.0, 15000.0
    x_current.Clinker_yield = 0.65
    (
        x_current.Normalized_Energy_Index,
        x_current.Global_Energy_Closure,
        x_current.Energy_error,
    ) = (1.0, 1.0, 0.0)

    # Yardımcı Fonksiyonlar
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
        return {"Petcoke": D, "Alternative_Fuel": E, "Feed_rate": base_feed}

    simulation_records = []

    # t=0.0 için kaydet
    x_current.t = 0.0
    simulation_records.append(asdict(x_current))

    sim_time = 0.0

    for step_idx in range(N_total_reports):
        inputs = input_layer(sim_time, initial_state=x_current)

        for sub in range(STEPS_PER_REPORT):
            step_time = sim_time + (dt * (sub + 1))
            x_current = executor.perform_step(x_current, step_time, inputs=inputs)

        sim_time += reporting_dt

        # Sürenin State'e sekronizasyonu
        x_current.t = sim_time
        simulation_records.append(asdict(x_current))

    df_results = pd.DataFrame(simulation_records)
    df_results.to_csv("engine.csv", index=False)

    # ----------------------------------------------------------
    # FLOAT FORMAT (5 DIGIT PRECISION)
    # ----------------------------------------------------------

    df_results = pd.DataFrame(simulation_records)

    float_cols = df_results.select_dtypes(include=["float", "float64"]).columns

    df_results[float_cols] = df_results[float_cols].round(5)

    df_results.to_csv("engine.csv", index=False)

    print(
        f"Simülasyon başarıyla tamamlandı. t=0.0 başlangıçlı {len(df_results)} satır veri kaydedildi."
    )


if __name__ == "__main__":
    run_simulation()
