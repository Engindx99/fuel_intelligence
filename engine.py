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

        # MASS BALANCE
        mat_acc = x.get("Material_acc", 15.0)
        kiln_out = mat_acc / (res_min / 60.0 + 1e-6)
        x_next["Kiln_solid_out"] = kiln_out
        x_next["Material_acc"] = (
            mat_acc + (x_next["Feed_rate"] / 1.55 - kiln_out) * self.dt
        )

        # O2 & HEAT INPUT (Q_in)
        air_fuel_ratio = x_next["Air_flow"] / (f_rate + 1e-6)
        o2_target = 2.5 + 4.5 * np.exp(-1.2 / (air_fuel_ratio / 20000.0 + 1e-3))
        x_next["O2"] = x.get("O2", 3.5) + 0.15 * (o2_target - x.get("O2", 3.5))

        x_next["Q_in"] = (
            (
                x_next["Lignite_Coal"] * 15000.0
                + x_next["Petcoke"] * 30000.0
                + x_next["Alternative_Fuel"] * 18000.0
            )
            * x_next["Fuel_rate"]
            * np.exp(-((x_next["O2"] - 3.5) ** 2) / 25.0)
        )

        Q_in_total = x_next["Q_in"] * 1000.0

        x_next["P_preheater"] = -269.0 + 149.0 * np.exp(-t / 15.0)

        # --- ENERJİ DAĞILIMI (Cascade & Source-Based) ---

        AIR_DENSITY = 1.293
        Cp_air = 1005.0
        T_ref = 25.0

        m_gas_pre = x_next["Air_flow"] * AIR_DENSITY / 3600.0
        m_solid_pre = x_next["Feed_rate"]
        Q_in_total = x_next["Q_in"] * 1000.0

        # -------------------------------
        # TERSİYER HAVA (CORRECTED PHYSICS)
        # -------------------------------
        m_air_tertiary = x_next.get("Tertiary_air_flow", 0.0) * AIR_DENSITY / 3600.0

        Q_tertiary_sink = m_air_tertiary * Cp_air * (1400.0 - T_ref)
        Q_tertiary_enthalpy = m_air_tertiary * Cp_air * (400.0 - T_ref)

        feed_factor = x_next["Feed_rate"] / 100.0

        # Enerji havuzu (mass-consistent)
        Q_burning_pool = Q_in_total * 0.56 - Q_tertiary_sink + Q_tertiary_enthalpy

        Q_calcination_budget = Q_in_total * (0.26 * min(feed_factor, 1.2))
        Q_preheater_budget = Q_in_total * 0.18

        dt_sec = self.dt * 3600.0

        # ------------------------------------------------------
        # BURNING ZONE
        # ------------------------------------------------------
        Cp_g_burn, Cp_s_burn, Cp_w_burn = 1250.0, 1150.0, 1000.0
        A_c_burn, h_c_burn = 13.85, 0.05
        A_s_burn, A_w_burn, A_ws_burn = 82.0, 70.0, 57.0

        m_air_s = x_next["Air_flow"] * AIR_DENSITY / 3600.0
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

        # -------------------------------
        # GAS ENERGY BALANCE
        # -------------------------------
        a_gas_burn = (
            (h_gs_burn * A_s_burn)
            + (h_gw_burn * A_w_burn)
            + (h_c_burn * 1000.0 * A_c_burn)
            + (m_air_s * Cp_g_burn)
        )

        b_gas_burn = (
            Q_burning_pool * 0.97  # efficiency corrected (no artificial +10%)
            + (h_gw_burn * A_w_burn * Tw_curr)
            + (h_c_burn * 1000.0 * A_c_burn * 30.0)
            + (m_air_s * Cp_g_burn * 400.0)
            + (h_gs_burn * A_s_burn * Ts_curr)
        )

        Tg_next_burn = (C_gas_total * Tg_curr + dt_sec * b_gas_burn) / (
            C_gas_total + dt_sec * a_gas_burn
        )

        # -------------------------------
        # WALL ENERGY BALANCE (FIXED LOSS MODEL)
        # -------------------------------
        T_amb = 25.0
        h_loss = 8.0

        Q_loss_wall = h_loss * A_w_burn * (Tw_curr - T_amb)
        Q_wall_to_solid = h_ws_burn * A_ws_burn * (Tw_curr - Ts_curr)

        Tw_next_burn = Tw_curr + (dt_sec / C_wall_total) * (
            h_gw_burn * A_w_burn * (Tg_next_burn - Tw_curr)
            - Q_loss_wall
            - Q_wall_to_solid
        )

        # -------------------------------
        # SOLID PHASE (SMOOTH EXOTHERMIC)
        # -------------------------------
        import numpy as np

        Q_exo_base = min(
            350000.0, m_solid_s * 500000.0 * (1.0 + max(0.0, Ts_curr - 1200.0) / 250.0)
        )

        # smooth activation instead of step function
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

        Ts_next_burn = (C_solid_total * Ts_curr + dt_sec * b_sol_burn) / (
            C_solid_total + dt_sec * a_sol_burn
        )

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        Tg_burning_pred = Tg_next_burn
        Ts_burning_pred = Ts_next_burn
        Tw_burning_pred = Tw_next_burn

        # ------------------------------------------------------
        # CALCINATION ZONE (ENERGY-CONSISTENT VERSION)
        # ------------------------------------------------------

        import numpy as np

        h_gs, A_s = 320.0, 950.0

        #  FIX: solid inventory (not feed rate)
        m_solid_calc = x_next["Kiln_solid_out"]

        Ts_calc_curr = x.get("Ts_calcination", 800.0)
        Tg_calc_curr = x.get("Tg_calcination", 900.0)

        Cp_solid_calc = 1050.0
        Cp_air = 1005.0
        T_ref = 25.0

        # -------------------------------
        # REACTION HEAT LOAD
        # -------------------------------

        activation = 1.0 / (1.0 + np.exp(-0.08 * (Ts_calc_curr - 850.0)))

        Q_reaction_rate = 1700.0 * 1000.0  # J/kg CaCO3

        Q_calc_raw = (m_solid_calc / 3600.0) * activation * Q_reaction_rate

        Q_calc_load = min(Q_calc_raw, Q_calcination_budget / max(dt_sec, 1e-6))

        Q_rxn_calc = -Q_calc_load  # endotermik (energy sink)

        # -------------------------------
        # GAS / AIR TERMS
        # -------------------------------

        m_air_kiln = x_next["Air_flow"] * AIR_DENSITY / 3600.0
        m_air_tertiary = x_next.get("Tertiary_air_flow", 0.0) * AIR_DENSITY / 3600.0

        Q_from_burning = m_air_kiln * Cp_air * (x_next["Tg_burning"] - T_ref)
        Q_from_tertiary = m_air_tertiary * Cp_air * (400.0 - T_ref)

        Q_calc_total_in = Q_from_burning + Q_from_tertiary

        # -------------------------------
        # SOLID ENERGY BALANCE
        # -------------------------------

        C_solid_calc_total = (m_solid_calc * 1000.0 * Cp_solid_calc) / 3600.0

        a_solid_calc = h_gs * A_s + (m_solid_calc * 1000.0 / 3600.0) * Cp_solid_calc

        b_solid_calc = (
            Q_calc_load
            + Q_rxn_calc
            + (h_gs * A_s * Tg_calc_curr)
            - (m_solid_calc * 1000.0 / 3600.0) * Cp_solid_calc * (Ts_calc_curr - T_ref)
        )

        Ts_calc_pred = (C_solid_calc_total * Ts_calc_curr + dt_sec * b_solid_calc) / (
            C_solid_calc_total + dt_sec * a_solid_calc
        )

        # -------------------------------
        # GAS ENERGY BALANCE
        # -------------------------------

        C_gas_calc_total = 100.0 * Cp_air
        wall_term = 4.0 * 13.85

        a_gas_calc = (m_air_kiln + m_air_tertiary) * Cp_air + wall_term + h_gs * A_s

        #  FIX: removed double counting of Q_calcination_budget
        b_gas_calc = (
            Q_calc_total_in
            + wall_term * x.get("Tg_preheater", 350.0)
            + h_gs * A_s * Ts_calc_pred
        )

        Tg_calc_pred = (C_gas_calc_total * Tg_calc_curr + dt_sec * b_gas_calc) / (
            C_gas_calc_total + dt_sec * a_gas_calc
        )

        # -------------------------------
        # FINAL STATE UPDATE (CLEAN BOUNDARIES)
        # -------------------------------

        Ts_calcination_pred = Ts_calc_pred
        Tg_calcination_pred = Tg_calc_pred

        x_next["Tg_calcination"] = Tg_calcination_pred

        #  IMPORTANT: proper energy cascade (NO overwrite chain collapse)
        x_next["Tg_preheater"] = Tg_calcination_pred

        # ==============================================================================
        # PREHEATER ZONE (TIME-CONSTANT BASED / MPC COMPLIANT / STRICTLY DIFFERENTIABLE)
        # ==============================================================================

        # Isıl kapasite katsayıları
        Cp_gas_pre = 1150.0  # J/(kg.K)
        Cp_solid_pre = 1050.0  # J/(kg.K)

        Tg_in = x_next["Tg_calcination"]
        Ts_pre = x.get("Ts_preheater", 25.0)

        # Akış Isıl Kapasite Debileri (W/K)
        m_gas_pre = (x_next["Air_flow"] * AIR_DENSITY) / 3600.0
        m_solid_pre = (x_next["Feed_rate"] * 1000.0) / 3600.0

        C_gas_flow = m_gas_pre * Cp_gas_pre + 1e-4
        C_solid_flow = m_solid_pre * Cp_solid_pre + 1e-4

        # Yapısal Isı Transferi
        h_pre = 18.0
        A_pre = 120.0
        UA_pre = h_pre * A_pre

        dT = Tg_in - Ts_pre

        # ------------------------------------------------------------------------------
        # HOLDOUP-BASED TIME CONSTANT MODEL
        # ------------------------------------------------------------------------------

        M_gas_hold = 150.0
        M_solid_hold = 3000.0

        C_gas_bulk = M_gas_hold * Cp_gas_pre
        C_solid_bulk = M_solid_hold * Cp_solid_pre

        tau_g = C_gas_bulk / (UA_pre + 1e-4)
        tau_s = C_solid_bulk / (UA_pre + 1e-4)

        eps_smooth = 1e-4

        tau_eff = 0.5 * (tau_g + tau_s - np.sqrt((tau_g - tau_s) ** 2 + eps_smooth))

        alpha = dt_sec / (dt_sec + tau_eff + 1e-4)

        alpha_max = 0.35
        alpha_stable = 0.5 * (
            alpha + alpha_max - np.sqrt((alpha - alpha_max) ** 2 + eps_smooth)
        )

        # ------------------------------------------------------------------------------
        # ENERGY TRANSFER (CLOSED FORM)
        # ------------------------------------------------------------------------------

        Q_preheater = alpha_stable * UA_pre * dT

        # Gazdan katıya ve katıdan gaza enerji dengesi (kapalı form)
        Q_gas_loss = Q_preheater
        Q_solid_gain = Q_preheater

        # ------------------------------------------------------------------------------
        # STATE UPDATE (ENERGY CONSISTENT DYNAMICS)
        # ------------------------------------------------------------------------------

        # gas cooling
        Tg_next = Tg_in - (Q_gas_loss / (C_gas_flow + 1e-9))

        # solid heating
        Ts_next = Ts_pre + (Q_solid_gain / (C_solid_flow + 1e-9))

        # smoothing (MPC stability)
        Tg_preheater_pred = 0.7 * Tg_in + 0.3 * Tg_next
        Ts_preheater_pred = 0.7 * Ts_pre + 0.3 * Ts_next

        # ------------------------------------------------------------------------------
        # STATE WRITE
        # ------------------------------------------------------------------------------

        x_next["Tg_preheater"] = Tg_preheater_pred
        x_next["Ts_preheater"] = Ts_preheater_pred

        # diagnostics (optional but consistent)
        x_next["Q_dot_preheater"] = Q_preheater
        x_next["UA_preheater_effective"] = alpha_stable * UA_pre
        # ------------------------------------------------------
        # COOLING ZONE (FINAL CLOSED-LOOP ENERGY VERSION)
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
        # TIME CONSTANTS (INERTIAL PHYSICS)
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
        # INTERNAL ENERGY TRANSFER (CONSISTENT FORM)
        # -------------------------------

        # NOTE: dt_sec already represents integration window → no extra scaling needed
        Q_transfer = UA * np.tanh(dT / 250.0)

        # -------------------------------
        # SOLID DYNAMICS (INERTIAL + COUPLED)
        # -------------------------------
        dTs_ambient = (Tamb_solid - Ts_cool_curr) * (
            1.0 - np.exp(-dt_sec / tau_klinker)
        )

        Ts_cool_next = Ts_cool_curr + dTs_ambient - (Q_transfer / (C_solid + 1e-9))

        # -------------------------------
        # GAS DYNAMICS (FLOW + COUPLING)
        # -------------------------------
        dTg_ambient = (Tamb_gas - Tg_cool_curr) * (1.0 - np.exp(-dt_sec / tau_gas))

        # symmetric coupling (energy conservation fix)
        Tg_cool_next = Tg_cool_curr + dTg_ambient + (Q_transfer / (C_gas + 1e-9))

        # -------------------------------
        # ENERGY RECOVERY (CLOSED LOOP CONSISTENCY)
        # -------------------------------
        Q_secondary_air = m_gas * Cp_gas * max(Tg_cool_curr - 200.0, 0.0)

        Q_burning_pool = x_next.get("Q_burning_pool", 0.0) + Q_secondary_air

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        Ts_cooling_pred = Ts_cool_next
        Tg_cooling_pred = Tg_cool_next
        Q_burning_pool_pred = Q_burning_pool

        x_next["Ts_Cooling"] = Ts_cooling_pred
        x_next["Tg_Cooling"] = Tg_cooling_pred
        x_next["Q_burning_pool"] = Q_burning_pool_pred

        # ----------------------------------------------------------
        # KALSİNASYON (STOKİYOMETRİK + KÜTLE KAPALI MODEL)
        # ----------------------------------------------------------

        T_calc = Ts_calcination_pred + 273.15

        if T_calc > 873.15:
            k_calc = 1e7 * np.exp(-160000.0 / (8.314 * T_calc))
        else:
            k_calc = 0.0

        CaCO3_in = x["CaCO3"]

        # Reaction extent
        reacted = CaCO3_in * (1.0 - np.exp(-k_calc * self.dt))
        reacted = np.clip(reacted, 0.0, CaCO3_in)

        # Products
        CaCO3_out = CaCO3_in - reacted
        CaO_generated = reacted * 0.5603
        CO2_generated_phys = reacted * 0.4397

        # STATE UPDATE (CaCO3 only here)
        x_next["CaCO3"] = CaCO3_out
        x_next["CaO_generated"] = CaO_generated
        x_next["CO2_generated"] = CO2_generated_phys

        # rates
        x_next["dCaO_calcination"] = CaO_generated / self.dt if self.dt > 0 else 0.0
        x_next["dCO2_calcination"] = (
            CO2_generated_phys / self.dt if self.dt > 0 else 0.0
        )

        # ==========================================================
        # 🔴 CaO MASS POOL (DOĞRU VE TEK NOKTA)
        # ==========================================================
        CaO_pool = x.get("CaO", 0.0) + CaO_generated

        # ----------------------------------------------------------
        # KALSİNASYON SONRASI GİRİŞLER
        # ----------------------------------------------------------

        Tg_burn = Tg_burning_pred + 273.15
        Ts_burn = Ts_burning_pred + 273.15
        Tw_burn = Tw_burning_pred + 273.15

        T_burn_eff = 0.7 * Ts_burn + 0.2 * Tg_burn + 0.1 * Tw_burn

        SiO2 = x["SiO2"]
        Al2O3 = x["Al2O3"]
        Fe2O3 = x["Fe2O3"]

        # ==========================================================
        # C2S FORMASYONU
        # ==========================================================

        if SiO2 <= 1e-6 or T_burn_eff < 1000.0:
            dC2S = 0.0
        else:
            k = 50000000.0 * np.exp(-170000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                SiO2 / 0.3488,
                max(0.0, CaO_pool) / 0.6512,
            )

            dC2S = stoich_limit * kinetic

        CaO_pool -= dC2S * 0.6512
        CaO_pool = max(0.0, CaO_pool)

        # ==========================================================
        # C3S FORMASYONU
        # ==========================================================

        if x["C2S"] + dC2S <= 1e-6 or T_burn_eff < 1473.15:
            dC3S = 0.0
        else:
            k = 228000000.0 * np.exp(-200000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                (x["C2S"] + dC2S) / 0.7544,
                max(0.0, CaO_pool) / 0.2456,
            )

            dC3S = stoich_limit * kinetic

        CaO_pool -= dC3S * 0.2456
        CaO_pool = max(0.0, CaO_pool)

        # ==========================================================
        # C3A FORMASYONU
        # ==========================================================

        if Al2O3 <= 1e-6 or T_burn_eff < 1100.0:
            dC3A = 0.0
        else:
            k = 100000.0 * np.exp(-120000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            stoich_limit = min(
                Al2O3 / 0.3773,
                max(0.0, CaO_pool) / 0.6227,
            )

            dC3A = stoich_limit * kinetic

        CaO_pool -= dC3A * 0.6227
        CaO_pool = max(0.0, CaO_pool)

        # ==========================================================
        # C4AF FORMASYONU
        # ==========================================================

        if Fe2O3 <= 1e-6 or T_burn_eff < 1100.0:
            dC4AF = 0.0
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

        CaO_pool -= dC4AF * 0.4616
        CaO_pool = max(0.0, CaO_pool)

        # ==========================================================
        # STATE UPDATE
        # ==========================================================

        x_next["C2S"] = x["C2S"] + dC2S - dC3S * 0.7544
        x_next["C3S"] = x["C3S"] + dC3S
        x_next["C3A"] = x["C3A"] + dC3A
        x_next["C4AF"] = x["C4AF"] + dC4AF

        x_next["SiO2"] = max(0.0, x["SiO2"] - dC2S * 0.3488)
        x_next["Al2O3"] = max(0.0, x["Al2O3"] - dC3A * 0.3773 - dC4AF * 0.2098)
        x_next["Fe2O3"] = max(0.0, x["Fe2O3"] - dC4AF * 0.3286)

        # 🔥 FINAL CaO CLOSURE (TEK NOKTA)
        x_next["CaO"] = max(0.0, CaO_pool)

        # ----------------------------------------------------------
        # CO2 CEBİRSEL KAPANIŞI
        # ----------------------------------------------------------

        Ca_inventory = (
            x_next["CaCO3"]
            + x_next["CaO"]
            + 0.6512 * x_next["C2S"]
            + 0.7368 * x_next["C3S"]
            + 0.6227 * x_next["C3A"]
            + 0.4616 * x_next["C4AF"]
        )

        x_next["CO2"] = max(
            0.0,
            80.0 - Ca_inventory,
        )

        # Global Denge & Klinker Verimi

        Q_total_out = m_gas_pre * 1150.0 * x_next["Tg_preheater"]
        x_next["Energy_Residual"] = Q_in_total - Q_total_out

        loss_on_ignition = 1.0 - x_next["Clinker_yield"]
        Clinker_output_rate = x_next["Feed_rate"] * (1.0 - loss_on_ignition)
        x_next["Clinker_output"] = (
            0.95 * x.get("Clinker_output", Clinker_output_rate)
        ) + (0.05 * Clinker_output_rate)

        x_next["dTg_burning"] = (x_next["Tg_burning"] - x["Tg_burning"]) / self.dt

        # CO ppm
        oxygen_deficit = max(0.0, 6.0 - x_next["O2"])
        target_CO = 20.0 + 800.0 * oxygen_deficit * (
            x_next["Fuel_rate"] / 6.0
        ) * np.exp(-x_next["Tg_burning"] / 1200.0)
        x_next["CO_ppm"] = x["CO_ppm"] + 0.25 * (target_CO - x["CO_ppm"])

        x_next["P_calcination"] = -406.0 + 226.0 * np.exp(-t / 20.0)

        # ----------------------------------------------------------
        # MASS BALANCE CHECK
        # ----------------------------------------------------------

        x_next["Mass_Balance_Error"] = abs(80.0 - (Ca_inventory + x_next["CO2"]))

        # ==========================================================
        # OPTIONAL: MASS CHECK (DEBUG ONLY)
        # ==========================================================

        x_next["CaO_balance_error"] = abs(
            (x["CaO"] + CaO_generated)
            - (
                dC2S * 0.6512
                + dC3S * 0.2456
                + dC3A * 0.6227
                + dC4AF * 0.4616
                + x_next["CaO"]
            )
        )

        # ==========================================================
        # MASS CHECK (corrected)
        # ==========================================================

        CaO_in = x["CaO"] + CaO_generated

        CaO_out = x_next["CaO"]

        CaO_consumed = dC2S * 0.6512 + dC3S * 0.2456 + dC3A * 0.6227 + dC4AF * 0.4616

        x_next["Mass_Balance_Error"] = abs(CaO_in - (CaO_out + CaO_consumed))

        x_next["Q_acc"] = x_next["Clinker_output"] * 150.0
        x_next["Q_loss"] = (
            0.0016 * 5.67e-8 * 0.8 * 110 * (x_next["Tg_burning"] ** 4 - 25**4)
        )
        x_next["Q_reaction"] = x_next["Feed_rate"] * (1 - x_next["Clinker_yield"]) * 3.2
        x_next["Q_out"] = x_next["Q_acc"] + x_next["Q_loss"] + x_next["Q_reaction"]
        x_next["Q_gas"] = x_next["Fuel_rate"] * 1.1 * (x_next["Tg_burning"] - 25.0)

        clinker_safe = max(x_next["Clinker_output"], 1e-6)
        x_next["Normalized_Energy_Index"] = x_next["Q_in"] / clinker_safe
        x_next["Global_Energy_Closure"] = (
            x_next["Q_in"] - x_next["Q_out"] + x_next["Q_reaction"] - x_next["Q_acc"]
        )

        q_in_safe = max(x_next["Q_in"], 1e-6)
        x_next["Energy_error"] = (x_next["Global_Energy_Closure"] / q_in_safe) * 100.0

        if t == 0:
            x_next["SCALE"] = 1.0

        # Kritik Dönüş İşlemi: None Type Error bu satırın eklenmesi ile çözüldü.

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
        0.1,
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
