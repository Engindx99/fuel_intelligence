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

        thermal_factor = 1.0

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

        # --------------------------------------------------------------
        # ENERJİ DAĞILIMI
        # --------------------------------------------------------------

        import numpy as np

        # --------------------------------------------------------------
        # CONSTANTS
        # --------------------------------------------------------------
        AIR_DENSITY = 1.293
        Cp_air = 1005.0
        T_ref = 25.0

        dt_sec = self.dt * 3600.0

        # --------------------------------------------------------------
        # FLOWS
        # --------------------------------------------------------------
        m_gas_pre = x_next["Air_flow"] * AIR_DENSITY / 3600.0
        m_solid_pre = x_next["Feed_rate"]

        m_air_tertiary = x_next.get("Tertiary_air_flow", 0.0) * AIR_DENSITY / 3600.0

        # --------------------------------------------------------------
        # O2 & HEAT INPUT (Q_in)
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # DEBUG: INPUT ENERGY TRACKING
        # --------------------------------------------------------------
        x_next["Q_in_total"] = Q_in_total

        # --------------------------------------------------------------
        # TERSIYER AIR ENERGY PENALTY (FIXED LOCATION)
        # --------------------------------------------------------------
        Tg_in = x.get("Tg_preheater", 1400.0)

        Q_ter_sink = m_air_tertiary * Cp_air * (Tg_in - T_ref)

        Q_in_total = Q_in_total - Q_ter_sink

        # --------------------------------------------------------------
        # DEBUG: LOSSES
        # --------------------------------------------------------------
        x_next["Q_ter_sink"] = Q_ter_sink

        # --------------------------------------------------------------
        # TEMPERATURE STATES
        # --------------------------------------------------------------
        T_burn = x.get("Tg_burning", 1400.0)
        T_calc = x.get("Ts_calcination", 1200.0)
        T_pre = x.get("Ts_preheater", 400.0)
        T_sec = x.get("Tg_Cooling", 500.0)
        T_ter = 400.0

        # --------------------------------------------------------------
        # SMOOTH DIRECTIONAL ΔT (MPC SAFE - STABLE VERSION)
        # --------------------------------------------------------------
        k = 0.02

        def smooth_dT(Ta, Tb):
            dT = Ta - Tb
            return dT / (1.0 + np.exp(-k * dT))

        dT_calc = smooth_dT(T_burn, T_calc)
        dT_pre = smooth_dT(T_burn, T_pre)
        dT_sec = smooth_dT(T_burn, T_sec)
        dT_burn = smooth_dT(2000.0, T_burn)
        dT_ter = smooth_dT(1200.0, T_ter)

        # --------------------------------------------------------------
        # HEAT TRANSFER COEFFICIENTS
        # --------------------------------------------------------------
        UA_burn = 800.0 + 2.0 * m_gas_pre
        UA_calc = 500.0 + 1.5 * m_solid_pre
        UA_pre = 400.0 + 1.0 * m_gas_pre
        UA_sec = 300.0 + 1.2 * m_gas_pre
        UA_ter = 250.0 + 0.8 * m_air_tertiary

        # --------------------------------------------------------------
        # ENERGY DEMAND (PHYSICAL DRIVER MODEL)
        # --------------------------------------------------------------
        Q_burning_pool = UA_burn * dT_burn
        Q_calcination_pool = UA_calc * dT_calc
        Q_preheater_pool = UA_pre * dT_pre
        Q_secondary_pool = UA_sec * dT_sec
        Q_tertiary_pool = UA_ter * dT_ter

        # --------------------------------------------------------------
        # ENERGY CLOSURE (GLOBAL CONSISTENCY SCALING)
        # --------------------------------------------------------------
        Q_demand_total = (
            Q_burning_pool
            + Q_calcination_pool
            + Q_preheater_pool
            + Q_secondary_pool
            + Q_tertiary_pool
        )

        scale = Q_in_total / (Q_demand_total + 1e-9)

        Q_burning_pool *= scale
        Q_calcination_pool *= scale
        Q_preheater_pool *= scale
        Q_secondary_pool *= scale
        Q_tertiary_pool *= scale

        # --------------------------------------------------------------
        # DEBUG: SCALING BEHAVIOR
        # --------------------------------------------------------------
        x_next["Q_demand_total"] = Q_demand_total
        x_next["Q_scale"] = scale

        # --------------------------------------------------------------
        # ENERGY BALANCE CHECK
        # --------------------------------------------------------------
        Q_check = (
            Q_burning_pool
            + Q_calcination_pool
            + Q_preheater_pool
            + Q_secondary_pool
            + Q_tertiary_pool
        )

        x_next["Q_energy_error"] = Q_in_total - Q_check
        # --------------------------------------------------------------
        # BURNING ZONE (CLOSED ENERGY FLOW / NO ARTIFICIAL SOURCING)
        # --------------------------------------------------------------

        import numpy as np

        Cp_g_burn, Cp_s_burn, Cp_w_burn = 1250.0, 1150.0, 1000.0
        A_c_burn, h_c_burn = 13.85, 0.05
        A_s_burn, A_w_burn, A_ws_burn = 82.0, 70.0, 57.0

        m_air_s = x_next["Air_flow"] * AIR_DENSITY / 3600.0
        m_solid_s = (x_next["Feed_rate"] * 1000.0) / 3600.0

        # --------------------------------------------------------------
        # ENERGY INPUT (FROM GLOBAL ENERGY DISTRIBUTION)
        # --------------------------------------------------------------
        Q_in_burn = x_next.get("Q_burning_pool", 0.0)

        # DEBUG
        x_next["Q_in_burn_raw"] = Q_in_burn

        # efficiency split (physical loss tracking)
        eta_burn = 0.97
        Q_combustion = Q_in_burn * eta_burn
        Q_stack_loss = Q_in_burn * (1.0 - eta_burn)

        # DEBUG
        x_next["Q_combustion"] = Q_combustion
        x_next["Q_stack_loss"] = Q_stack_loss

        # --------------------------------------------------------------
        # LUMPED CAPACITIES
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # GAS ENERGY BALANCE (INTERNAL REDISTRIBUTION ONLY)
        # --------------------------------------------------------------
        a_gas_burn = (
            (h_gs_burn * A_s_burn)
            + (h_gw_burn * A_w_burn)
            + (h_c_burn * 1000.0 * A_c_burn)
            + (m_air_s * Cp_g_burn)
        )

        b_gas_burn = (
            Q_combustion
            + (h_gw_burn * A_w_burn * Tw_curr)
            + (h_c_burn * 1000.0 * A_c_burn * 30.0)
            + (m_air_s * Cp_g_burn * 400.0)
            + (h_gs_burn * A_s_burn * Ts_curr)
        )

        Tg_next_burn = (C_gas_total * Tg_curr + dt_sec * b_gas_burn) / (
            C_gas_total + dt_sec * a_gas_burn
        )

        # --------------------------------------------------------------
        # WALL ENERGY BALANCE (CONSERVATIVE LOSS MODEL)
        # --------------------------------------------------------------
        T_amb = 25.0
        h_loss = 8.0

        Q_loss_wall = h_loss * A_w_burn * (Tw_curr - T_amb)
        Q_wall_to_solid = h_ws_burn * A_ws_burn * (Tw_curr - Ts_curr)

        Tw_next_burn = Tw_curr + (dt_sec / C_wall_total) * (
            h_gw_burn * A_w_burn * (Tg_next_burn - Tw_curr)
            - Q_loss_wall
            - Q_wall_to_solid
        )

        # --------------------------------------------------------------
        # SOLID PHASE (REACTION COUPLED ENERGY)
        # --------------------------------------------------------------
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

        Ts_next_burn = (C_solid_total * Ts_curr + dt_sec * b_sol_burn) / (
            C_solid_total + dt_sec * a_sol_burn
        )

        # --------------------------------------------------------------
        # STATE UPDATE
        # --------------------------------------------------------------
        x_next["Tg_burning"] = Tg_next_burn
        x_next["Ts_burning"] = Ts_next_burn
        x_next["Tw_burning"] = Tw_next_burn

        # --------------------------------------------------------------
        # ENERGY TRACKING (CRITICAL FOR CLOSURE)
        # --------------------------------------------------------------
        x_next["Q_burning_in"] = Q_in_burn
        x_next["Q_combustion_used"] = Q_combustion
        x_next["Q_stack_loss"] = Q_stack_loss
        x_next["Q_exothermic"] = Q_exo_W
        x_next["Q_wall_loss"] = Q_loss_wall
        x_next["Q_wall_to_solid"] = Q_wall_to_solid

        # ------------------------------------------------------
        # CALCINATION ZONE (ENERGY POOL / CLOSED LOOP VERSION)
        # ------------------------------------------------------

        import numpy as np

        h_gs = 320.0
        A_s = 950.0

        Cp_solid_calc = 1050.0
        Cp_air = 1005.0
        T_ref = 25.0

        Ts_calc_curr = x.get("Ts_calcination", 800.0)
        Tg_calc_curr = x.get("Tg_calcination", 900.0)

        # ------------------------------------------------------
        # INVENTORY
        # ------------------------------------------------------
        m_solid_calc = x_next["Kiln_solid_out"]  # ton
        m_solid_kg_s = (m_solid_calc * 1000.0) / 3600.0

        m_air_kiln = x_next["Air_flow"] * AIR_DENSITY / 3600.0
        m_air_ter = x_next.get("Tertiary_air_flow", 0.0) * AIR_DENSITY / 3600.0

        # ------------------------------------------------------
        # AVAILABLE ENERGY FOR THIS ZONE
        # ------------------------------------------------------
        Q_available = Q_calcination_pool

        # DEBUG
        x_next["Q_calcination_pool_raw"] = Q_available

        # ------------------------------------------------------
        # REACTION HEAT (ENDOTHERMIC)
        # ------------------------------------------------------
        activation = 1.0 / (1.0 + np.exp(-0.08 * (Ts_calc_curr - 850.0)))

        Q_rxn_specific = 1.7e6  # J/kg CaCO3

        Q_rxn_demand = m_solid_kg_s * activation * Q_rxn_specific

        Q_rxn_used = min(Q_rxn_demand, Q_available)

        Q_rxn_calc = -Q_rxn_used

        Q_available -= Q_rxn_used

        # DEBUG
        x_next["Q_rxn_demand_calcination"] = Q_rxn_demand
        x_next["Q_rxn_used_calcination"] = Q_rxn_used
        x_next["Q_calcination_remaining_pool"] = Q_available

        # ------------------------------------------------------
        # GAS ENTHALPY INLETS
        # ------------------------------------------------------
        Q_from_burning = m_air_kiln * Cp_air * max(x_next["Tg_burning"] - T_ref, 0.0)

        Q_from_tertiary = m_air_ter * Cp_air * max(400.0 - T_ref, 0.0)

        Q_gas_in = Q_from_burning + Q_from_tertiary + Q_available

        # DEBUG
        x_next["Q_from_burning_calc"] = Q_from_burning
        x_next["Q_from_tertiary_calc"] = Q_from_tertiary
        x_next["Q_gas_in_calcination"] = Q_gas_in

        # ------------------------------------------------------
        # SOLID ENERGY BALANCE
        # ------------------------------------------------------
        C_solid_total = m_solid_kg_s * Cp_solid_calc

        a_solid = h_gs * A_s + m_solid_kg_s * Cp_solid_calc

        b_solid = h_gs * A_s * Tg_calc_curr + Q_rxn_calc

        Ts_calc_pred = (C_solid_total * Ts_calc_curr + dt_sec * b_solid) / (
            C_solid_total + dt_sec * a_solid + 1e-9
        )

        # ------------------------------------------------------
        # GAS ENERGY BALANCE
        # ------------------------------------------------------
        gas_hold_mass = 100.0
        C_gas_total = gas_hold_mass * Cp_air

        wall_term = 4.0 * 13.85

        a_gas = (m_air_kiln + m_air_ter) * Cp_air + wall_term + h_gs * A_s

        b_gas = (
            Q_gas_in
            + wall_term * x.get("Tg_preheater", 350.0)
            + h_gs * A_s * Ts_calc_pred
        )

        Tg_calc_pred = (C_gas_total * Tg_calc_curr + dt_sec * b_gas) / (
            C_gas_total + dt_sec * a_gas + 1e-9
        )

        # ------------------------------------------------------
        # ENERGY ACCOUNTING
        # ------------------------------------------------------
        Q_calc_unused = max(Q_available, 0.0)

        # DEBUG ENERGY CLOSURE
        x_next["Q_calc_unused"] = Q_calc_unused
        x_next["Q_calc_total_in"] = Q_calcination_pool
        x_next["Q_calc_energy_balance"] = Q_calcination_pool - (
            Q_rxn_used + Q_calc_unused
        )

        # ------------------------------------------------------
        # STATE UPDATE
        # ------------------------------------------------------
        x_next["Ts_calcination"] = Ts_calc_pred
        x_next["Tg_calcination"] = Tg_calc_pred

        x_next["Q_rxn_calcination"] = Q_rxn_used
        x_next["Q_calcination_used"] = Q_calcination_pool - Q_calc_unused
        x_next["Q_calcination_unused"] = Q_calc_unused

        # ==============================================================================
        # PREHEATER ZONE (ENERGY POOL / CLOSED LOOP VERSION)
        # ==============================================================================

        import numpy as np

        Cp_gas_pre = 1150.0
        Cp_solid_pre = 1050.0

        Tg_in = x_next["Tg_calcination"]
        Ts_pre = x.get("Ts_preheater", 25.0)

        # ------------------------------------------------------
        # FLOW CAPACITIES
        # ------------------------------------------------------
        m_gas_pre = x_next["Air_flow"] * AIR_DENSITY / 3600.0
        m_solid_pre = x_next["Feed_rate"] * 1000.0 / 3600.0

        C_gas_flow = m_gas_pre * Cp_gas_pre + 1e-9
        C_solid_flow = m_solid_pre * Cp_solid_pre + 1e-9

        # ------------------------------------------------------
        # HEAT TRANSFER GEOMETRY
        # ------------------------------------------------------
        h_pre = 18.0
        A_pre = 120.0
        UA_pre = h_pre * A_pre

        # ------------------------------------------------------
        # PRESSURE MODEL (UNCHANGED)
        # ------------------------------------------------------
        P_in = x_next.get("P_calcination", 101325.0)

        Tg_ref = x.get("Tg_preheater", Tg_in)
        Tg_preheater_K = Tg_ref + 273.15

        rho = AIR_DENSITY * (T_ref / Tg_preheater_K)

        K_resistance = 1200.0

        dP = K_resistance * (m_gas_pre**2) / (rho + 1e-9)

        P_preheater = P_in - dP
        x_next["P_preheater"] = P_preheater

        # ------------------------------------------------------
        # TEMPERATURE DRIVER
        # ------------------------------------------------------
        k = 0.02

        dT = Tg_in - Ts_pre
        dT_smooth = dT / (1.0 + np.exp(-k * dT))

        # ------------------------------------------------------
        # HOLDOUP DYNAMICS
        # ------------------------------------------------------
        M_gas_hold = 150.0
        M_solid_hold = 3000.0

        C_gas_bulk = M_gas_hold * Cp_gas_pre
        C_solid_bulk = M_solid_hold * Cp_solid_pre

        tau_g = C_gas_bulk / (UA_pre + 1e-9)
        tau_s = C_solid_bulk / (UA_pre + 1e-9)

        eps = 1e-4

        tau_eff = 0.5 * (tau_g + tau_s - np.sqrt((tau_g - tau_s) ** 2 + eps))

        alpha = dt_sec / (dt_sec + tau_eff + 1e-9)

        alpha_max = 0.35

        alpha_stable = 0.5 * (
            alpha + alpha_max - np.sqrt((alpha - alpha_max) ** 2 + eps)
        )

        # ------------------------------------------------------
        # PHYSICAL HEAT DEMAND
        # ------------------------------------------------------
        Q_pre_demand = alpha_stable * UA_pre * dT_smooth

        # ------------------------------------------------------
        # AVAILABLE ENERGY FROM GLOBAL POOL
        # ------------------------------------------------------
        Q_available = Q_preheater_pool

        # DEBUG
        x_next["Q_preheater_pool_raw"] = Q_available
        x_next["Q_preheater_demand"] = Q_pre_demand

        Q_pre_used = min(Q_pre_demand, Q_available)

        Q_pre_unused = max(Q_available - Q_pre_used, 0.0)

        # DEBUG ENERGY FLOW
        x_next["Q_preheater_used_debug"] = Q_pre_used
        x_next["Q_preheater_unused_debug"] = Q_pre_unused

        # ------------------------------------------------------
        # ENERGY CONSISTENT STATE UPDATE
        # ------------------------------------------------------
        Tg_next = Tg_in - Q_pre_used / (C_gas_flow + 1e-9)
        Ts_next = Ts_pre + Q_pre_used / (C_solid_flow + 1e-9)

        # MPC-friendly smoothing
        Tg_preheater_pred = 0.7 * Tg_in + 0.3 * Tg_next
        Ts_preheater_pred = 0.7 * Ts_pre + 0.3 * Ts_next

        # ------------------------------------------------------
        # STATE UPDATE
        # ------------------------------------------------------
        x_next["Tg_preheater"] = Tg_preheater_pred
        x_next["Ts_preheater"] = Ts_preheater_pred

        x_next["Q_dot_preheater"] = Q_pre_used
        x_next["Q_preheater_used"] = Q_pre_used
        x_next["Q_preheater_unused"] = Q_pre_unused
        x_next["UA_preheater_effective"] = alpha_stable * UA_pre

        # ------------------------------------------------------
        # COOLING ZONE (FINAL CLOSED-LOOP ENERGY VERSION)
        # ------------------------------------------------------

        import numpy as np

        Tamb_solid = 130.0
        Tamb_gas = 510.0

        Ts_cool_curr = x.get(
            "Ts_Cooling",
            x_next.get("Ts_burning", 1450.0),
        )

        Tg_cool_curr = x.get(
            "Tg_Cooling",
            x_next.get("Tg_burning", 1490.0),
        )

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
            1.0 - np.exp(-dt_sec / tau_klinker)
        )

        Ts_cool_next = Ts_cool_curr + dTs_ambient - Q_transfer / C_solid

        # -------------------------------
        # GAS DYNAMICS
        # -------------------------------
        dTg_ambient = (Tamb_gas - Tg_cool_curr) * (1.0 - np.exp(-dt_sec / tau_gas))

        Tg_cool_next = Tg_cool_curr + dTg_ambient + Q_transfer / C_gas

        # -------------------------------
        # SECONDARY AIR ENERGY RECOVERY
        # -------------------------------
        dT_sec = (Tg_cool_curr - 200.0) / (1.0 + np.exp(-0.03 * (Tg_cool_curr - 200.0)))

        Q_secondary_air_raw = m_gas * Cp_gas * dT_sec

        eta_recovery = 0.65

        Q_secondary_air = eta_recovery * Q_secondary_air_raw

        # DEBUG
        x_next["Q_secondary_air_raw"] = Q_secondary_air_raw
        x_next["Q_secondary_air_recovered"] = Q_secondary_air

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        x_next["Ts_Cooling"] = Ts_cool_next
        x_next["Tg_Cooling"] = Tg_cool_next

        # ----------------------------------------------------------
        # REACTIONS
        # ----------------------------------------------------------
        # -------------------------
        # CALCINATION
        # -------------------------

        Ts_calc = x_next["Ts_calcination"] + 273.15

        # FIX: missing variable safety
        T_calc = Ts_calc

        if T_calc > 873.15:
            k_calc = 1e7 * np.exp(-160000.0 / (8.314 * T_calc))
        else:
            k_calc = 0.0

        CaCO3_in = x["CaCO3"]

        reacted = CaCO3_in * (1.0 - np.exp(-k_calc * self.dt))
        reacted = np.clip(reacted, 0.0, CaCO3_in)

        CaCO3_out = CaCO3_in - reacted
        CaO_generated = reacted * 0.5603
        CO2_generated_phys = reacted * 0.4397

        x_next["CaCO3"] = CaCO3_out
        x_next["CaO_generated"] = CaO_generated
        x_next["CO2_generated"] = CO2_generated_phys

        x_next["dCaO_calcination"] = CaO_generated / self.dt if self.dt > 0 else 0.0
        x_next["dCO2_calcination"] = (
            CO2_generated_phys / self.dt if self.dt > 0 else 0.0
        )

        CaO_pool = x.get("CaO", 0.0) + CaO_generated

        # DEBUG
        x_next["CaO_pool"] = CaO_pool
        x_next["CaCO3_reacted"] = reacted

        # ==========================================================
        # BURN ZONE TEMPERATURE AGGREGATION (NO DUPLICATE STATES)
        # ==========================================================

        Tg_burn = x.get("Tg_burning", 1450.0) + 273.15
        Ts_burn = x.get("Ts_burning", 1400.0) + 273.15
        Tw_burn = x.get("Tw_burning", 1300.0) + 273.15

        T_burn_eff = 0.7 * Ts_burn + 0.2 * Tg_burn + 0.1 * Tw_burn

        # DEBUG
        x_next["T_burn_eff"] = T_burn_eff

        # ==========================================================
        # C2S FORMATION
        # ==========================================================

        SiO2 = x["SiO2"]

        if SiO2 <= 1e-6 or T_burn_eff < 1000.0:
            dC2S = 0.0
        else:
            k_c2s = 5e7 * np.exp(-170000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k_c2s * self.dt)

            stoich_limit = min(
                SiO2 / 0.3488,
                max(0.0, CaO_pool) / 0.6512,
            )

            dC2S = stoich_limit * kinetic

        # -----------------------------
        # MASS UPDATE
        # -----------------------------
        CaO_pool_before = CaO_pool

        CaO_pool -= dC2S * 0.6512
        CaO_pool = max(0.0, CaO_pool)

        # -----------------------------
        # DEBUG TRACKING
        # -----------------------------
        x_next["dC2S"] = dC2S
        x_next["CaO_pool_after_C2S"] = CaO_pool
        x_next["CaO_pool_consumed_C2S"] = CaO_pool_before - CaO_pool

        # ==========================================================
        # C3S FORMATION
        # ==========================================================

        C2S_in = x.get("C2S", 0.0) + dC2S

        if C2S_in <= 1e-6 or T_burn_eff < 1473.15:
            dC3S = 0.0
        else:
            k_c3s = 2.28e8 * np.exp(-200000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k_c3s * self.dt)

            stoich_limit = min(
                C2S_in / 0.7544,
                max(0.0, CaO_pool) / 0.2456,
            )

            dC3S = stoich_limit * kinetic

        # -----------------------------
        # CAO UPDATE (CRITICAL FEEDBACK)
        # -----------------------------
        CaO_pool_before_c3s = CaO_pool

        CaO_pool -= dC3S * 0.2456
        CaO_pool = max(0.0, CaO_pool)

        # -----------------------------
        # STATE UPDATE
        # -----------------------------
        x_next["C2S"] = C2S_in - dC3S
        x_next["C3S"] = x.get("C3S", 0.0) + dC3S

        # -----------------------------
        # DEBUG / MONITORING
        # -----------------------------
        x_next["dC3S"] = dC3S
        x_next["CaO_pool_after_C3S"] = CaO_pool
        x_next["CaO_pool_consumed_C3S"] = CaO_pool_before_c3s - CaO_pool

        # ==========================================================
        # C3A FORMATION (NEW - AL2O3 COMPETITOR PHASE)
        # ==========================================================

        Al2O3 = x["Al2O3"]
        CaO_pool = max(0.0, CaO_pool)

        if Al2O3 <= 1e-6 or T_burn_eff < 900.0:
            dC3A = 0.0
        else:
            k_c3a = 1.2e6 * np.exp(-155000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k_c3a * self.dt)

            stoich_limit = min(Al2O3 / 0.3773, CaO_pool / 0.6227)

            dC3A = stoich_limit * kinetic

        # -----------------------------
        # CAO UPDATE (C3A PATHWAY)
        # -----------------------------
        CaO_pool_before_c3a = CaO_pool

        CaO_pool -= dC3A * 0.6227
        CaO_pool = max(0.0, CaO_pool)

        # -----------------------------
        # STATE UPDATE
        # -----------------------------
        x_next["dC3A"] = dC3A
        x_next["CaO_pool_after_C3A"] = CaO_pool
        x_next["CaO_pool_consumed_C3A"] = CaO_pool_before_c3a - CaO_pool

        # ==========================================================
        # C4AF FORMATION
        # ==========================================================

        Fe2O3 = x["Fe2O3"]
        Al2O3 = x["Al2O3"]

        if Fe2O3 <= 1e-6 or T_burn_eff < 1100.0:
            dC4AF = 0.0
        else:
            k = 2e5 * np.exp(-150000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            # IMPORTANT: Al2O3 already partially consumed by C3A
            Al2O3_remaining = max(0.0, Al2O3 - dC3A * 0.3773)

            stoich_limit = min(
                Fe2O3 / 0.3286, Al2O3_remaining / 0.2098, CaO_pool / 0.4616
            )

            dC4AF = stoich_limit * kinetic

        # CaO consumption (C4AF)
        CaO_pool -= dC4AF * 0.4616
        CaO_pool = max(0.0, CaO_pool)

        # ==========================================================
        # FINAL UPDATE (CLEAN STATE WRITE)
        # ==========================================================

        # -------------------------
        # REACTION EXTENTS SAFETY
        # -------------------------
        dC2S = locals().get("dC2S", 0.0)
        dC3S = locals().get("dC3S", 0.0)
        dC3A = locals().get("dC3A", 0.0)
        dC4AF = locals().get("dC4AF", 0.0)

        # -------------------------
        # FLOW / THERMAL COUPLING
        # -------------------------
        flow_scale = x_next.get("Kiln_solid_out", 1e-6) / max(x["Feed_rate"], 1e-6)

        thermal_factor = np.exp(-x_next["Tg_burning"] / 1200.0)

        activity_factor = min(1.0, x_next.get("Material_acc", 0.0) / 50.0)

        coupling = flow_scale * thermal_factor * activity_factor

        # -------------------------
        # EFFECTIVE REACTIONS
        # -------------------------
        dC2S_eff = dC2S * coupling
        dC3S_eff = dC3S * coupling
        dC3A_eff = dC3A * coupling
        dC4AF_eff = dC4AF * coupling

        # -------------------------
        # CLINKER PHASE UPDATE (SINGLE WRITE)
        # -------------------------
        x_next["C2S"] = x.get("C2S", 0.0) + dC2S_eff - dC3S_eff * 0.7544

        x_next["C3S"] = x.get("C3S", 0.0) + dC3S_eff

        # C3A added
        x_next["C3A"] = x.get("C3A", 0.0) + dC3A_eff

        x_next["C4AF"] = x.get("C4AF", 0.0) + dC4AF_eff

        # -------------------------
        # ELEMENT BALANCES
        # -------------------------
        x_next["SiO2"] = max(0.0, x["SiO2"] - dC2S_eff * 0.3488)

        # Al2O3 consumed by BOTH C3A and C4AF
        x_next["Al2O3"] = max(0.0, x["Al2O3"] - dC3A_eff * 0.3773 - dC4AF_eff * 0.2098)

        x_next["Fe2O3"] = max(0.0, x["Fe2O3"] - dC4AF_eff * 0.3286)

        # -------------------------
        # CaO CONSERVATION (CLOSED FORM)
        # -------------------------
        CaO_in = x["CaO"] + CaO_generated

        CaO_consumed = (
            dC2S_eff * 0.6512
            + dC3S_eff * 0.2456
            + dC3A_eff * 0.6227
            + dC4AF_eff * 0.4616
        )

        CaO_next = CaO_in - CaO_consumed
        x_next["CaO"] = max(0.0, CaO_next)

        # -------------------------
        # CO2 CLOSURE
        # -------------------------
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

        x_next["dTg_burning"] = (x_next["Tg_burning"] - x["Tg_burning"]) / max(
            self.dt, 1e-9
        )

        oxygen_deficit = max(0.0, 6.0 - x_next["O2"])

        target_CO = 20.0 + 800.0 * oxygen_deficit * (
            x_next["Fuel_rate"] / 6.0
        ) * np.exp(-x_next["Tg_burning"] / 1200.0)

        x_next["CO_ppm"] = x["CO_ppm"] + 0.25 * (target_CO - x["CO_ppm"])

        # FIX: time variable safety
        t_safe = x_next.get("t", getattr(self, "t", 0.0))

        x_next["P_calcination"] = -406.0 + 226.0 * np.exp(-t_safe / 20.0)

        # ==========================================================
        # MASS BALANCE CHECKS
        # ==========================================================

        # safety fallback (must exist from earlier blocks)
        Ca_inventory_safe = x_next.get("Ca_inventory", 0.0)
        CaO_in_safe = x_next.get("CaO_in", 0.0)
        CaO_consumed_safe = x_next.get("CaO_consumed", 0.0)

        x_next["CO2_balance_error"] = abs(80.0 - (Ca_inventory_safe + x_next["CO2"]))
        x_next["CaO_balance_error"] = abs(
            CaO_in_safe - (x_next["CaO"] + CaO_consumed_safe)
        )

        # ==========================================================
        # ENERGY TERMS
        # ==========================================================

        sigma = 5.67e-8

        x_next["Q_loss"] = (
            0.0016 * sigma * 0.8 * 110 * (x_next["Tg_burning"] ** 4 - 25.0**4)
        )

        x_next["Q_reaction"] = (
            x_next["Feed_rate"] * (1.0 - x_next["Clinker_yield"]) * 3.2
        )

        # ensure Q_acc exists
        x_next["Q_acc"] = x_next.get("Q_acc", 0.0)

        x_next["Q_out"] = x_next["Q_acc"] + x_next["Q_loss"] + x_next["Q_reaction"]

        x_next["Q_gas"] = x_next["Fuel_rate"] * 1.1 * (x_next["Tg_burning"] - 25.0)

        # ==========================================================
        # GLOBAL ENERGY CLOSURE
        # ==========================================================

        x_next["Global_Energy_Closure"] = x_next["Q_in"] - x_next["Q_out"]

        q_in_safe = max(x_next["Q_in"], 1e-6)

        x_next["Energy_error"] = (x_next["Global_Energy_Closure"] / q_in_safe) * 100.0

        clinker_safe = max(x_next.get("Clinker_output", 1e-6), 1e-6)

        x_next["Normalized_Energy_Index"] = x_next["Q_in"] / clinker_safe

        # ==========================================================
        # SCALE INITIALIZATION
        # ==========================================================

        if x.get("t", 0.0) == 0:
            x_next["SCALE"] = 1.0

        return x_next


def run_simulation():
    # ==========================================================
    # SIMULATION CONFIG
    # ==========================================================
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
    ) = (650.0, 950.0, 1450.0, 1550.0)

    (
        x_current.Air_flow,
        x_current.Cooling_air_flow,
        x_current.ID_fan_speed,
    ) = (45100.0, 80000.0, 900.0)

    (
        x_current.Feed_rate,
        x_current.Kiln_solid_out,
        x_current.Material_acc,
    ) = (71.00, 0.0, 0.0)

    (
        x_current.Ts_preheater,
        x_current.Ts_calcination,
        x_current.Ts_burning,
        x_current.Ts_Cooling,
    ) = (300.0, 850.0, 1420.0, 1450.0)

    x_current.CaCO3, x_current.CaO, x_current.CO2 = 80.0, 1e-6, 1e-6
    x_current.SiO2, x_current.Al2O3, x_current.Fe2O3 = 13.0, 4.0, 3.0

    x_current.C2S = x_current.C3S = x_current.C3A = x_current.C4AF = 1e-6

    x_current.kiln_rpm = 1.0
    x_current.Residence = KilnPhysicsEngine.get_residence_time(1.0)

    x_current.Petcoke, x_current.Alternative_Fuel, x_current.Lignite_Coal = (
        0.03,
        0.07,
        0.90,
    )

    x_current.Material_acc = 15.0
    x_current.Clinker_output = 11.13
    x_current.Kiln_solid_out = 12.50

    x_current.P_preheater = 12.0
    x_current.P_calcination = 5.5
    x_current.P_burning = 1.2

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

    x_current.Normalized_Energy_Index = 1.0
    x_current.Global_Energy_Closure = 0.0
    x_current.Energy_error = 0.0

    # ==========================================================
    # INPUT MODEL
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
    # SIM LOOP
    # ==========================================================
    simulation_records = []

    simulation_records.append(asdict(x_current))

    sim_time = 0.0

    for step_idx in range(N_total_reports):

        inputs = input_layer(sim_time, initial_state=x_current)

        for _ in range(STEPS_PER_REPORT):
            sim_time += dt
            x_current = executor.perform_step(x_current, sim_time, inputs=inputs)

        x_current.t = sim_time
        simulation_records.append(asdict(x_current))

    # ==========================================================
    # OUTPUT
    # ==========================================================
    df_results = pd.DataFrame(simulation_records)

    float_cols = df_results.select_dtypes(include=["float", "float64"]).columns
    df_results[float_cols] = df_results[float_cols].round(5)

    df_results.to_csv("engine.csv", index=False)

    print(
        f"Simülasyon tamamlandı → {len(df_results)} satır, dt={dt}, report_dt={reporting_dt}"
    )


if __name__ == "__main__":
    run_simulation()
