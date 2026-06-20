import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, fields

# 1. Veri Yapısı
@dataclass
class KilnState:
    t: float = 0.0
    Regime: str = "R1_HEATING_STABILIZATION"
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
    Fuel_rate: float = 2.5
    Tg_preheater: float = 350.0
    Ts_preheater: float = 80.0
    Tg_calcination: float = 850.0
    Ts_calcination: float = 800.0
    Tg_burning: float = 1200.0
    Ts_burning: float = 1050.0
    Tg_Cooling: float = 25.0
    Ts_Cooling: float = 400.0
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
    Clinker_yield: float = 0.65  # Fiziksel kütle dengesi gereği (CO2 kaybı) 0.89'dan 0.65'e güncellendi
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
        rpm = max(0.1, kiln_rpm)
        rpm_eff = rpm / (0.26 + rpm)
        filling = max(0.01, filling_rate)
        filling_factor = (0.08 / filling) ** 0.3
        v_axial = (5.61 * D * rpm_eff * (1.5 + 44.8 * slope)) * filling_factor
        return (L / (v_axial + 1e-6)) * 60.0


# 3. İcra Edici
class StepExecutor:
    def __init__(self, dt=0.05):
        self.dt = dt  # saat cinsinden (0.05 saat = 180 saniye)

    def perform_step(self, x: KilnState, t: float, inputs: dict = None) -> KilnState:
        inputs = inputs or {}
        x_next = x.copy()

        regime = inputs.get("regime", "R1_HEATING_STABILIZATION")
        x_next["Regime"] = regime

        for key in [
            "Fuel_rate", "Feed_rate", "Air_flow", "kiln_rpm", 
            "Petcoke", "Alternative_Fuel"
        ]:
            if key in inputs:
                setattr(x_next, key, inputs[key])

        # FUEL COMPOSITION
        x_next["Petcoke"] = inputs.get("Petcoke", x.get("Petcoke", 0.0))
        x_next["Alternative_Fuel"] = inputs.get("Alternative_Fuel", x.get("Alternative_Fuel", 0.0))
        x_next["Lignite_Coal"] = max(0.0, 1.0 - x_next["Petcoke"] - x_next["Alternative_Fuel"])

        # CONTROL INPUTS
        if "Feed_rate" in inputs:
            x_next["Feed_rate"] = inputs["Feed_rate"]
        else:
            feed = x.get("Feed_rate", 40.0)
            x_next["Feed_rate"] = feed + (120.0 - feed) * 0.0005 * self.dt

        if "Air_flow" in inputs:
            x_next["Air_flow"] = inputs["Air_flow"]
        else:
            x_next["Air_flow"] = 45000.0 + (95000.0 - 45000.0) * (1.0 - np.exp(-t / 120.0))

        if "Cooling_air_flow" in inputs:
            x_next["Cooling_air_flow"] = inputs["Cooling_air_flow"]
        else:
            x_next["Cooling_air_flow"] = 80000.0 + (83000.0 - 80000.0) * (1.0 - np.exp(-t / 140.0))

        if "ID_fan_speed" in inputs:
            x_next["ID_fan_speed"] = inputs["ID_fan_speed"]
        else:
            x_next["ID_fan_speed"] = 900.0 + (2550.0 - 900.0) * (1.0 - np.exp(-t / 110.0))

        if "Damper_position" in inputs:
            x_next["Damper_position"] = inputs["Damper_position"]
        else:
            x_next["Damper_position"] = 33.0 + (85.0 - 33.0) * np.exp(-t / 25.0)

        if "Fuel_rate" in inputs:
            x_next["Fuel_rate"] = inputs["Fuel_rate"]
        else:
            x_next["Fuel_rate"] = 4.0 + (6.8 - 2.5) * (1.0 - np.exp(-t / 35.0))
        f_rate = max(0.1, x_next["Fuel_rate"])

        # RPM & Residence
        if "kiln_rpm" in inputs:
            x_next["kiln_rpm"] = max(0.1, inputs["kiln_rpm"])
        else:
            rpm_current = x.get("kiln_rpm", 1.0)
            rpm_setpoint = 2.4
            alpha = 0.005
            x_next["kiln_rpm"] = max(0.1, rpm_current + alpha * (rpm_setpoint - rpm_current))

        res_min = KilnPhysicsEngine.get_residence_time(x_next["kiln_rpm"])
        x_next["Residence"] = res_min

        # Mass balance
        mat_acc = x.get("Material_acc", 15.0)
        kiln_out = mat_acc / (res_min / 60.0 + 1e-6)
        x_next["Kiln_solid_out"] = kiln_out
        x_next["Material_acc"] = mat_acc + (x_next["Feed_rate"] / 1.55 - kiln_out) * (self.dt / 60.0)

        # O2 & Q_in
        air_fuel_ratio = x_next["Air_flow"] / (f_rate + 1e-6)
        o2_target = 2.5 + 4.5 * np.exp(-1.2 / (air_fuel_ratio / 20000.0 + 1e-3))
        x_next["O2"] = x.get("O2", 3.5) + 0.15 * (o2_target - x.get("O2", 3.5))

        x_next["Q_in"] = (
            (x_next["Lignite_Coal"] * 15000 + x_next["Petcoke"] * 30000 + x_next["Alternative_Fuel"] * 18000)
            * x_next["Fuel_rate"]
            * np.exp(-((x_next["O2"] - 3.5) ** 2) / 25)
        )

        x_next["P_preheater"] = -269.0 + 149.0 * np.exp(-t / 15.0)

        m_gas_pre = x_next["Air_flow"] * 0.001293
        m_solid_pre = x_next["Feed_rate"]
        Q_in_total = x_next["Q_in"] * 1000.0

        feed_factor = x_next["Feed_rate"] / 100.0
        Q_burning = Q_in_total * 0.60
        Q_calcination = Q_in_total * (0.30 * min(feed_factor, 1.2))

        dt_sec = self.dt * 3600.0

        # BURNING ZONE
        Cp_g_burn, Cp_s_burn, A_c_burn, h_c_burn, A_s_burn = 1250.0, 1150.0, 13.85, 0.05, 25.0
        m_air_s = (x_next["Air_flow"] * 1.293) / 3600.0
        m_solid_s = (x_next["Feed_rate"] * 1000.0) / 3600.0
        C_gas_total = 120.0 * Cp_g_burn
        C_solid_total = 4500.0 * Cp_s_burn

        Tg_curr = x.get("Tg_burning", 1200.0)
        Ts_curr = x.get("Ts_burning", 1050.0)
        h_gs_burn = 1500.0 if Tg_curr > 1000.0 else 500.0

        a_gas_burn = (h_gs_burn * A_s_burn) + (h_c_burn * 1000.0 * A_c_burn) + (m_air_s * Cp_g_burn)
        b_gas_burn = (Q_burning) + (h_c_burn * 1000.0 * A_c_burn * 30.0) + (m_air_s * Cp_g_burn * 400.0) + (h_gs_burn * A_s_burn * Ts_curr)
        Tg_next_burn = (C_gas_total * Tg_curr + dt_sec * b_gas_burn) / (C_gas_total + dt_sec * a_gas_burn)

        Q_exo_W = min(350000.0, m_solid_s * 500000.0 * (1.0 + max(0, Ts_curr - 1200.0) / 250.0)) if Ts_curr > 1200.0 else 0.0
        a_sol_burn = (h_gs_burn * A_s_burn) + (m_solid_s * Cp_s_burn)
        b_sol_burn = Q_exo_W + (m_solid_s * Cp_s_burn * 900.0) + (h_gs_burn * A_s_burn * Tg_curr)
        Ts_next_burn = (C_solid_total * Ts_curr + dt_sec * b_sol_burn) / (C_solid_total + dt_sec * a_sol_burn)

        x_next["Tg_burning"], x_next["Ts_burning"] = Tg_next_burn, Ts_next_burn

        # CALCINATION ZONE
        h_gs, A_s = 110.0, 840.0
        m_solid_calc = x_next["Feed_rate"]
        Ts_calc_curr = x.get("Ts_calcination", 800.0)

        dynamic_rate = 0.05 + 0.10 * (1.0 / (1.0 + np.exp(-0.02 * (Ts_calc_curr - 1000.0))))
        reaction_throttle = 1.0 / (1.0 + np.exp(-0.08 * (Ts_calc_curr - 850.0)))
        Q_calc_load = (m_solid_calc / 3600.0) * dynamic_rate * reaction_throttle * 1700.0 * 1000.0

        C_solid_calc_total = (m_solid_calc * 1000.0 * 1050.0) / 3600.0
        b_s = (h_gs * A_s * x.get("Tg_calcination", 900.0)) + (Q_calcination / 3600.0) - Q_calc_load - ((m_solid_calc * 1000.0 / 3600.0) * 1050.0 * (Ts_calc_curr - 25.0))
        x_next["Ts_calcination"] = min((C_solid_calc_total * Ts_calc_curr + dt_sec * b_s) / (C_solid_calc_total + dt_sec * (h_gs * A_s)), 1200.0)

        m_air_calc = x_next["Air_flow"] * 0.001270
        a_gas_calc = (m_air_calc * 1150.0) + (4.0 * 13.85) + (h_gs * A_s)
        b_gas_calc = (m_air_calc * 1150.0 * Tg_next_burn) + Q_calcination + (4.0 * 13.85 * x.get("Tg_preheater", 350.0)) + (h_gs * A_s * x_next["Ts_calcination"])
        x_next["Tg_calcination"] = min(
            (1000.0 * 1150.0 * x.get("Tg_calcination", 900.0) + dt_sec * (b_gas_calc - (m_air_calc * 1150.0 * x.get("Tg_calcination", 900.0)))) 
            / (1000.0 * 1150.0 + dt_sec * a_gas_calc), 1300.0
        )

        # PREHEATER ZONE
        UA_pre = 570000
        a_gas_pre = (m_gas_pre * 1150.0) + UA_pre
        b_gas_pre = (m_gas_pre * 1150.0 * x_next["Tg_calcination"]) + (UA_pre * x.get("Ts_preheater", 25.0))
        x_next["Tg_preheater"] = (200000.0 * x.get("Tg_preheater", 350.0) + dt_sec * b_gas_pre) / (200000.0 + dt_sec * a_gas_pre)

        a_s_pre = (m_solid_pre * 1000.0) + UA_pre
        b_s_pre = (m_solid_pre * 1000.0 * 25.0) + (UA_pre * x_next["Tg_preheater"])
        x_next["Ts_preheater"] = (300000.0 * x.get("Ts_preheater", 25.0) + dt_sec * b_s_pre) / (300000.0 + dt_sec * a_s_pre)

        # COOLING ZONE
        T_amb_cooling, Cp_solid, Cp_gas, h_gs_cool, A_s_cool, M_solid_bed = 30.0, 1150.0, 1050.0, 120.0, 45.0, 12000.0
        Ts_cool_curr = x.Ts_Cooling if x.Ts_Cooling > 0.0 else x_next["Ts_burning"]

        m_air_cool_kgs = (x_next["Cooling_air_flow"] * 1.2) / 3600.0
        m_solid_cool_kgs = max(0.1, (x_next["Kiln_solid_out"] * 1000.0) / 3600.0)

        W_g = m_air_cool_kgs * Cp_gas
        W_s = m_solid_cool_kgs * Cp_solid
        UA_eff = (h_gs_cool * A_s_cool * W_g) / (W_g + h_gs_cool * A_s_cool + 1e-6)

        coeff_A = W_s + UA_eff
        coeff_B = W_s * x_next["Ts_burning"] + UA_eff * T_amb_cooling

        Ts_steady_state = coeff_B / (coeff_A + 1e-6)
        tau_solid = (M_solid_bed * Cp_solid) / (coeff_A + 1e-6)

        Ts_cool_next = Ts_steady_state + (Ts_cool_curr - Ts_steady_state) * np.exp(-dt_sec / tau_solid)
        Tg_cool_next = (W_g * T_amb_cooling + h_gs_cool * A_s_cool * Ts_cool_next) / (W_g + h_gs_cool * A_s_cool + 1e-6)

        x_next["Ts_Cooling"] = np.clip(Ts_cool_next, T_amb_cooling, x_next["Ts_burning"])
        x_next["Tg_Cooling"] = np.clip(Tg_cool_next, T_amb_cooling, x_next["Ts_burning"])

        # GLOBAL BALANCE & YIELD
        Q_total_out = m_gas_pre * 1150.0 * x_next["Tg_preheater"]
        x_next["Energy_Residual"] = Q_in_total - Q_total_out

        loss_on_ignition = 1.0 - x_next["Clinker_yield"]
        Clinker_output_rate = x_next["Feed_rate"] * (1.0 - loss_on_ignition)
        x_next["Clinker_output"] = (0.95 * x.get("Clinker_output", Clinker_output_rate)) + (0.05 * Clinker_output_rate)

        x_next["t"] = t + self.dt
        x_next["dTg_burning"] = (x_next["Tg_burning"] - x["Tg_burning"]) / self.dt

        # CO ppm
        oxygen_deficit = max(0.0, 6.0 - x_next["O2"])
        target_CO = 20.0 + 800.0 * oxygen_deficit * (x_next["Fuel_rate"] / 6.0) * np.exp(-x_next["Tg_burning"] / 1200.0)
        x_next["CO_ppm"] = x["CO_ppm"] + 0.25 * (target_CO - x["CO_ppm"])

        # Reactions
        x_next["P_calcination"] = -406.0 + 226.0 * np.exp(-t / 20.0)

        T_calc = x["Ts_calcination"] + 273.15
        k_calc = 1e7 * np.exp(-160000.0 / (8.314 * T_calc)) if T_calc > 873.15 else 0.0
        x_next["CaCO3"] = x["CaCO3"] * np.exp(-k_calc * self.dt)

        dCaCO3 = x["CaCO3"] - x_next["CaCO3"]
        CaO_generated = dCaCO3 * 0.5603
        x_next["dCaO_calcination"] = CaO_generated / self.dt if self.dt > 0 else 0.0

        T_burn = x["Ts_burning"] + 273.15

        # dC2S (Belite)
        if x["SiO2"] <= 1e-6 or T_burn < 1000.0:
            x_next["dC2S"] = 0.0
        else:
            kinetic = 1.0 - np.exp(-(50000000.0 * np.exp(-170000.0 / (8.314 * T_burn))) * self.dt)
            x_next["dC2S"] = max(0.0, min(x["SiO2"] / 0.3488, (x["CaO"] + CaO_generated) / 0.6512) * kinetic)

        # dC3S (Alite)
        if x["C2S"] <= 1e-6 or T_burn < 1473.15:
            x_next["dC3S"] = 0.0
        else:
            kinetic = 1.0 - np.exp(-(228000000.0 * np.exp(-200000.0 / (8.314 * T_burn))) * self.dt)
            x_next["dC3S"] = max(0.0, min(x["C2S"] / 0.7544, max(0.0, x["CaO"] + CaO_generated - x_next["dC2S"] * 0.6512) / 0.2456) * kinetic)

        # dC3A
        if x["Al2O3"] <= 1e-6 or T_burn < 1100.0:
            x_next["dC3A"] = 0.0
        else:
            kinetic = 1.0 - np.exp(-(100000.0 * np.exp(-120000.0 / (8.314 * T_burn))) * self.dt)
            lim_ca = max(0.0, x["CaO"] + CaO_generated - x_next["dC2S"] * 0.6512 - x_next["dC3S"] * 0.2456) / 0.6227
            x_next["dC3A"] = max(0.0, min(x["Al2O3"] / 0.3773, lim_ca) * kinetic)

        # dC4AF
        if x["Fe2O3"] <= 1e-6 or T_burn < 1100.0:
            x_next["dC4AF"] = 0.0
        else:
            kinetic = 1.0 - np.exp(-(200000.0 * np.exp(-150000.0 / (8.314 * T_burn))) * self.dt)
            lim_al = max(0.0, x["Al2O3"] - x_next["dC3A"] * 0.3773) / 0.2098
            lim_ca = max(0.0, x["CaO"] + CaO_generated - x_next["dC2S"] * 0.6512 - x_next["dC3S"] * 0.2456 - x_next["dC3A"] * 0.6227) / 0.4616
            x_next["dC4AF"] = max(0.0, min(x["Fe2O3"] / 0.3286, lim_al, lim_ca) * kinetic)

        x_next["C2S"] = max(0.0, x["C2S"] + x_next["dC2S"] - (x_next["dC3S"] * 0.7544))
        x_next["C3S"] = x["C3S"] + x_next["dC3S"]
        x_next["C3A"] = x["C3A"] + x_next["dC3A"]
        x_next["C4AF"] = x["C4AF"] + x_next["dC4AF"]

        x_next["SiO2"] = max(0.0, x["SiO2"] - (x_next["dC2S"] * 0.3488))
        x_next["Al2O3"] = max(0.0, x["Al2O3"] - (x_next["dC3A"] * 0.3773) - (x_next["dC4AF"] * 0.2098))
        x_next["Fe2O3"] = max(0.0, x["Fe2O3"] - (x_next["dC4AF"] * 0.3286))

        CaO_consumed = (x_next["dC2S"] * 0.6512) + (x_next["dC3S"] * 0.2456) + (x_next["dC3A"] * 0.6227) + (x_next["dC4AF"] * 0.4616)
        x_next["CaO"] = max(0.0, x["CaO"] + CaO_generated - CaO_consumed)
        x_next["CO2"] = 80.0 - (x_next["CaCO3"] + x_next["CaO"])

        x_next["LSF"] = x_next["CaO"] / (2.8 * x_next["SiO2"] + 1.2 * x_next["Al2O3"] + 0.65 * x_next["Fe2O3"] + 1e-9)
        x_next["Mass_Balance_Error"] = abs((x_next["CaO"] - x["CaO"]) - (CaO_generated - CaO_consumed))

        x_next["Q_acc"] = x_next["Clinker_output"] * 150
        x_next["Q_loss"] = 0.0016 * 5.67e-8 * 0.8 * 110 * (x_next["Tg_burning"] ** 4 - 25**4)
        x_next["Q_reaction"] = x_next["Feed_rate"] * (1 - x_next["Clinker_yield"]) * 3.2
        x_next["Q_out"] = x_next["Q_acc"] + x_next["Q_loss"] + x_next["Q_reaction"]
        x_next["Q_gas"] = x_next["Fuel_rate"] * 1.1 * (x_next["Tg_burning"] - 25)

        clinker_safe = max(x_next["Clinker_output"], 1e-6)
        x_next["Normalized_Energy_Index"] = x_next["Q_in"] / clinker_safe
        x_next["Global_Energy_Closure"] = x_next["Q_in"] - x_next["Q_out"] + x_next["Q_reaction"] - x_next["Q_acc"]
        
        q_in_safe = max(x_next["Q_in"], 1e-6)
        x_next["Energy_error"] = (x_next["Global_Energy_Closure"] / q_in_safe) * 100

        if t == 0:
            x_next["SCALE"] = 1.0
        return x_next


def run_simulation():
    # --- MÜHENDİSLİK YAPILANDIRMASI (Tanımlanmayan Değişkenler Eklendi) ---
    sim_duration = 100.0  # Toplam simülasyon süresi (saat)
    dt = 0.05             # Fiziksel entegrasyon zaman adımı (saat)
    reporting_dt = 1 / 6  # Veri kaydetme frekansı (10 dakika)
    
    executor = StepExecutor(dt=dt)
    STEPS_PER_REPORT = max(1, int(reporting_dt / dt))
    N_total_reports = int(sim_duration / reporting_dt)

    REGIME_FEED_MULT = {
        "R1_HEATING_STABILIZATION": 1.0,
        "R2_EARLY_CALCINATION": 1.0,
        "R3_ACTIVE_CALCINATION": 1.0,
        "R4_TRANSITION_TO_CLINKERIZATION": 1.0,
        "R5_EARLY_CLINKERIZATION": 1.0,
        "R6_STEADY_CLINKERIZATION": 1.0,
        "R7_FUEL_SWITCH_TRANSIENT": 1.0,
        "R8_RESTABILIZATION": 1.0,
    }

    # Başlangıç Koşulları
    x_current = KilnState()
    x_current.Fuel_rate, x_current.O2, x_current.CO_ppm = 2.5, 6.0, 900.0
    x_current.Tg_preheater, x_current.Tg_calcination, x_current.Tg_burning, x_current.Tg_Cooling = 400.0, 848.37, 1245.08, 250.0
    x_current.Air_flow, x_current.Cooling_air_flow, x_current.ID_fan_speed = 45100, 172440.0, 900.0
    x_current.Feed_rate, x_current.Kiln_solid_out, x_current.Material_acc = 71.00, 0.1, 0.0
    x_current.Ts_preheater, x_current.Ts_calcination, x_current.Ts_burning, x_current.Ts_Cooling = 100.0, 802.02, 1060.03, 1450.0
    x_current.CaCO3, x_current.CaO, x_current.CO2 = 80.0, 1e-6, 1e-6
    x_current.SiO2, x_current.Al2O3, x_current.Fe2O3 = 13.0, 4.0, 3.0
    x_current.C2S, x_current.C3S, x_current.C3A, x_current.C4AF = 1e-6, 1e-6, 1e-6, 1e-6
    x_current.kiln_rpm = 1.0
    x_current.Residence = KilnPhysicsEngine.get_residence_time(1.0)
    x_current.Petcoke, x_current.Alternative_Fuel, x_current.Lignite_Coal = 0.03, 0.07, 0.90
    x_current.Material_acc, x_current.Clinker_output, x_current.Kiln_solid_out = 15.0, 11.13, 12.50
    x_current.P_preheater, x_current.P_calcination, x_current.P_burning = 12.0, 5.5, 1.2
    x_current.Damper_position = 33.0
    x_current.Q_in, x_current.Q_out, x_current.Q_acc, x_current.Q_loss, x_current.Q_reaction = 35575.0, 24285.0, 1669.0, 22600.0, 15.175
    x_current.Q_gas, x_current.Q_clinker = 3524.0, 15000.0
    x_current.Clinker_yield = 0.65
    x_current.Normalized_Energy_Index, x_current.Global_Energy_Closure, x_current.Energy_error = 1.0, 1.0, 0.0

    def input_layer(t: float, regime: str, initial_state: KilnState = None) -> dict:
        if t < 1e-4 and initial_state is not None:
            return {
                "Petcoke": initial_state.Petcoke,
                "Alternative_Fuel": initial_state.Alternative_Fuel,
                "Feed_rate": initial_state.Feed_rate,
            }
        D = 0.03 + (0.1 - 0.03) * (1 - np.exp(-t / 80.0))
        E = 0.07 + (0.14 - 0.07) * (1 - np.exp(-t / 100.0))
        base_feed = 132.0 if t >= 72.0 else 72.0 + 60.0 * ((1.0 / (1.0 + np.exp(-0.065 * (t - 36.0)))) - 0.09) / 0.82
        
        return {
            "Petcoke": D,
            "Alternative_Fuel": E,
            "Feed_rate": base_feed * REGIME_FEED_MULT.get(regime, 1.0),
        }

    def determine_regime(t):
        # Dinamik DataFrame olmadığı için zaman bağımlı rejim jeneratörü
        if t < 20.0: return "R1_HEATING_STABILIZATION"
        elif t < 40.0: return "R3_ACTIVE_CALCINATION"
        else: return "R6_STEADY_CLINKERIZATION"

    # ==============================
    # STABİL SİMÜLASYON DÖNGÜSÜ (List Accumulation)
    # ==============================
    simulation_records = []
    sim_time = 0.0

    for step_idx in range(N_total_reports):
        current_regime = determine_regime(sim_time)
        inputs = input_layer(sim_time, current_regime, initial_state=x_current)
        inputs["regime"] = current_regime

        # Alt adımlarda diferansiyel çözüm
        for sub in range(STEPS_PER_REPORT):
            step_time = sim_time + (dt * (sub + 1))
            x_current = executor.perform_step(x_current, step_time, inputs=inputs)

        # Raporlama zamanı (veri biriktirme)
        sim_time += reporting_dt
        simulation_records.append(asdict(x_current))

    # O(N) karmaşıklığı ile tek seferde Dataframe oluşturma (Güvenli Yöntem)
    df_results = pd.DataFrame(simulation_records)
    df_results.to_csv("engine.csv", index=False)
    print(f"Simülasyon tamamlandı. Toplam {len(df_results)} satır veri 'engine.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    run_simulation()