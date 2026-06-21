import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

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
    Fuel_rate: float = 4.0
    Tg_preheater: float = 350.0
    Ts_preheater: float = 100.0
    Tg_calcination: float = 850.0
    Ts_calcination: float = 780
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
        rpm = max(0.1, kiln_rpm)
        rpm_eff = rpm / (0.26 + rpm)
        filling = max(0.01, filling_rate)
        filling_factor = (0.08 / filling) ** 0.3
        v_axial = (5.61 * D * rpm_eff * (1.5 + 44.8 * slope)) * filling_factor
        return (L / (v_axial + 1e-6)) * 60.0


# 3. İcra Edici
class StepExecutor:
    def __init__(self, dt=0.01):
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
            x_next["Fuel_rate"] = 4.0 + 1.5 * (1.0 - np.exp(-t / 35.0))
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

        
# --- ENERJİ DAĞILIMI (Cascade & Source-Based) ---
        m_gas_pre = x_next["Air_flow"] * 0.001293
        m_solid_pre = x_next["Feed_rate"]
        Q_in_total = x_next["Q_in"] * 1000.0

        # Tersiyer Hava Desteği (Dinamik Enerji Girişi)
        m_air_tertiary = x_next.get("Tertiary_air_flow", 0.0) * 1.293 / 3600.0 
        Q_tertiary_cost = m_air_tertiary * 1150.0 * 800.0  # Burning Zone'dan çekilen maliyet
        Q_tertiary = m_air_tertiary * 1150.0 * 800.0 

        feed_factor = x_next["Feed_rate"] / 100.0

        # Enerji Havuzu: Tersiyer maliyeti düşülmüş net Burning Zone havuzu
        Q_burning_pool = (Q_in_total * 0.56) + Q_tertiary - Q_tertiary_cost
        Q_calcination_budget = Q_in_total * (0.26 * min(feed_factor, 1.2))
        Q_preheater_budget = Q_in_total * 0.18

        dt_sec = self.dt * 3600.0  

        # --- BURNING ZONE ---
        Cp_g_burn, Cp_s_burn, Cp_w_burn = 1250.0, 1150.0, 1000.0
        A_c_burn, h_c_burn = 13.85, 0.05
        A_s_burn, A_w_burn, A_ws_burn = 82.0, 70.0, 57.0

        m_air_s = (x_next["Air_flow"] * 1.293) / 3600.0
        m_solid_s = (x_next["Feed_rate"] * 1000.0) / 3600.0

        C_gas_total = 220.0 * Cp_g_burn
        C_solid_total = 6500.0 * Cp_s_burn
        C_wall_total = 15000.0 * Cp_w_burn

        Tg_curr, Ts_curr, Tw_curr = x.get("Tg_burning", 1450.0), x.get("Ts_burning", 1400.0), x.get("Tw_burning", 1300.0)
        h_gs_burn, h_gw_burn, h_ws_burn = 1450.0, 350.0, 400.0 

        a_gas_burn = (h_gs_burn * A_s_burn) + (h_gw_burn * A_w_burn) + (h_c_burn * 1000.0 * A_c_burn) + (m_air_s * Cp_g_burn)
        b_gas_burn = (Q_burning_pool * 1.1) + (h_gw_burn * A_w_burn * Tw_curr) + (h_c_burn * 1000.0 * A_c_burn * 30.0) + (m_air_s * Cp_g_burn * 400.0) + (h_gs_burn * A_s_burn * Ts_curr)
        Tg_next_burn = (C_gas_total * Tg_curr + dt_sec * b_gas_burn) / (C_gas_total + dt_sec * a_gas_burn)

        # Duvar Fazı
        Q_loss_wall = 15.0 * A_w_burn * (Tw_curr - 25.0) 
        Q_wall_to_solid = h_ws_burn * A_ws_burn * (Tw_curr - Ts_curr)
        Tw_next_burn = Tw_curr + (dt_sec / C_wall_total) * (h_gw_burn * A_w_burn * (Tg_next_burn - Tw_curr) - Q_loss_wall - Q_wall_to_solid)

        # Katı Fazı
        Q_exo_base = min(350000.0, m_solid_s * 500000.0 * (1.0 + max(0.0, Ts_curr - 1200.0) / 250.0))
        Q_exo_W = Q_exo_base * float(Ts_curr > 1200.0)
        a_sol_burn = (h_gs_burn * A_s_burn) + (h_ws_burn * A_ws_burn) + (m_solid_s * Cp_s_burn)
        b_sol_burn = Q_exo_W + (m_solid_s * Cp_s_burn * 900.0) + (h_gs_burn * A_s_burn * Tg_next_burn) + (h_ws_burn * A_ws_burn * Tw_next_burn)
        Ts_next_burn = (C_solid_total * Ts_curr + dt_sec * b_sol_burn) / (C_solid_total + dt_sec * a_sol_burn)
        
        x_next["Tg_burning"], x_next["Ts_burning"], x_next["Tw_burning"] = Tg_next_burn, Ts_next_burn, Tw_next_burn

        # --- CALCINATION ZONE ---
        h_gs, A_s = 320.0, 950.0 
        m_solid_calc = x_next["Feed_rate"]
        Ts_calc_curr = x.get("Ts_calcination", 800.0)
        Tg_calc_curr = x.get("Tg_calcination", 900.0)

        # Kalsinasyon Yükü (Clamp: Budget ile sınırla)
        dynamic_rate = 0.05 + 0.50 * (1.0 / (1.0 + np.exp(-0.08 * (Ts_calc_curr - 850.0))))
        Q_calc_raw = (m_solid_calc / 3600.0) * dynamic_rate * 1700.0 * 1000.0
        Q_calc_load = min(Q_calc_raw, Q_calcination_budget / dt_sec) # Bütçe ile kısıtlı

        # Enerji Akışı
        m_air_kiln = (x_next["Air_flow"] * 1.293) / 3600.0
        Q_from_burning = m_air_kiln * 1150.0 * (Tg_next_burn * 0.78)
        Q_from_tertiary = m_air_tertiary * 1150.0 * 800.0
        Q_calc_total_in = Q_from_burning + Q_from_tertiary

        # Katı Güncelleme
        C_solid_calc_total = (m_solid_calc * 1000.0 * 1050.0) / 3600.0
        b_s = (h_gs * A_s * Tg_calc_curr) + (Q_calcination_budget / 3600.0) - Q_calc_load - ((m_solid_calc * 1000.0 / 3600.0) * 1050.0 * (Ts_calc_curr - 25.0))
        x_next["Ts_calcination"] = (C_solid_calc_total * Ts_calc_curr + dt_sec * b_s) / (C_solid_calc_total + dt_sec * (h_gs * A_s))

        # Gaz Güncelleme
        C_gas_calc_total = 100.0 * 1150.0 
        a_gas_calc = ((m_air_kiln + m_air_tertiary) * 1150.0) + (4.0 * 13.85) + (h_gs * A_s)
        b_gas_calc = Q_calc_total_in + Q_calcination_budget + (4.0 * 13.85 * x.get("Tg_preheater", 350.0)) + (h_gs * A_s * x_next["Ts_calcination"])
        x_next["Tg_calcination"] = (C_gas_calc_total * Tg_calc_curr + dt_sec * b_gas_calc) / (C_gas_calc_total + dt_sec * a_gas_calc)


        # PREHEATER
        # 1. Sıcaklık farkı ve teorik transfer potansiyeli
        delta_T_pre = x_next["Tg_calcination"] - x.get("Ts_preheater", 25.0)
        UA_pre = m_gas_pre * 1150.0 * 0.24
        
        # 2. Enerji Bütçesi ile Kısıtlı Isı Transferi
        # Teorik olarak transfer edilebilecek ısıyı hesapla
        theoretical_heat = UA_pre * delta_T_pre
        
        # Q_preheater_budget (yukarıdaki tanımın) ile teorik ısıyı kıyasla
        # Transfer, bütçeyi (limit) geçemez
        heat_transferred = min(theoretical_heat, Q_preheater_budget)
        
        # 3. Gaz Entalpi Dengesi:
        gas_capacity = m_gas_pre * 1150.0 + 1e-6
        x_next["Tg_preheater"] = x_next["Tg_calcination"] - (heat_transferred / gas_capacity)
        
        # 4. Katı Atalet Dengesi:
        # Bütçeyle kısıtlanmış heat_transferred kullanılıyor
        C_solid_pre = (m_solid_pre * 3000.0) * 1150.0 # Kapasite (kJ/C)
        
        # Isı transferinin katı sıcaklığına etkisi: dTs = Q / C
        # Zaman adımı (dt_sec) ile entegre ediyoruz
        dTs_pre = (heat_transferred / C_solid_pre) * dt_sec
        x_next["Ts_preheater"] = x.get("Ts_preheater", 25.0) + dTs_pre
        
        
        
        # --- COOLING ZONE (Ataletli Newton Soğuma & Isı Transferi) ---
        # Ön tanımlar ve varsayılanlar
        Tamb_solid = 130.0
        Tamb_gas = 510.0
        Ts_cool_curr = x.get("Ts_Cooling", x_next.get("Ts_burning", 1450.0))
        Tg_cool_curr = x.get("Tg_Cooling", x_next.get("Tg_burning", 1490.0))

        # Fiziksel Atalet Parametreleri
        tau_klinker = 9700.0
        tau_gas = 3800.0

        # Isı transfer katsayısı (Katıdan gaza transfer verimliliği)
        h_transfer = 0.15

        # Isıl kapasiteler (J/K)
        Cp_solid = 1150.0
        Cp_gas = 1050.0

        m_solid = (x["Feed_rate"] * 1000.0) / 3600.0
        m_gas = x["Cooling_air_flow"] * 0.001293

        C_solid = max(m_solid * Cp_solid, 1.0)
        C_gas = max(m_gas * Cp_gas, 1.0)

        # 1. Klinker Sıcaklığı (Katı Ataletli Güncelleme)
        # Katı, ortamla soğurken aynı zamanda gaza da ısı verir
        dTs_cool = (Tamb_solid - Ts_cool_curr) * (1.0 - np.exp(-dt_sec / tau_klinker))

        Q_transfer = (
            h_transfer
            * (Ts_cool_curr - Tg_cool_curr)
            * dt_sec
        )

        dTs_transfer = -Q_transfer / C_solid
        Ts_cool_next = Ts_cool_curr + dTs_cool + dTs_transfer

        # 2. Gaz Sıcaklığı (Katıya göre daha dinamik)
        # Gaz, ortamla soğurken katıdan aldığı ısı ile ısınır
        dTg_cool = (Tamb_gas - Tg_cool_curr) * (1.0 - np.exp(-dt_sec / tau_gas))

        dTg_transfer = Q_transfer / C_gas
        Tg_cool_next = Tg_cool_curr + dTg_cool + dTg_transfer

        # 3. Sonuçları güvenli bir şekilde kaydet
        x_next["Ts_Cooling"] = Ts_cool_next
        x_next["Tg_cool_next"] = Tg_cool_next
        x_next["Tg_Cooling"] = Tg_cool_next

        
    
    
    

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


import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass

def run_simulation():
    # --- MÜHENDİSLİK YAPILANDIRMASI ---
    sim_duration = 72.0      # Toplam simülasyon süresi (saat)
    dt = 0.05                # Fiziksel entegrasyon zaman adımı (saat)
    reporting_dt = 1/6       # Veri kaydetme frekansı (10 dakika)
    
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

    # --- Başlangıç Koşulları (t=0.0) ---
    x_current = KilnState()
    x_current.t = 0.0
    x_current.Fuel_rate, x_current.O2, x_current.CO_ppm = 4.0, 6.0, 900.0
    x_current.Tg_preheater, x_current.Tg_calcination, x_current.Tg_burning, x_current.Tg_Cooling = 650.0, 950.0, 1450.0, 1550.0
    x_current.Air_flow, x_current.Cooling_air_flow, x_current.ID_fan_speed = 45100, 80000.0, 900.0
    x_current.Feed_rate, x_current.Kiln_solid_out, x_current.Material_acc = 71.00, 0.1, 0.0
    x_current.Ts_preheater, x_current.Ts_calcination, x_current.Ts_burning, x_current.Ts_Cooling = 300.0, 850.0, 1420.0, 1450.0
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

    # Yardımcı Fonksiyonlar
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
        if t < 20.0: return "R1_HEATING_STABILIZATION"
        elif t < 40.0: return "R3_ACTIVE_CALCINATION"
        else: return "R6_STEADY_CLINKERIZATION"

    # ========================================================
    # DÜZELTİLMİŞ SİMÜLASYON DÖNGÜSÜ (IC Korumalı)
    # ========================================================
    simulation_records = []
    
    # 1. Başlangıç Koşulunu t=0.0 için kaydet
    x_current.t = 0.0
    simulation_records.append(asdict(x_current))
    
    sim_time = 0.0

    for step_idx in range(N_total_reports):
        # Rejim ve girişleri belirle
        current_regime = determine_regime(sim_time)
        inputs = input_layer(sim_time, current_regime, initial_state=x_current)
        inputs["regime"] = current_regime

        # Alt adımlarda diferansiyel çözüm
        for sub in range(STEPS_PER_REPORT):
            step_time = sim_time + (dt * (sub + 1))
            x_current = executor.perform_step(x_current, step_time, inputs=inputs)

        # Zamanı ilerlet ve kaydet
        sim_time += reporting_dt
        x_current.t = sim_time
        simulation_records.append(asdict(x_current))

    # Dataframe oluşturma ve dosya kaydı
    df_results = pd.DataFrame(simulation_records)
    df_results.to_csv("engine.csv", index=False)
    print(f"Simülasyon başarıyla tamamlandı. t=0.0 başlangıçlı {len(df_results)} satır veri kaydedildi.")

if __name__ == "__main__":
    run_simulation()