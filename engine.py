import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, fields

# 1. Veri Yapısı
@dataclass
class KilnState:
    t: float = 0.0
    Regime: str = "R1_HEATING_STABILIZATION"
    Lignite_Coal: float = 0.0; Petcoke: float = 0.0; Alternative_Fuel: float = 0.0
    Feed_rate: float = 40.0; Kiln_solid_out: float = 0.0; Material_acc: float = 15.0
    Clinker_output: float = 0.0; Air_flow: float = 45000.0
    Cooling_air_flow: float = 80000.0; ID_fan_speed: float = 900.0; Fuel_rate: float = 2.5
    Tg_preheater: float = 350.0; Ts_preheater: float = 80.0
    Tg_calcination: float = 850.0; Ts_calcination: float = 800.0; Tg_burning: float = 1200.0
    Ts_burning: float = 1050.0; Tg_Cooling: float = 25.0; Ts_Cooling: float = 400.0
    O2: float = 3.5; CO_ppm: float = 0.0; P_preheater: float = -120.0; P_calcination: float = 0.0
    P_burning: float = 0.0; Damper_position: float = 33.0; kiln_rpm: float = 1.0
    Residence: float = 0.0; Q_in: float = 0.0
    CaCO3: float = 80.0; CaO: float = 1e-6; CO2: float = 1e-6; SiO2: float = 13.0
    Al2O3: float = 4.0; Fe2O3: float = 3.0; C2S: float = 1e-6; C3S: float = 1e-6
    C3A: float = 1e-6; C4AF: float = 1e-6; Q_out: float = 0.0; Q_acc: float = 0.0
    Q_loss: float = 0.0; Q_reaction: float = 0.0; Q_gas: float = 0.0; Q_clinker: float = 0.0
    Clinker_yield: float = 0.89; dTg_burning: float = 0.0; Normalized_Energy_Index: float = 1.0
    Global_Energy_Closure: float = 1.0; Energy_error: float = 0.0; dCaO_calcination: float = 0.0
    LSF: float = 0.0; dC3A: float = 0.0; dC4AF: float = 0.0; dC2S: float = 0.0; dC3S: float = 0.0
    Mass_Balance_Error: float = 0.0; SCALE: float = 1.0

    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): setattr(self, key, value)
    def get(self, key, default): return getattr(self, key, default)
    def copy(self): return KilnState(**asdict(self))

# 2. Dinamik Sütun ve Rejim Yönetimi
columns = [f.name for f in fields(KilnState)]

regime_series = [
    ("R1_HEATING_STABILIZATION", 50),
    ("R2_EARLY_CALCINATION", 50),
    ("R3_ACTIVE_CALCINATION", 60),
    ("R4_TRANSITION_TO_CLINKERIZATION", 40),
    ("R5_EARLY_CLINKERIZATION", 30),
    ("R6_STEADY_CLINKERIZATION", 130),
    ("R7_FUEL_SWITCH_TRANSIENT", 20),
    ("R8_RESTABILIZATION", 54),
]

regime_list = []
for r, n in regime_series:
    regime_list.extend([r] * n)

# 3. DataFrame İnşaası
N = len(regime_list)
df = pd.DataFrame(0.0, index=np.arange(N), columns=columns)

# 4. Zaman ve Rejim Değerlerinin Atanması
dt = 0.05
df["t"] = np.arange(N) * dt
df["Regime"] = regime_list

print(f"DataFrame başarıyla oluşturuldu. Boyut: {df.shape}")

    # Residence time calculation is managed via KilnPhysicsEngine.get_residence_time


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
        self.dt = dt

    def perform_step(self, x: KilnState, t: float, inputs: dict = None) -> KilnState:
            # Girdileri güvenli bir şekilde al
        inputs = inputs or {}
            
            # Yeni state'i kopyala
        x_next = x.copy()
            
            # Regime bilgisini al
        regime = inputs.get("regime", "R1_HEATING_STABILIZATION")
            
            # Kontrol girdilerini güncelle (Eğer inputs içinde varsa)
            # Bu kısım eksikti, bu sayede dışarıdan gelen (MPC/RL) komutlar işlenir
        for key in ["Fuel_rate", "Feed_rate", "Air_flow", "kiln_rpm", "Petcoke", "Alternative_Fuel"]:
            if key in inputs:
                setattr(x_next, key, inputs[key])
        
        # ---------------------------------------------------------
        # FUEL COMPOSITION
        # ---------------------------------------------------------
        x_next["Petcoke"] = inputs.get("Petcoke", x.get("Petcoke", 0.0))
        x_next["Alternative_Fuel"] = inputs.get("Alternative_Fuel", x.get("Alternative_Fuel", 0.0))
        x_next["Lignite_Coal"] = max(
            0.0,
            1.0 - x_next["Petcoke"] - x_next["Alternative_Fuel"]
        )

        # ---------------------------------------------------------
        # CONTROL INPUTS (MPC/RL override veya Open-Loop fallback)
        # ---------------------------------------------------------
        if "Feed_rate" in inputs:
            x_next["Feed_rate"] = inputs["Feed_rate"]
        else:
            feed = x.get("Feed_rate", 40.0)
            x_next["Feed_rate"] = feed + (120.0 - feed) * 0.0005 * dt

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

        # RPM
        if "kiln_rpm" in inputs:
            x_next["kiln_rpm"] = max(0.1, inputs["kiln_rpm"])
        else:
            rpm_current = x.get("kiln_rpm", 1.0)
            rpm_setpoint = 2.4
            alpha = 0.005
            rpm_next = rpm_current + alpha * (rpm_setpoint - rpm_current)
            x_next["kiln_rpm"] = max(0.1, rpm_next)

        # Residence
        res_min = KilnPhysicsEngine.get_residence_time(x_next["kiln_rpm"])
        x_next["Residence"] = res_min

        # Mass balance
        mat_acc = x.get("Material_acc", 15.0)

        kiln_out = mat_acc / (res_min / 60.0 + 1e-6)
        x_next["Kiln_solid_out"] = kiln_out

        x_next["Material_acc"] = mat_acc + (
            x_next["Feed_rate"] / 1.55 - kiln_out
        ) * (dt / 60.0)
        
        # O2
        air_fuel_ratio = x_next["Air_flow"] / (f_rate + 1e-6)
        o2_target = 2.5 + 4.5 * np.exp(-1.2 / (air_fuel_ratio / 20000.0 + 1e-3))

        x_next["O2"] = x.get("O2", 3.5) + 0.15 * (o2_target - x.get("O2", 3.5))

        # -------------------------
        # Q_in (kJ/kg) -> Bağımlı değişkenlerin önünde hesaplanmalı
        # -------------------------
        x_next["Q_in"] = (
            (x_next["Lignite_Coal"] * 15000
            + x_next["Petcoke"] * 30000
            + x_next["Alternative_Fuel"] * 18000)
            * x_next["Fuel_rate"]
            * np.exp(-((x_next["O2"] - 3.5) ** 2) / 25)
        )
        
        # -------------------------
        # P_preheater (Pa)
        # -------------------------
        x_next["P_preheater"] = -269.0 + 149.0 * np.exp(-t / 15.0)

        #===================================== Operational Zones =====================================

        # -------------------------------------------------------------
        # ZON HESAPLAMALARI ÖNCESİ GEREKLİ TANIMLAR
        # -------------------------------------------------------------
        # Kütle akışlarını x_next içinden veya hesaplayarak tanımlayın:
        # Örnek: Eğer x_next içinde bu anahtarlar varsa:
        m_gas_pre = x_next.get("Air_flow", 0.0) * 0.001293
        m_solid_pre = x_next.get("Feed_rate", 0.0)

        # Veya daha önce hesapladığınız bir değer varsa (örn: m_air_s, m_solid_s gibi) 
        # bunları burada m_gas_pre/m_solid_pre'ye atayabilirsiniz.

        # -------------------------------------------------------------
        # GLOBAL ENERGY SOURCE (TEK NOKTA)
        # -------------------------------------------------------------
        Q_in_total = x_next["Q_in"] * 1000.0  # W
        
        # -------------------------------------------------------------
        # DİNAMİK ENERJİ PAYLAŞIMI (Energy Allocation)
        # -------------------------------------------------------------
        # Fırın içindeki toplam ısıl yükü besleme hızıyla ilişkilendiriyoruz
        feed_factor = x_next.get("Feed_rate", 10.0) / 100.0 
        
        # Yanma zonu (Burning) en büyük payı alır (Alev enerjisi)
        Q_burning = Q_in_total * 0.60 
        
        # Kalsinasyon zonu (Reaksiyon yüküne bağlı kısmi pay)
        # Eğer besleme yüksekse kalsinasyon yükü artar
        Q_calcination = Q_in_total * (0.30 * min(feed_factor, 1.2))
        
        # Preheater zonu (Kalan enerji recovery için)
        Q_preheater = Q_in_total - Q_burning - Q_calcination

        # -------------------------------------------------------------
        # 1. ZON: BURNING ZONE (PRIMARY ENERGY SOURCE)
        # -------------------------------------------------------------
        Cp_g_burn = 1250.0  
        Cp_s_burn = 1150.0  
        A_c_burn = 13.85    
        h_c_burn = 0.05    
        A_s_burn = 25.0 

        m_air_s = (x_next["Air_flow"] * 1.293) / 3600.0  
        m_solid_s = (x_next["Feed_rate"] * 1000.0) / 3600.0 
        C_gas_total = 120.0 * Cp_g_burn       
        C_solid_total = 4500.0 * Cp_s_burn  

        Tg_curr = x.get("Tg_burning", 1200.0)
        Ts_curr = x.get("Ts_burning", 1050.0)
        h_gs_burn = 1500.0 if Tg_curr > 1000.0 else 500.0 

        dt_sec = dt * 3600.0
        
        # Gaz Fazı Dengesi
        a_gas_burn = (h_gs_burn * A_s_burn) + (h_c_burn * 1000.0 * A_c_burn) + (m_air_s * Cp_g_burn)
        b_gas_burn = (Q_burning) + (h_c_burn * 1000.0 * A_c_burn * 30.0) + (m_air_s * Cp_g_burn * 400.0) + (h_gs_burn * A_s_burn * Ts_curr)
        Tg_next_burn = (C_gas_total * Tg_curr + dt_sec * b_gas_burn) / (C_gas_total + dt_sec * a_gas_burn)

        # Katı Faz Dengesi
        Q_exo_W = min(350000.0, m_solid_s * 500000.0 * (1.0 + max(0, Ts_curr - 1200.0)/250.0)) if Ts_curr > 1200.0 else 0.0
        a_sol_burn = (h_gs_burn * A_s_burn) + (m_solid_s * Cp_s_burn)
        b_sol_burn = Q_exo_W + (m_solid_s * Cp_s_burn * 900.0) + (h_gs_burn * A_s_burn * Tg_curr)
        Ts_next_burn = (C_solid_total * Ts_curr + dt_sec * b_sol_burn) / (C_solid_total + dt_sec * a_sol_burn)

        x_next["Tg_burning"] = Tg_next_burn
        x_next["Ts_burning"] = Ts_next_burn

        # -------------------------------------------------------------
        # 2. ZON: CALCINATION ZONE (Burning gazı ile beslenir)
        # -------------------------------------------------------------
        h_gs = 110.0
        A_s = 840.0
        m_solid_calc = x_next["Feed_rate"]
        Ts_calc_curr = x.get("Ts_calcination", 800.0)
        
        # Kalsinasyon yükü
        dynamic_rate = 0.05 + 0.10 * (1.0 / (1.0 + np.exp(-0.02 * (Ts_calc_curr - 1000.0))))
        reaction_throttle = 1.0 / (1.0 + np.exp(-0.08 * (Ts_calc_curr - 850.0)))
        Q_calc_load = (m_solid_calc / 3600.0) * dynamic_rate * reaction_throttle * 1700.0 * 1000.0
        
        # Katı Faz
        C_solid_calc_total = (m_solid_calc * 1000.0 * 1050.0) / 3600.0
        b_s = (h_gs * A_s * x.get("Tg_calcination", 900.0)) + (Q_calcination / 3600.0) - Q_calc_load - ((m_solid_calc * 1000.0 / 3600.0) * 1050.0 * (Ts_calc_curr - 25.0))
        x_next["Ts_calcination"] = min((C_solid_calc_total * Ts_calc_curr + dt_sec * b_s) / (C_solid_calc_total + dt_sec * (h_gs * A_s)), 1200.0)

        # Gaz Fazı (Burning'den gelen Tg_next_burn kullanıldı)
        m_air_calc = x_next["Air_flow"] * 0.001270
        a_gas_calc = (m_air_calc * 1150.0) + (4.0 * 13.85) + (h_gs * A_s)
        b_gas_calc = (m_air_calc * 1150.0 * Tg_next_burn) + (Q_calcination) + (4.0 * 13.85 * x.get("Tg_preheater", 350.0)) + (h_gs * A_s * x_next["Ts_calcination"])
        x_next["Tg_calcination"] = min((1000.0 * 1150.0 * x.get("Tg_calcination", 900.0) + dt_sec * (b_gas_calc - (m_air_calc * 1150.0 * x.get("Tg_calcination", 900.0)))) / (1000.0 * 1150.0 + dt_sec * a_gas_calc), 1300.0)

        # -------------------------------------------------------------
        # 3. ZON: PREHEATER ZONE (Calcination gazı ile beslenir)
        # -------------------------------------------------------------
        UA_pre = 570000
        # Gaz fazı (Calcination çıkışı ile besleme)
        a_gas_pre = (m_gas_pre * 1150.0) + UA_pre
        b_gas_pre = (m_gas_pre * 1150.0 * x_next["Tg_calcination"]) + (UA_pre * x.get("Ts_preheater", 25.0))
        x_next["Tg_preheater"] = (200000.0 * x.get("Tg_preheater", 350.0) + dt_sec * b_gas_pre) / (200000.0 + dt_sec * a_gas_pre)

        # Katı faz
        a_s_pre = (m_solid_pre * 1000.0) + UA_pre
        b_s_pre = (m_solid_pre * 1000.0 * 25.0) + (UA_pre * x_next["Tg_preheater"])
        x_next["Ts_preheater"] = (300000.0 * x.get("Ts_preheater", 25.0) + dt_sec * b_s_pre) / (300000.0 + dt_sec * a_s_pre)
        
        # -------------------------------------------------------------
        # 4. ZON: COOLING (İteratif Referans Modeli)
        # -------------------------------------------------------------
        # Önceki adımdaki sıcaklıkları al (eğer yoksa başlangıç değerlerini kullan)
        Tg_current = x.get("Tg_Cooling", 1550.0)
        Ts_current = x.get("Ts_Cooling", 1450.0)
        Tamb = 130.0
        
        # İterasyon katsayısı (dt'ye bağlı olarak değişebilir)
        alpha = 0.024 
        
        # Her adımda bir önceki değeri güncelleyerek ilerle
        Tg_next_cool = Tg_current - alpha * (Tg_current - Tamb)
        x_next["Tg_Cooling"] = Tg_next_cool
        
        Ts_next_cool = Ts_current - alpha * (Ts_current - Tamb)
        x_next["Ts_Cooling"] = Ts_next_cool




        # -------------------------------------------------------------
        # GLOBAL ENERGY BALANCE & KÜTLE ÇIKIŞI
        # -------------------------------------------------------------
        # Baca gazı çıkışı (Preheater çıkışı)
        Q_total_out = (m_gas_pre * 1150.0 * x_next["Tg_preheater"])
        x_next["Energy_Residual"] = Q_in_total - Q_total_out
        
        # Klinker Çıkış Hesabı
        loss_on_ignition = 0.38
        Clinker_output_rate = (x_next["Feed_rate"] * (1.0 - loss_on_ignition))
        x_next["Clinker_output"] = (0.95 * x.get("Clinker_output", Clinker_output_rate)) + (0.05 * Clinker_output_rate)
        
        # Energy Check
        Q_total_out = (m_gas_pre * 1150.0 * x_next["Tg_preheater"]) # Gaz fazı baca kaybı
        # Eğer Q_total_out, Q_in_total'den çok farklıysa sistemde enerji sızıntısı var demektir.
        x_next["Energy_Residual"] = Q_in_total - Q_total_out
        
        loss_on_ignition = 0.38
        
        Clinker_output_rate = (x_next["Feed_rate"] * (1.0 - loss_on_ignition))
        
        x_next["Clinker_output"] = (0.95 * x.get("Clinker_output", Clinker_output_rate)) + (0.05 * Clinker_output_rate)


        # -------------------------------------------------------------
        # TIME UPDATE
        # -------------------------------------------------------------
        x_next["t"] = t + dt  
                    
        # -------------------------
        # dTg_burning
        # -------------------------
        x_next["dTg_burning"] = (x_next["Tg_burning"] - x["Tg_burning"]) / dt

        # -------------------------
        # CO ppm
        # -------------------------
        oxygen_deficit = max(0.0, 6.0 - x_next["O2"])
        T_eff = x_next["Tg_burning"]
        temp_factor = np.exp(-T_eff / 1200.0)
        fuel_factor = x_next["Fuel_rate"] / 6.0
        target_CO = 20.0 + 800.0 * oxygen_deficit * fuel_factor * temp_factor

        x_next["CO_ppm"] = x["CO_ppm"] + 0.25 * (target_CO - x["CO_ppm"])
                
                
        #===================================== Reactions =====================================
        
        # -------------------------
        # P_calcination (Pa)
        # -------------------------
        x_next["P_calcination"] = -406.0 + 226.0 * np.exp(-t / 20.0)

        # ---------------------------------------------------------
        # 1. Calcination (CaCO3 -> CaO + CO2)
        # ---------------------------------------------------------
        T_calc = x["Ts_calcination"] + 273.15
        k_calc = 1e7 * np.exp(-160000.0 / (8.314 * T_calc)) if T_calc > 873.15 else 0.0
        x_next["CaCO3"] = x["CaCO3"] * np.exp(-k_calc * dt)
        
        dCaCO3 = x["CaCO3"] - x_next["CaCO3"]
        CaO_generated = dCaCO3 * 0.5603
        x_next["dCaO_calcination"] = CaO_generated / dt if dt > 0 else 0.0

        # ---------------------------------------------------------
        # 2. Clinkerization Kinetics (Burning Zone)
        # ---------------------------------------------------------
        T_burn = x["Ts_burning"] + 273.15
        
        # dC2S (Belite)
        if x["SiO2"] <= 1e-6 or T_burn < 1000.0:
            x_next["dC2S"] = 0.0
        else:
            arrhenius = np.exp(-170000.0 / (8.314 * T_burn))
            rate_term = 50000000.0 * arrhenius
            kinetic = 1.0 - np.exp(-rate_term * dt)
            limiting = min(x["SiO2"] / 0.3488, (x["CaO"] + CaO_generated) / 0.6512)
            x_next["dC2S"] = max(0.0, limiting * kinetic)

        # dC3S (Alite)
        if x["C2S"] <= 1e-6 or T_burn < 1473.15:
            x_next["dC3S"] = 0.0
        else:
            arrhenius = np.exp(-200000.0 / (8.314 * T_burn))
            rate_term = 228000000.0 * arrhenius
            kinetic = 1.0 - np.exp(-rate_term * dt)
            limiting = min(x["C2S"] / 0.7544, max(0.0, x["CaO"] + CaO_generated - x_next["dC2S"] * 0.6512) / 0.2456)
            x_next["dC3S"] = max(0.0, limiting * kinetic)

        # dC3A (Aluminate)
        if x["Al2O3"] <= 1e-6 or T_burn < 1100.0:
            x_next["dC3A"] = 0.0
        else:
            arrhenius = np.exp(-120000.0 / (8.314 * T_burn))
            rate_term = 100000.0 * arrhenius
            kinetic = 1.0 - np.exp(-rate_term * dt)
            lim_al = x["Al2O3"] / 0.3773
            lim_ca = max(0.0, x["CaO"] + CaO_generated - x_next["dC2S"] * 0.6512 - x_next["dC3S"] * 0.2456) / 0.6227
            limiting = min(lim_al, lim_ca)
            x_next["dC3A"] = max(0.0, limiting * kinetic)

        # dC4AF (Ferrite)
        if x["Fe2O3"] <= 1e-6 or T_burn < 1100.0:
            x_next["dC4AF"] = 0.0
        else:
            arrhenius = np.exp(-150000.0 / (8.314 * T_burn))
            rate_term = 200000.0 * arrhenius
            kinetic = 1.0 - np.exp(-rate_term * dt)
            lim_fe = x["Fe2O3"] / 0.3286
            lim_al = max(0.0, x["Al2O3"] - x_next["dC3A"] * 0.3773) / 0.2098
            lim_ca = max(0.0, x["CaO"] + CaO_generated - x_next["dC2S"] * 0.6512 - x_next["dC3S"] * 0.2456 - x_next["dC3A"] * 0.6227) / 0.4616
            limiting = min(lim_fe, lim_al, lim_ca)
            x_next["dC4AF"] = max(0.0, limiting * kinetic)

        # ---------------------------------------------------------
        # 3. Mass Balance Updates (Stoichiometrically Sound)
        # ---------------------------------------------------------
        x_next["C2S"] = max(0.0, x["C2S"] + x_next["dC2S"] - (x_next["dC3S"] * 0.7544))
        x_next["C3S"] = x["C3S"] + x_next["dC3S"]
        x_next["C3A"] = x["C3A"] + x_next["dC3A"]
        x_next["C4AF"] = x["C4AF"] + x_next["dC4AF"]

        x_next["SiO2"] = max(0.0, x["SiO2"] - (x_next["dC2S"] * 0.3488))
        x_next["Al2O3"] = max(0.0, x["Al2O3"] - (x_next["dC3A"] * 0.3773) - (x_next["dC4AF"] * 0.2098))
        x_next["Fe2O3"] = max(0.0, x["Fe2O3"] - (x_next["dC4AF"] * 0.3286))
        
        CaO_consumed = (x_next["dC2S"] * 0.6512) + (x_next["dC3S"] * 0.2456) + (x_next["dC3A"] * 0.6227) + (x_next["dC4AF"] * 0.4616)
        x_next["CaO"] = max(0.0, x["CaO"] + CaO_generated - CaO_consumed)

        # Kullanıcı isteği üzerine: Tam kütle kapanışı için CO2 hesabını kalan kütle üzerinden yapıyoruz.
        x_next["CO2"] = 80.0 - (x_next["CaCO3"] + x_next["CaO"])

        # ---------------------------------------------------------
        # 4. Process Indices
        # ---------------------------------------------------------
        x_next["LSF"] = x_next["CaO"] / (2.8 * x_next["SiO2"] + 1.2 * x_next["Al2O3"] + 0.65 * x_next["Fe2O3"] + 1e-9)
        x_next["Mass_Balance_Error"] = abs((x_next["CaO"] - x["CaO"]) - (CaO_generated - CaO_consumed))

        # -------------------------
        # Q_acc (kJ/kg)
        # -------------------------
        x_next["Q_acc"] = x_next["Clinker_output"] * 150
        
        # -------------------------
        # Q_loss
        # -------------------------
        x_next["Q_loss"] = (
            0.0016
            * 5.67e-8
            * 0.8
            * 110
            * (
                x_next["Tg_burning"]**4
                - 25**4
            )
        )
        
        # -------------------------
        # Q_reaction
        # -------------------------
        x_next["Q_reaction"] = (
            x_next["Feed_rate"]
            * (1 - x_next["Clinker_yield"])
            * 3.2
        )
        
        # -------------------------
        # Q_out (kJ/kg)
        # -------------------------
        x_next["Q_out"] = (
            x_next["Q_acc"]
            + x_next["Q_loss"]
            + x_next["Q_reaction"]
        )

        # -------------------------
        # Q_gas
        # -------------------------
        x_next["Q_gas"] = (
            x_next["Fuel_rate"]
            * 1.1
            * (x_next["Tg_burning"] - 25)
        )

        # -------------------------
        # Normalized_Energy_Index
        # -------------------------
        clinker_safe = max(x_next["Clinker_output"], 1e-6)

        x_next["Normalized_Energy_Index"] = (
            x_next["Q_in"]
            / clinker_safe
        )
        # -------------------------
        # Global_Energy_Closure
        # -------------------------
        x_next["Global_Energy_Closure"] = (
            x_next["Q_in"]
            - x_next["Q_out"]
            + x_next["Q_reaction"]
            - x_next["Q_acc"]
        )
        # -------------------------
        # Energy_error (%)
        # -------------------------
        q_in_safe = max(x_next["Q_in"], 1e-6)

        x_next["Energy_error"] = (
            x_next["Global_Energy_Closure"]
            / q_in_safe
        ) * 100 
        
        # -------------------------
        # SCALE + CONSTRAINT CHECK
        # -------------------------
        if t == 0:
            x_next["SCALE"] = 1.0                                                                                                          
        return x_next

def run_simulation():
    # --- SİMÜLASYON BAŞLATMA VE KONFİGÜRASYON ---
    executor = StepExecutor() 
    sim_time = 0.0
    
    reporting_dt = 1/6  # 10 dakika
    STEPS_PER_REPORT = int(reporting_dt / dt) 
    
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

    # ==============================
    # INITIAL STATE ALLOCATION
    # ==============================
    x_current = KilnState()
        
    x_current["Fuel_rate"] = 2.5
    x_current["O2"] = 6.0
    x_current["CO_ppm"] = 900.0
    
    # -------------------------
    # GAS PHASE INITIALIZATION
    # -------------------------
    
    x_current["Tg_preheater"] = 400.0
    x_current["Tg_calcination"] = 848.374628
    x_current["Tg_burning"] = 1245.088639
    x_current["Tg_Cooling"] = 250.0
    
    
    x_current["Air_flow"] = 45100
    x_current["Cooling_air_flow"] = 172440.0
    x_current["ID_fan_speed"] = 900.0
    
    # -------------------------
    # SOLID PHASE INITIALIZATION
    # -------------------------
    x_current["Feed_rate"] = 71.00
    x_current["Kiln_solid_out"] = 0.1
    x_current["Material_acc"] = 0.0
    
    x_current["Ts_preheater"] = 100.0
    x_current["Ts_calcination"] = 802.022
    x_current["Ts_burning"] = 1060.038
    x_current["Ts_Cooling"] = 1450.0
    
    # -------------------------
    # CHEMISTRY INITIALIZATION
    # -------------------------
    x_current["CaCO3"] = 80.0
    x_current["CaO"] = 1e-6
    x_current["CO2"] = 1e-6
    
    x_current["SiO2"] = 13.0
    x_current["Al2O3"] = 4.0
    x_current["Fe2O3"] = 3.0
    
    x_current["C2S"] = 1e-6
    x_current["C3S"] = 1e-6
    x_current["C3A"] = 1e-6
    x_current["C4AF"] = 1e-6
    
    # -------------------------
    # CONTROL INPUTS
    # -------------------------
    x_current["kiln_rpm"] = 1.0
    x_current["Residence"] = KilnPhysicsEngine.get_residence_time(1.0)
    x_current["Petcoke"] = 0.03
    x_current["Alternative_Fuel"] = 0.07
    x_current["Lignite_Coal"] = 0.90
    
    # 1. Fiziksel stok ve akış
    x_current["Material_acc"] = 15.0
    x_current["Clinker_output"] = 11.13
    x_current["Kiln_solid_out"] = 12.50
    
    # 2. Basınç Profili (Pa)
    x_current["P_preheater"] = 12.0
    x_current["P_calcination"] = 5.5
    x_current["P_burning"] = 1.2
    
    # 3. Damper kontrol
    x_current["Damper_position"] = 33.0
    
    # ---------------------------------------------------------
    # ENERGY BALANCE INITIALIZATION (STATIONARY STATE)
    # ---------------------------------------------------------
    
    # Enerji akışları (Q) [MW veya kW cinsinden, simülasyon birimine göre]
    x_current["Q_in"] = 35575.0      # Giriş yakıt enerjisi
    x_current["Q_out"] = 24285.0     # Çıkış (baca + klinker ile giden)
    x_current["Q_acc"] = 1669.0      # Termal birikim
    x_current["Q_loss"] = 22600.0     # Gövde ısı kaybı
    x_current["Q_reaction"] = 15.175 # Kalsinasyon reaksiyon ısısı
    
    # Enerji taşıyıcıları
    x_current["Q_gas"] = 3524.0     # Gaz fazındaki ısı
    x_current["Q_clinker"] = 15000.0 # Klinker fazındaki ısı
    
    # Performans Göstergeleri
    x_current["Clinker_yield"] = 0.89
    x_current["dTg_burning"] = 0.0
    
    # Enerji İndeksleri
    x_current["Normalized_Energy_Index"] = 1.0
    x_current["Global_Energy_Closure"] = 1.0
    x_current["Energy_error"] = 0.0  # Başlangıçta hata 0 olmalı
    
    x_current.Residence = KilnPhysicsEngine.get_residence_time(1.0)

    # ==============================
    # INPUT LAYER (CONTROL SYSTEM)
    # ==============================
    def input_layer(t: float, regime: str, initial_state: KilnState = None) -> dict[str, float]:
        """
        Zamana ve rejime bağlı kontrol girdilerini hesaplar.
        t=0 anında başlangıç değerlerini korumak için korumalıdır.
        """
        # BAŞLANGIÇ KORUMASI: t=0 anında manuel değerleri ezme
        if t < 1e-4 and initial_state is not None:
            return {
                "Petcoke": initial_state.Petcoke,
                "Alternative_Fuel": initial_state.Alternative_Fuel,
                "Feed_rate": initial_state.Feed_rate
            }

        D = 0.03 + (0.1 - 0.03) * (1 - np.exp(-t / 80.0))
        E = 0.07 + (0.14 - 0.07) * (1 - np.exp(-t / 100.0))
    
        base_feed = (
            132.0 if t >= 72.0 else
            72.0 + 60.0 * ((1.0 / (1.0 + np.exp(-0.065 * (t - 36.0)))) - 0.09) / 0.82
        )
    
        return {
            "Petcoke": D,
            "Alternative_Fuel": E,
            "Feed_rate": base_feed * REGIME_FEED_MULT.get(regime, 1.0)
        }
    
    # ==============================
    # MAIN SIMULATION LOOP
    # ==============================
    for t_idx in range(N): 
        current_regime = df.at[t_idx, "Regime"]
        sim_time = df.at[t_idx, "t"]
        
        # 1) Girdileri al (Başlangıç durumunu referans gönderiyoruz)
        inputs = input_layer(sim_time, current_regime, initial_state=x_current)
        inputs["regime"] = current_regime
        
        # 2) Fiziksel adımlar (StepExecutor içinde fiziksel güncellemeler gerçekleşiyor)
        for sub in range(STEPS_PER_REPORT):
            step_time = sim_time + (dt * (sub + 1))
            x_current = executor.perform_step(x_current, step_time, inputs=inputs)
        
        # 3) df'i satır satır güncelle
        state_dict = asdict(x_current)
        for col, val in state_dict.items():
            if col in df.columns:
                df.at[t_idx, col] = val
        
    df.to_csv("engine.csv", index=False)

if __name__ == '__main__':
    run_simulation()