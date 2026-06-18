import pandas as pd
import numpy as np


columns = [
    "t",
    "Regime", "Lignite_Coal", "Petcoke", "Alternative_Fuel",
    "Feed_rate", "Kiln_solid_out", "Material_acc", "Clinker_output",
    "Air_flow", "Cooling_air_flow", "ID_fan_speed", "Fuel_rate",
    "Tg_preheater", "Ts_preheater", "Tg_calcination", "Ts_calcination",
    "Tg_burning", "Ts_burning", "Tg_Cooling", "Ts_Cooling",
    "O2", "CO_ppm",
    "P_preheater", "P_calcination", "P_burning",
    "Damper_position",
    "CaCO3", "CaO", "CO2", "SiO2", "Al2O3", "Fe2O3",
    "LSF",
    "C3A", "C4AF", "C2S", "C3S",
    "dC2S", "dC3S", "dC3A", "dC4AF", "dCaO_calcination",
    "Mass_Balance_Error",
    "kiln_rpm", "Residence",
    "Q_in", "Q_out", "Q_acc", "Q_loss", "Q_reaction", "Q_gas", "Q_clinker",
    "Clinker_yield",
    "dTg_burning", "dFuel_rate",
    "Normalized_Energy_Index", "Global_Energy_Closure", "Energy_error"
]

N = 434
df = pd.DataFrame(0.0, index=np.arange(N), columns=columns)

dt = 0.05
df["t"] = np.arange(N) * dt

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

df["Regime"] = regime_list[:N]

# =============================================================
# STABLE RESIDENCE TIME (FIXED + FILLING COUPLING)
# =============================================================
def get_residence_time(kiln_rpm, filling_rate=0.10):
    L = 60.0
    D = 4.2
    slope = 0.03

    rpm = max(0.1, kiln_rpm)

    # daha fiziksel RPM influence (log saturation değil, mild saturation)
    rpm_eff = rpm / (0.26 + rpm)

    # filling effect (higher filling -> slower axial movement)
    filling = max(0.01, filling_rate)  # safety clamp

    filling_factor = (0.08 / filling) ** 0.3

    # corrected axial velocity scale (calibrated to industrial range)
    v_axial = (
        5.61 * D * rpm_eff * (1.5 + 44.8 * slope)
    ) * filling_factor

    residence = (L / (v_axial + 1e-6)) * 60.0

    return residence

# =============================================================
# STEP FUNCTION (OPEN-LOOP PLANT MODEL)
# =============================================================
def step(x, t, regime):

    x_next = x.copy()

    Tg_calcination = x.get("Tg_calcination", 850.0)
    Ts_calcination = x.get("Ts_calcination", 800.0)

    m_air = x.get("Air_flow", 45000.0) * 0.001293
    m_solid = x.get("Feed_rate", 40.0)

    x_next["Cooling_air_flow"] = x.get("Feed_rate", 40.0) * 4000.0

    # Fuel composition
    x_next["Lignite_Coal"] = max(
        0.0,
        1.0 - x.get("Petcoke", 0.0) - x.get("Alternative_Fuel", 0.0)
    )

    # ---------------------------------------------------------
    # INPUT TRAJECTORIES (OPEN-LOOP DRIVERS - NO CONTROL LOOP)
    # ---------------------------------------------------------
    feed = x.get("Feed_rate", 40.0)
    x_next["Feed_rate"] = feed + (120.0 - feed) * 0.0005 * dt
    x_next["Air_flow"] = 45000.0 + (95000.0 - 45000.0) * (1.0 - np.exp(-t / 120.0))
    x_next["Cooler_air_flow"] = 8000.0 + (83000.0 - 8000.0) * (1.0 - np.exp(-t / 140.0))
    x_next["ID_fan_speed"] = 900.0 + (2550.0 - 900.0) * (1.0 - np.exp(-t / 110.0))
    x_next["Damper_position"] = 33.0 + (85.0 - 33.0) * np.exp(-t / 25.0)

    # Saf Eksponansiyel Fuel Rampa:

    x_next["Fuel_rate"] = 2.5 + (6.8 - 2.5) * (1.0 - np.exp(-t / 35.0))
    f_rate = max(0.1, x_next["Fuel_rate"])

    # RPM
    rpm_current = x.get("kiln_rpm", 1.0)
    rpm_setpoint = x.get("rpm_setpoint", 2.4)
    alpha = 0.005

    rpm_next = rpm_current + alpha * (rpm_setpoint - rpm_current)
    x_next["kiln_rpm"] = max(0.1, rpm_next)

    # Residence
    res_min = get_residence_time(x_next["kiln_rpm"])
    x_next["Residence"] = res_min

    # Mass balance
    mat_acc = x.get("Material_acc", 15.0)

    kiln_out = mat_acc / (res_min / 60.0 + 1e-6)
    x_next["Kiln_solid_out"] = kiln_out

    x_next["Material_acc"] = mat_acc + (
        x_next["Feed_rate"] / 1.55 - kiln_out
    ) * (dt / 60.0)
    
    # Thermal
    Tg = np.clip(x.get("Tg_burning", 1200.0), 400.0, 1600.0)
    Ts = np.clip(x.get("Ts_burning", 1050.0), 400.0, 1600.0)

    m_air = (x_next["Air_flow"] * 1.293) / 3600.0
    m_sol = (x_next["Feed_rate"] * 1000.0) / 3600.0

    # Fiziksel İyileştirme: Sıcaklıktan bağımsız kütle çıkışı ve kalite ayrımı
    conversion_factor = 0.89
    x_next["Clinker_output"] = kiln_out * conversion_factor
    
    temp_eff = np.clip((Ts - 800.0) / 400.0, 0.1, 1.0)
    x_next["Clinker_quality"] = temp_eff

    Q_clink = x_next["Clinker_output"] * 1000.0 * 1.15 * (Ts - 100.0) / 3600.0
    Q_g2s = (220.0 if Tg > 1000.0 else 50.0) * x.get("A_s", 25.0) * (Tg - Ts)
    Q_loss = 50.0 * 13.85 * (Tg - 30.0)
    Q_g_adv = m_air * 1250.0 * (Tg - 400.0)
    Q_s_adv = m_sol * 1150.0 * (Ts - 900.0)
    Q_exo = min(400000.0, m_sol * 350000.0) if Ts > 1250.0 else 0.0

    dTg = (x.get("Q_in", 0.0) * 1000.0 - Q_g2s - Q_loss - Q_g_adv) / (max(10.0, m_air) * 5.0 * 1250.0)
    dTs = (Q_g2s + Q_exo - Q_s_adv - Q_clink) / (max(1.0, m_sol) * 120.0 * 1150.0)

    x_next["Tg_burning"] = Tg + dTg * dt
    x_next["Ts_burning"] = Ts + dTs * dt

    # O2
    air_fuel_ratio = x_next["Air_flow"] / (f_rate + 1e-6)

    o2_target = 2.5 + 4.5 * np.exp(-1.2 / (air_fuel_ratio / 20000.0 + 1e-3))

    x_next["O2"] = np.clip(
        x.get("O2", 3.5) + 0.15 * (o2_target - x.get("O2", 3.5)),
        0.0,
        21.0
    )

    x.update(x_next)
    x["t"] = t + dt

    
    
    # -------------------------
    # Q_in (kJ/kg)
    # -------------------------
    x_next["Q_in"] = (
        (x_next["Lignite_Coal"] * 15000
         + x_next["Petcoke"] * 30000
         + x_next["Alternative_Fuel"] * 18000)
        * x_next["Fuel_rate"]
        * np.exp(-((x_next["O2"] - 3.5) ** 2) / 25)
    )
    
    # -------------------------
    # dFuel_rate
    # -------------------------
    if t == 0:
        x_next["dFuel_rate"] = 1e-6
    else:

        x_next["dFuel_rate"] = (
            x_next["Fuel_rate"]
            - x["Fuel_rate"]
        ) / dt
            

    #===================================== Operational Zones =====================================
    
    # -------------------------
    # P_preheater (Pa)
    # -------------------------
    if t == 0:
        x_next["P_preheater"] = -120.0
    else:
        x_next["P_preheater"] = (
            -269.0
            + (-120.0 + 269.0) * np.exp(-t / 15)
        )
        
    # -------------------------------------------------------------
    # Sabitlerin Tanımlanması (Zon Başında)
    # -------------------------------------------------------------
    Cp_g = 1150.0            
    Cp_s = 1000.0            
    T_ambient = 25.0         
    
    C_gas_pre = 200000.0
    C_solid_pre = 300000.0

    m_gas = x["Air_flow"] * 0.001293
    m_solid = x["Feed_rate"]
    
    UA_pre = 570000 # (Genel Isı Transfer Katsayısı)
    
    # -------------------------------------------------------------
    # Tg_preheater (°C) - Gaz Fazı
    # -------------------------------------------------------------
    a_gas_pre = (m_gas * Cp_g) + UA_pre
    b_gas_pre = (m_gas * Cp_g * x["Tg_burning"]) + (UA_pre * x["Ts_preheater"])
    
    Tg_next_pre = (C_gas_pre * x["Tg_preheater"] + dt * b_gas_pre) / (C_gas_pre + dt * a_gas_pre)
    x_next["Tg_preheater"] = Tg_next_pre

    # -------------------------------------------------------------
    # Ts_preheater (°C) - Katı Faz
    # -------------------------------------------------------------
    a_s_pre = (m_solid * Cp_s) + UA_pre
    b_s_pre = (m_solid * Cp_s * T_ambient) + (UA_pre * Tg_next_pre)
    
    Ts_next_pre = (C_solid_pre * x["Ts_preheater"] + dt * b_s_pre) / (C_solid_pre + dt * a_s_pre)
    x_next["Ts_preheater"] = Ts_next_pre
        

    # -------------------------------------------------------------
    # Ts_calcination (°C) - Katı Faz (Enerji Alıcı)
    # -------------------------------------------------------------
    Cp_s = 1050.0            
    m_solid = x["Feed_rate"] 
    h_gs = 165.0             
    A_s = 840.0             
    Ts_calcination = x["Ts_calcination"]
    
    # Reaksiyon yükü: Sigmoid throttle + Dinamik oran (1000C üstünde tam kapasite)
    delta_H_rxn = 1700.0   
    base_rate = 0.05
    max_rate = 0.15
    dynamic_rate = base_rate + (max_rate - base_rate) * (1.0 / (1.0 + np.exp(-0.02 * (Ts_calcination - 1000.0))))
    reaction_throttle = 1.0 / (1.0 + np.exp(-0.08 * (Ts_calcination - 850.0)))
    
    Q_calcination_load = m_solid * 1000.0 * dynamic_rate * reaction_throttle * delta_H_rxn
    
    C_solid = (m_solid * 1000.0) * Cp_s / 3600.0
    a_s = h_gs * A_s
    b_s = (h_gs * A_s * Tg_calcination) - Q_calcination_load
    
    Ts_next = (C_solid * Ts_calcination + dt * b_s) / (C_solid + dt * a_s)
    x_next["Ts_calcination"] = Ts_next

    # -------------------------------------------------------------
    # Tg_calcination (°C) - Gaz Fazı (Enerji Kaynağı)
    # -------------------------------------------------------------
    Cp_g = 1150.0            
    A_c = 13.85             
    h_c = 1.5                
    m_air = x["Air_flow"] * 0.001270  
    
    Tg_preheater = x.get("Tg_preheater", 400.0) 
    T_tertiary_nominal = 950.0  
    unit_conversion = 1000.0  
    
    chi_gas = 0.75           
    Q_in_effective = x["Q_in"] * unit_conversion * chi_gas
    
    rho_V_g_effective = 1000.0 
    C_gas = rho_V_g_effective * Cp_g 

    Q_gs_factor = h_gs * A_s 
    a_gas = (m_air * Cp_g) + (h_c * A_c) + Q_gs_factor
    
    # Gaz dengesi: Ts_next ile coupled (Bağlı) denge
    b_gas = (m_air * Cp_g * T_tertiary_nominal) + Q_in_effective + \
            (h_c * A_c * Tg_preheater) + (Q_gs_factor * Ts_next * 1.02)
    
    Tg_next = (C_gas * Tg_calcination + dt * b_gas) / (C_gas + dt * a_gas)
    x_next["Tg_calcination"] = Tg_next
    
                
# -------------------------------------------------------------
    # Tg_burning & Ts_burning (°C) - Birinci İlkeler Modeli
    # -------------------------------------------------------------
    
    # Sabitler ve Boyutsal Çevrimler
    Cp_g = 1250.0  # J/(kg·K)
    Cp_s = 1150.0  # J/(kg·K)
    A_c = 13.85    # Kayıp yüzey alanı m²
    h_c = 0.05     # Çevreye kayıp katsayısı kW/(m²·K)
    
    # Alan tanımlaması (Eğer fırın geometrisinden gelmiyorsa güvenli default)
    A_s = x.get("A_s", 25.0) 
    
    # Akışların Saatlikten Saniyelik (kg/s) Birimine Çevrilmesi (Fiziksel Tutarlılık)
    # x["Air_flow"] Nm³/h -> kg/s çevrimi
    m_air_s = (x["Air_flow"] * 1.293) / 3600.0  
    # x["Feed_rate"] ton/h -> kg/s çevrimi
    m_solid_s = (x["Feed_rate"] * 1000.0) / 3600.0 
    
    # Enerji Girdisi (Q_in kW cinsinden kabul edilmiştir, Watt'a çevrilir)
    Q_burn_in_W = x.get("Q_in", 0.0) * 1000.0 
    
    # FIRIN İÇİ STATİK KÜTLE HESABI (Sıfıra bölmeyi önleyen eylemsizlik)
    # Fırının o bölgesinde her an çakılı duran malzeme ve gaz kütlesi:
    M_gas_zone = 120.0    # kg (Fırın içi anlık gaz kütlesi)
    M_solid_zone = 4500.0 # kg (Fırın içi anlık malzeme kütlesi - Hold-up kütlesi)
    
    C_gas_total = M_gas_zone * Cp_g      # J/K (Gazın gerçek termal kapasitesi)
    C_solid_total = M_solid_zone * Cp_s  # J/K (Katının gerçek termal kapasitesi)
    
    # Mevcut Sıcaklık Durumları
    Tg_curr = x["Tg_burning"]
    Ts_curr = x["Ts_burning"]
    
    # Dinamik Isı Transfer Katsayısı
    h_gs_burn = 220.0 if Tg_curr > 1000.0 else 50.0 # W/(m²·K)
    
    # -------------------------------------------------------------
    # Gelişmiş Enerji Dengesi Denklemleri (Karşılıklı Etkileşimli)
    # -------------------------------------------------------------
    
    # A. Gaz Bölgesi Akıları (Watt)
    Q_gas_to_solid = h_gs_burn * A_s * (Tg_curr - Ts_curr)
    Q_gas_loss_ambient = (h_c * 1000.0) * A_c * (Tg_curr - 30.0) # h_c kW'dan W'a çekildi
    Q_gas_advection = m_air_s * Cp_g * (Tg_curr - 400.0) # Giriş havası 400°C kabul edildi
    
    # Gaz Türevi (Net Isı / Termal Kapasite)
    dTg_dt = (Q_burn_in_W - Q_gas_to_solid - Q_gas_loss_ambient - Q_gas_advection) / C_gas_total
    
    # B. Katı Bölgesi Akıları (Watt)
    # Klinker ekzotermik reaksiyon ısısı (Sadece kalsinasyon üstü sıcaklıklarda tetiklenir)
    Q_exo_W = 0.0
    if Ts_curr > 1200.0:
        # Reaksiyon hızı sıcaklıkla orantılı (Maksimum 350 kW ekzotermik katkı)
        Q_exo_W = min(350000.0, m_solid_s * 500000.0 * (1.0 + (Ts_curr - 1200.0)/250.0))
        
    # Malzeme akışının taşıdığı duyu sıcaklığı kaybı (Advection)
    Q_solid_advection = m_solid_s * Cp_s * (Ts_curr - 900.0) # Kalsinasyondan 900°C'de giriyor
    
    # Katı Türevi
    dTs_dt = (Q_gas_to_solid + Q_exo_W - Q_solid_advection) / C_solid_total
    
    # -------------------------------------------------------------
    # Sayısal İntegrasyon (Yarı-%100 Kararlı Örtük Yaklaşım sönümlemesi)
    # -------------------------------------------------------------
    Tg_next = Tg_curr + dTg_dt * dt
    Ts_next = Ts_curr + dTs_dt * dt
    
    # İkinci Termodinamik Kanun Koruması: Isı soğuktan sıcağa akamaz.
    # Çözücü adımı kaçırsa bile katı gazı geçemez.
    if Ts_next > Tg_next:
        Ts_next = Tg_next - 5.0
        
    x_next["Tg_burning"] = Tg_next
    x_next["Ts_burning"] = Ts_next
    
    

    # -------------------------------------------------------------
    # Cooling Zone - Epsilon-NTU Tabanlı Verimlilik Modeli
    # -------------------------------------------------------------
    
    # 1. Fiziksel Sabitler
    Cp_g = 1150.0            
    Cp_s = 1150.0            
    T_air_in = 25.0          
    C_gas_cool = 200000.0    
    C_solid_cool = 220.000 

    # 2. Debiler ve Rezidans Faktörü
    m_gas = x.get("Cooling_air_flow", 10000.0) * 0.001293 * 0.5
    m_solid = x.get("Feed_rate", 100.0)
    
    # Residence süresi arttıkça soğutucu verimi (epsilon) artar
    res_val = x_next.get("Residence", 1.0)
    epsilon = min(0.75 * (res_val / 30.0), 0.9) # Max %90 verim
    
    # 3. Enerji Transferi ve Kayıplar
    # Maksimum potansiyel enerji transferi (Q_max)
    delta_T_max = x.get("Ts_burning", 1400.0) - T_air_in
    Q_max = m_solid * Cp_s * delta_T_max
    
    # Gerçek transfer edilen enerji (Epsilon * Q_max)
    Q_transfer = epsilon * Q_max
    
    # Çevreye giden ısı kaybı (Radyasyon/Konveksiyon)
    Q_loss_cooler = 150000.0 * (x.get("Ts_Cooling", 400.0) - 25.0)
    
    # -------------------------------------------------------------
    # Dinamik Denklemler
    # -------------------------------------------------------------
    
    # Gaz (Tg) denklemi:
    # Gazın enerji dengesi: (Masaüstü kütlesi * Cp * T_in) + Gelen Isı
    # a_g_cool'u sadece gazın kendi akış kapasitesine indirgedik (sıkışma bitti)
    a_g_cool = (m_gas * Cp_g)
    b_g_cool = (m_gas * Cp_g * T_air_in) + Q_transfer
    
    Tg_next_cool = (C_gas_cool * x.get("Tg_Cooling", 25.0) + dt * b_g_cool) / (C_gas_cool + dt * a_g_cool)
    x_next["Tg_Cooling"] = Tg_next_cool
    
    # Katı (Ts) denklemi:
    # Katı, giriş sıcaklığından gelir, transfer edilen ısıyı ve kayıpları düşeriz
    a_s_cool = (m_solid * Cp_s)
    
    # Normalize edilmiş ısı kaybı
    heat_loss_term = Q_loss_cooler / (m_solid * Cp_s + 1e-6)
    b_s_cool = (m_solid * Cp_s * x.get("Ts_burning", 1400.0)) - Q_transfer - heat_loss_term
    
    Ts_next_cool = (C_solid_cool * x.get("Ts_Cooling", 400.0) + dt * b_s_cool) / (C_solid_cool + dt * a_s_cool)
    x_next["Ts_Cooling"] = Ts_next_cool
                       
                
    # -------------------------
    # dTg_burning
    # -------------------------
    if t == 0:
        x_next["dTg_burning"] = 1e-6
    else:

        x_next["dTg_burning"] = (
            x_next["Tg_burning"]
            - x["Tg_burning"]
        ) / dt         
            
        
    # -------------------------
    # CO ppm
    # -------------------------
    if t == 0:
        x_next["CO_ppm"] = 900.0
    else:
        oxygen_deficit = max(0.0, 6.0 - x_next["O2"])

        T_eff = x_next["Tg_burning"]

        temp_factor = np.exp(-T_eff / 1200)

        fuel_factor = x_next["Fuel_rate"] / 6.0

        target_CO = 20 + 800 * oxygen_deficit * fuel_factor * temp_factor

        x_next["CO_ppm"] = (
            x["CO_ppm"]
            + 0.25 * (target_CO - x["CO_ppm"])
        )
            

            
    #===================================== Reactions =====================================
    
    # -------------------------
    # P_calcination (Pa)
    # -------------------------
    if t == 0:
        x_next["P_calcination"] = -180.0
    else:
        x_next["P_calcination"] = (
            -406.0
            + (-180.0 + 406.0) * np.exp(-t / 20)
        )

    # ---------------------------------------------------------
    # 1. CaCO3 Kalsinasyonu (Kinetik model doğru)
    # ---------------------------------------------------------
    if t == 0:
        x_next["CaCO3"] = 80.0
    else:
        T_kelvin = x["Ts_calcination"] + 273.15
        k = 1e7 * np.exp(-160000 / (8.314 * T_kelvin)) if T_kelvin > 873.15 else 0.0
        x_next["CaCO3"] = float(round(x["CaCO3"] * np.exp(-k * dt), 6))

    # ---------------------------------------------------------
    # 2. CaO (Kütle Korunumu)
    # ---------------------------------------------------------
    # Oluşan CaO, parçalanan CaCO3 miktarı ile doğrudan ilişkilidir.
    dCaCO3 = x["CaCO3"] - x_next["CaCO3"]  # Parçalanan CaCO3 (Pozitif)
    x_next["CaO"] = x["CaO"] + (dCaCO3 * 0.5603) 

    # ---------------------------------------------------------
    # 3. CO2 (Kütle Korunumu)
    # ---------------------------------------------------------
    # CO2, CaCO3'ün parçalanmasıyla açığa çıkan gazdır.
    # CO2_gen = dCaCO3 * 0.4397 (Stoikiyometrik oran)
    x_next["CO2"] = x["CO2"] + (dCaCO3 * 0.4397)
        
    # -------------------------
    # dCaO_calcination
    # -------------------------
    if t == 0:
        x_next["dCaO_calcination"] = 1e-6
    else:
        # AB = CaCO3 feed
        ab = x_next["CaCO3"]
        T_kelvin_calc = x_next["Ts_calcination"] + 273.15

        if T_kelvin_calc < 873.15:
            arrhenius = 0.0
        else:
            arrhenius = np.exp(
                -190000 / (8.314 * T_kelvin_calc)
            )

        rate_term = 100000000 * arrhenius
        kinetic = 1 - np.exp(-rate_term * dt)

        limiting = min(
            ab,
            (80 - ab)
        )
    
        x_next["dCaO_calcination"] = max(0.0, limiting * kinetic)
    
        # -------------------------
    # SiO2
    # -------------------------
    if t == 0:
        x_next["SiO2"] = 13.0
    else:
        x_next["SiO2"] = max(
            0.0,
            x["SiO2"] - (x["C3A"] * 0.3488383838 + x["C4AF"] * 0.2631526466)
        )

    # -------------------------
    # Al2O3
    # -------------------------
    if t == 0:
        x_next["Al2O3"] = 4.0
    else:
        x_next["Al2O3"] = max(
            0.0,
            x["Al2O3"] - (x["C3A"] * 0.3773565314 + x["C4AF"] * 0.209816997)
        )

    # -------------------------
    # Fe2O3
    # -------------------------
    if t == 0:
        x_next["Fe2O3"] = 3.0
    else:
        x_next["Fe2O3"] = max(
            0.0,
            x["Fe2O3"] - (x["C4AF"] * 0.3286167885)
        )

    # -------------------------
    # LSF
    # -------------------------
    x_next["LSF"] = x_next["CaO"] / (
        2.8 * x_next["SiO2"] +
        1.2 * x_next["Al2O3"] +
        0.65 * x_next["Fe2O3"] + 1e-9
    )

    # -------------------------
    # C3A
    # -------------------------
    if t == 0:
        x_next["C3A"] = 1e-6
    else:
       x_next["C3A"] = max(
    0.0,
    x["C3A"] + x_next["dC3A"] * dt
)
    # -------------------------
    # C4AF
    # -------------------------
    if t == 0:
        x_next["C4AF"] = 1e-6
    else:
        x_next["C4AF"] = (
            x["C4AF"]
            + x_next["dC4AF"]
        )  
    # -------------------------
    # C2S
    # -------------------------
    if t == 0:
        x_next["C2S"] = 1e-6
    else:
        x_next["C2S"] = max(
            0.0,
            x["C2S"]
            + x_next["dC2S"]
            - 0.75 * x_next["dC3S"]
        )
    # -------------------------
    # C3S
    # -------------------------
    if t == 0:
        x_next["C3S"] = 1e-6
    else:
        x_next["C3S"] = (
            x["C3S"]
            + x_next["dC3S"]
        )
    # -------------------------
    # dC2S
    # -------------------------
    if t == 0:
        x_next["dC2S"] = 1e-6
    else:

        if x_next["SiO2"] <= 1e-6:
            x_next["dC2S"] = 1e-6
        else:

            arrhenius = np.exp(
                -170000 / (8.314 * (x_next["Ts_calcination"] + 273.15))
            )

            rate_term = 50000000 * arrhenius

            kinetic = 1 - np.exp(-rate_term *dt)

            limiting = min(
                x_next["SiO2"] - x_next["dC3S"],
                x_next["CaO"] / 2
            )

            x_next["dC2S"] = max(
                0.0,
                limiting * kinetic
            )

    # -------------------------
    # dC3S
    # -------------------------
    if t == 0:
        x_next["dC3S"] = 1e-6
    else:
        # AK = C2S
        if x_next["C2S"] <= 1e-6:
            x_next["dC3S"] = 1e-6
        else:
            T_kelvin_sinter = x_next["Ts_calcination"] + 273.15

            # FİZİKSEL EŞİK: Alit (C3S) oluşumu 1200°C altında sıvı faz olmadığından 
            # ve katı hal difüzyonu yetersiz kaldığından tamamen durur.
            if T_kelvin_sinter < 1473.15:
                arrhenius = 0.0
            else:
                arrhenius = np.exp(
                    -200000 / (8.314 * T_kelvin_sinter)
                )

            rate_term = 228000000 * arrhenius

            kinetic = 1 - np.exp(-rate_term * dt)

            limiting = min(
                x_next["C2S"],
                x_next["CaO"]
            )

            x_next["dC3S"] = max(
                0.0,
                limiting * kinetic
            )
    # -------------------------
    # dC3A
    # -------------------------
    if t == 0:
        x_next["dC3A"] = 1e-6
    else:

        # AF = Al2O3
        if x_next["Al2O3"] <= 1e-6:
            x_next["dC3A"] = 1e-6
        else:

            arrhenius = np.exp(
                -120000 / (8.314 * (x_next["Ts_calcination"] + 273.15))
            )

            rate_term = 100000 * arrhenius

            kinetic = 1 - np.exp(-rate_term *dt)

            limiting = min(
                x_next["Al2O3"],
                (x_next["CaO"] - 3 * x_next["dC3S"] - 2 * x_next["dC2S"]) / 3
            )

            x_next["dC3A"] = max(
                0.0,
                limiting * kinetic
            )
    # -------------------------
    # dC4AF
    # -------------------------
    if t == 0:
        x_next["dC4AF"] = 1e-6
    else:

        # AF = Al2O3, AG = Fe2O3
        if x_next["Al2O3"] <= 1e-6 or x_next["Fe2O3"] <= 1e-6:
            x_next["dC4AF"] = 1e-6
        else:

            arrhenius = np.exp(
                -150000 / (8.314 * (x_next["Ts_calcination"] + 273.15))
            )

            rate_term = 200000 * arrhenius

            kinetic = 1 - np.exp(-rate_term *dt)

            if x_next["CaO"] > 0:
                ca_limit = (x_next["CaO"]
                            - 3 * x_next["dC3S"]
                            - 2 * x_next["dC2S"]
                            - 3 * x_next["dC3A"]) / 4
            else:
                ca_limit = 100

            limiting = min(
                x_next["Al2O3"],
                x_next["Fe2O3"],
                ca_limit
            )

            x_next["dC4AF"] = max(
                0.0,
                limiting * kinetic
            )
            
    #===================================== Flows =====================================
        


    # -------------------------
    # Mass_Balance_Error
    # -------------------------
    if t == 0:
        x_next["Mass_Balance_Error"] = 1e-6
    else:

        Ca_balance = x_next["CaCO3"] - (
            x_next["CaO"]
            + 3*x_next["C3A"]
            + 4*x_next["C4AF"]
            + 2*x_next["C2S"]
            + 3*x_next["C3S"]
        )

        Si_balance = x_next["SiO2"] - (
            x_next["C2S"]
            + x_next["C3S"]
        )

        Al_balance = x_next["Al2O3"] - (
            x_next["C3A"]
            + x_next["C4AF"]
        )

        Fe_balance = x_next["Fe2O3"] - (
            x_next["C4AF"]
        )

        x_next["Mass_Balance_Error"] = (
            Ca_balance + Si_balance + Al_balance + Fe_balance
        )

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
    eps = 1e-9

    denom1 = abs(3*x_next["dC3S"] + 2*x_next["dC2S"] + 3*x_next["dC3A"] + 4*x_next["dC4AF"]) + eps
    denom2 = abs(x_next["SiO2"] + x_next["dC2S"]) + eps
    denom3 = abs(x_next["Al2O3"] + x_next["dC3A"]) + eps
    denom4 = abs(x_next["Fe2O3"]) + eps

    x_next["constraint_violation"] = (
        max(0, denom1 - x_next["CaO"])
        + max(0, denom2 - x_next["SiO2"])
        + max(0, denom3 - x_next["Al2O3"])
        + max(0, denom4 - x_next["Fe2O3"])
    )

    if t == 0:
        x_next["SCALE"] = 1.0                                                                                                          
    return x_next


#  ===================================== GLOBAL CONFIGURATION =====================================

reporting_dt = 1/6  # 10 dakika (saat cinsinden)
STEPS_PER_REPORT = int(reporting_dt / dt) 

REGIME_FEED_MULT = {
    "R1_HEATING_STABILIZATION": 0.6,
    "R2_EARLY_CALCINATION": 0.75,
    "R3_ACTIVE_CALCINATION": 0.85,
    "R4_TRANSITION_TO_CLINKERIZATION": 0.92,
    "R5_EARLY_CLINKERIZATION": 0.97,
    "R6_STEADY_CLINKERIZATION": 1.0,
    "R7_FUEL_SWITCH_TRANSIENT": 1.03,
    "R8_RESTABILIZATION": 0.9,
}

# ==============================
# INPUT LAYER (CONTROL SYSTEM)
# ==============================

def input_layer(t: float, regime: str) -> dict[str, float]:
    """
    Zamana ve rejime bağlı kontrol girdilerini hesaplar.
    """
    D = 0.03 + (0.1 - 0.03) * (1 - np.exp(-t / 80.0))
    E = 0.07 + (0.14 - 0.07) * (1 - np.exp(-t / 100.0))

    base_feed = (
        132.0 if t >= 72.0 else
        72.0 + 60.0 * ((1.0 / (1.0 + np.exp(-0.065 * (t - 36.0)))) - 0.09) / 0.82
    )

    return {
        "Petcoke": D,
        "Alternative_Fuel": E,
        "Feed_rate": base_feed * REGIME_FEED_MULT.get(regime, 1.0) # Fallback katsayısı eklendi
    }

# ==============================
# INITIAL STATE ALLOCATION
# ==============================

# Tüm değişkenleri başlangıçta 0.0 olarak atıyoruz (Regime hariç, onu aşağıda eziyoruz)
x_current = {col: 0.0 for col in columns if col != "Regime"}


# -------------------------------------
# Explicit Physical initial Conditions
# -------------------------------------

x_current["Fuel_rate"] = 2.5
x_current["O2"] = 6.0
x_current["CO_ppm"] = 900.0

# -------------------------
# GAS PHASE INITIALIZATION
# -------------------------

x_current["Tg_preheater"] = 400.0
x_current["Tg_calcination"] = 848.374628
x_current["Tg_burning"] = 1245.088639
x_current["Tg_Cooling"] = 1582.903


x_current["Air_flow"] = 45100
x_current["Cooling_air_flow"] = 172440.0
x_current["ID_fan_speed"] = 900.0

# -------------------------
# SOLID PHASE INITIALIZATION
# -------------------------
x_current["Feed_rate"] = 43.00
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
x_current["Residence"] = get_residence_time(1.0)
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
x_current["dFuel_rate"] = 0.0

# Enerji İndeksleri
x_current["Normalized_Energy_Index"] = 1.0
x_current["Global_Energy_Closure"] = 1.0
x_current["Energy_error"] = 0.0  # Başlangıçta hata 0 olmalı

# Pandas satır atamaları döngü içinde yavaştır. Çözüm sonuçlarını tutmak için liste kullanıyoruz.
results_list = []
sim_time = 0.0

# Başlangıç durumunu ilk satır olarak kaydediyoruz. (Regime verisi df'den okunur)
initial_record = x_current.copy()
initial_record["t"] = sim_time
initial_record["Regime"] = df.at[0, "Regime"]
results_list.append(initial_record)

# ==============================================================================
# 2. SİMÜLASYON DÖNGÜSÜNDEKİ BAĞLANTI (SUB-STEP PHYSICS)
# ==============================================================================

sim_time = 0.0

for t_idx in range(N - 1):

    # 1) Rejim ve Kontrol Girdilerinin Okunması
    current_regime = df.at[t_idx, "Regime"]
    inputs = input_layer(sim_time, current_regime)

    x_current["Petcoke"] = inputs["Petcoke"]
    x_current["Alternative_Fuel"] = inputs["Alternative_Fuel"]
    x_current["Feed_rate"] = inputs["Feed_rate"]
    x_current["Regime"] = current_regime 

    # 2) SUB-STEP PHYSICS: Fiziksel hesaplama adımları
    # Fonksiyon, her sub-step'te x_current'i günceller
    for sub in range(STEPS_PER_REPORT):
        # step_time artık sadece kayıt için değil, fonksiyon içindeki fiziksel mantık için referans
        step_time = sim_time + dt * (sub + 1)
        x_current = step(x_current, step_time, current_regime)

    # 3) ZAMAN İLERLETME
    sim_time += reporting_dt
    
    # 4) SAVE OUTPUT: Sadece tek bir kayıt işlemi (Double-counting hatası giderildi)
    record = x_current.copy()
    record["t"] = sim_time
    results_list.append(record)

# Döngü bittiğinde sonuçları DataFrame'e geçiriyoruz
df_results = pd.DataFrame(results_list)

# Eğer df_results index'leri 0'dan başlıyorsa ve df ile hizalamak istiyorsanız:
# df_results.index = df.index[:len(df_results)]
df.update(df_results)

# ==============================
# SAVE OUTPUT
# ==============================

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].round(5)

output_path = "kiln_simulation_output.csv"
df.to_csv(output_path, index=False)

print(f"Simülasyon başarıyla tamamlandı: {output_path}")