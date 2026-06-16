import pandas as pd
import numpy as np
import math

columns = [
    "t",
    "Regime","Lignite_Coal","Petcoke","Alternative_Fuel",
    "Feed_rate","Kiln_solid_out","Material_acc","Clinker_output",
    "Air_flow","Cooler_air_flow","ID_fan_speed","Fuel_rate",
    "Tg_preheater","Ts_preheater","Tg_calcination","Ts_calcination",
    "Tg_burning","Ts_burning","Tg_Cooling","Ts_Cooling",
    "O2","CO_ppm",
    "P_preheater","P_calcination","P_burning",
    "Damper_position",
    "CaCO3","CaO","CO2","SiO2","Al2O3","Fe2O3",
    "LSF",
    "C3A","C4AF","C2S","C3S",
    "dC2S","dC3S","dC3A","dC4AF","dCaO_calcination",
    "Mass_Balance_Error",
    "kiln_rpm","Filling_rate","Residence",
    "Q_in","Q_out","Q_acc","Q_loss","Q_reaction","Q_gas","Q_clinker",
    "Clinker_yield",
    "dTg_burning","dFuel_rate",
    "Normalized_Energy_Index","Global_Energy_Closure","Energy_error"
]

# 2. INIT DATA
N = 434

df = pd.DataFrame(0.0, index=np.arange(N), columns=columns)

dt = 0.05
df["t"] = np.arange(N) * dt

# 3. REGIME DEFINITION (OUTSIDE LOOP - FIXED ORDER)

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

regime_list = regime_list[:N]

df["Regime"] = regime_list

def step(x, t, regime):

    x_next = x.copy()
    
    #===================================== INPUTS =====================================
    
    # -------------------------
    # Fuel_composition
    # -------------------------

    x_next["Lignite_Coal"] = max(
        0.0,
        1.0 - x_next["Petcoke"] - x_next["Alternative_Fuel"]
    )
    
    # -------------------------
    # Fuel_rate (ton/h)
    # -------------------------

    F_min = 2.5
    F_max = 7.0

    if t == 0:
        x_next["Fuel_rate"] = F_min
    else:

        # 1) Smooth target ramp (operational reality)
        F_target = F_min + (F_max - F_min) * (
            1 / (1 + np.exp(-0.07 * (t - 14)))
        )

        # 2) Physical inertia (1st order lag)
        tau_F = 2.5  # system response time

        x_next["Fuel_rate"] = (
            x["Fuel_rate"]
            + (dt / tau_F) * (F_target - x["Fuel_rate"])
        )

        # 3) Safety bounds (physical + operational limits)
        x_next["Fuel_rate"] = np.clip(
            x_next["Fuel_rate"],
            F_min,
            F_max
        )
     
    # -------------------------
    # O2 (%)
    # -------------------------
    if t == 0:
        x_next["O2"] = 6.0
    else:
        air_fuel_ratio = x_next["Air_flow"] / (x_next["Fuel_rate"] + 1e-6)

        target_O2 = 2.5 + 4.5 * np.exp(-1.2 / (air_fuel_ratio + 1e-3))

        x_next["O2"] = (
            x["O2"]
            + 0.15 * (target_O2 - x["O2"])
        )
    
    
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
            
    # -------------------------
    # Feed_rate
    # -------------------------
    # Feed_rate is controlled externally (input_layer)
    x_next["Feed_rate"] = x["Feed_rate"]
        
    # -------------------------
    # Air_flow (EXP RAMP)
    # -------------------------
    x_next["Air_flow"] = 45000 + (95000 - 45000) * (1 - np.exp(-t / 120))
    
    # -------------------------
    # Cooler_air_flow (EXP RAMP)
    # -------------------------
    x_next["Cooler_air_flow"] = 8000 + (83000 - 8000) * (1 - np.exp(-t / 140))
    
    # -------------------------
    # ID_fan_speed (EXP RAMP)
    # -------------------------
    x_next["ID_fan_speed"] = 900 + (2550 - 900) * (1 - np.exp(-t / 110))
    
    # -------------------------
    # Damper_position (%)
    # -------------------------
    if t == 0:
        x_next["Damper_position"] = 85.0
    else:
        x_next["Damper_position"] = (
            33.0
            + (85.0 - 33.0) * np.exp(-dt / 25)
        )
    #===================================== Gas Phase =====================================
    
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
        
    # -------------------------
    # Tg_preheater (°C)
    # -------------------------
    if t == 0:
        x_next["Tg_preheater"] = 399.6
    else:
        w_up = 1 / (1 + np.exp(-0.03 * (x["Tg_preheater"] - 847)))
        w_down = 1 / (1 + np.exp(-0.03 * (x["Tg_preheater"] - 835)))
        w = (w_up + w_down) / 2

        T_pre = x["Tg_preheater"] + 0.0135 * (0.0131 * x["Air_flow"] - x["Tg_preheater"])
        T_calc = 847 + 0.0135 * ((0.0113 * x["Air_flow"] - 800) * 0.05)

        noise = np.random.normal(0, 1.8)

        x_next["Tg_preheater"] = (1 - w) * T_pre + w * T_calc + w * noise
        
    # -------------------------
    # Tg_calcination (°C) - Fiziksel Enerji Dengesi (dt = 0.05h)
    # -------------------------
    Cp_g = 1150.0          # kJ/tonK
    h_c = 35.0             
    A_c = 1.0              
    rho_V_g = 12000.0      
    eta_comb = 0.85 

    # 1. Kütlesel Debiler (ton/h)
    # Air_flow Nm³/h -> ton/h dönüşümü
    m_air = x["Air_flow"] * 0.001293  
    
    # 2. Durumlar
    Tg_preheater = x.get("Tg_preheater", 400.0)
    Tg_calcination = x["Tg_calcination"]
    
    # 3. Enerji Girdileri (kJ/h)
    # Q_in zaten toplam ısı girişi (kJ/h)
    Q_in = x["Q_in"] 
    
    # 4. Konvektif taşıma (kJ/h)
    Q_conv_in = m_air * Cp_g * (Tg_preheater - Tg_calcination) * eta_comb
    
    # 5. Toplam Net Enerji (kJ/h)
    Q_total = Q_conv_in + Q_in
    
    # 6. Isıl Eylemsizlik (kJ/K) - Sistemin termal kapasitesi
    # C_gas ne kadar büyükse sistem o kadar "ağır" tepki verir.
    C_gas = 1.2e6 
    
    # 7. İntegrasyon (Zaman adımıyla ölçeklendirilmiş enerji)
    # dt (saat) * Q_total (kJ/h) = Toplam Enerji (kJ)
    # Toplam Enerji (kJ) / C_gas (kJ/K) = Sıcaklık Değişimi (K)
    delta_Tg = (Q_total * dt) / C_gas
    
    # 8. Güncelleme
    Tg_next = Tg_calcination + delta_Tg + np.random.normal(0, 0.5)
    
    x_next["Tg_calcination"] = Tg_next
    
                
    # -------------------------
    # Tg_burning (°C)
    # -------------------------
    if t == 0:
        x_next["Tg_burning"] = x_next["Tg_calcination"]
    else:

        if t <= 36:

            w = 1 / (1 + np.exp(-0.25 * (t - 36)))

            R_new = x["Tg_burning"] + 0.00927 * (1605 - x["Tg_burning"])

            noise = np.random.normal(0, 1.5) * w

            x_next["Tg_burning"] = R_new + noise

        else:

            if t <= 218:

                noise = np.random.normal(0, 1.5)

                x_next["Tg_burning"] = x["Tg_burning"] + noise

            else:

                noise = np.random.normal(0, 1.5)

                ref = x["Tg_burning"]  # approx INDEX(R:R,218)

                x_next["Tg_burning"] = ref + noise   
                
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
    
    #-------------------------
    # Tg_Cooling
    #-------------------------
    if t == 0:
        x_next["Tg_Cooling"] = x["Tg_burning"]
    else:
        if t <= 36:
            w = 1 / (1 + np.exp(-0.25 * (t - 36)))
            T_new = x["Tg_Cooling"] + 0.0183 * (140 - x["Tg_Cooling"])
            noise = np.random.normal(0, 1.5) * w
            x_next["Tg_Cooling"] = T_new + noise
        else:
            x_next["Tg_Cooling"] = x["Tg_Cooling"] + np.random.normal(0, 1.5)
            
        
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
            
    #===================================== Solid Phase =====================================
    # -------------------------
    # Tg_preheater (°C)
    # -------------------------
    Tg = x["Tg_preheater"]
    Ts = x["Ts_preheater"]
            
    # 2. Fiziksel Parametreler
    h_gs = 96.0  # kJ/h*m2*C - Gaz-katı arası etkili ısı transfer katsayısı
    A_gs = 50.0  # m2 - Temas yüzey alanı (fırın boyutuna göre değişir)
            
    # 3. Enerji Transferi (kJ/h)
    Q_gs = h_gs * A_gs * (Tg - Ts)
            
    # 4. Katı Faz Enerji Dengesi (Isıl Eylemsizlik ile)
    # Cs: Katının toplam ısıl kapasitesi (kJ/K)
    Cs = 1.0e5 
            
    # dT/dt = Q / Cs
    dTs_dt = Q_gs / Cs
            
    # 5. Güncelleme
    Ts_next = Ts + (dTs_dt * dt)
            
    # Gaz fazı için (Enerji korumu gereği gazdan çıkan enerji)
    # Tg_next = Tg - (Q_gs * dt) / Cg
    x_next["Ts_preheater"] =Ts_next       
    
    # -------------------------
    # Ts_calcination (°C)
    # -------------------------
    # effective time constant (zone inertia)
    tau_s = 2.34  # tuneable but physically meaningful

    target = x["Tg_calcination"]

    x_next["Ts_calcination"] = (
        x["Ts_calcination"]
        + (dt / tau_s) * (target - x["Ts_calcination"])
    )
        
           
    # -------------------------
    # Ts_burning (°C)
    # -------------------------
    if t == 0:
        x_next["Ts_burning"] = x_next["Ts_calcination"]
    else:

        w = 1 / (1 + np.exp(-0.25 * (t - 36)))

        S_new = x["Ts_burning"] + 0.0221 * (1455 - x["Ts_burning"])

        noise = np.random.normal(0, 1.5) * w

        x_next["Ts_burning"] = S_new + noise
        
       
    # -------------------------
    # Ts_Cooling (°C)
    # -------------------------
    if t == 0:
        x_next["Ts_Cooling"] = x["Ts_Cooling"]
    else:

        w = 1 / (1 + np.exp(-0.25 * (t - 36)))

        # cooling relaxation toward ambient (~100°C)
        U_new = x["Ts_Cooling"] - 0.01295 * (x["Ts_Cooling"] - 100)

        noise = np.random.normal(0, 1.5) * w

        if t <= 36:
            x_next["Ts_Cooling"] = U_new + noise
        elif t <= 218:
            x_next["Ts_Cooling"] = x["Ts_Cooling"] + np.random.normal(0, 1.5)
        else:
            x_next["Ts_Cooling"] = x["Ts_Cooling"] + np.random.normal(0, 1.5)
            
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

    # -------------------------
    # CaCO3 (ton/h)
    # -------------------------
    if t == 0:
        x_next["CaCO3"] = 80.0
    else:

        k = (
            10000000
            * np.exp(-160000 / (8.314 * (x["Ts_calcination"] + 273.15)))
        )

        x_next["CaCO3"] = x["CaCO3"] * np.exp(-k *dt)
    # -------------------------
    # CaO (ton/h)
    # -------------------------
    if t == 0:
        x_next["CaO"] = 1e-6
    else:

        # previous CaCO3 (AB3 in Excel logic)
        CaCO3_prev = x["CaCO3"]

        # next CaCO3 (AB2 in Excel logic -> already computed in this step)
        CaCO3_now = x_next["CaCO3"]

        dCaCO3 = (CaCO3_now - CaCO3_prev)

        x_next["CaO"] = (
            x["CaO"]
            + (dCaCO3 * 0.560331)
            - (x["Fuel_rate"] * 2)
            - (x["Alternative_Fuel"] * 3)
        )
    # -------------------------
    # CO2 (ton/h)
    # -------------------------
    if t == 0:
        x_next["CO2"] = 1e-6
    else:
        x_next["CO2"] = (
            80.0
            - (x_next["CaCO3"] + x_next["CaO"])
        )
        
    # -------------------------
    # dCaO_calcination
    # -------------------------
    if t == 0:
        x_next["dCaO_calcination"] = 1e-6
    else:

        # AB = CaCO3 feed
        ab = x_next["CaCO3"]

        arrhenius = np.exp(
            -190000 / (8.314 * (x_next["Ts_calcination"] + 273.15))
        )

        rate_term = 100000000 * arrhenius

        kinetic = 1 - np.exp(-rate_term *dt)

        limiting = min(
            ab,
            (80 - ab)
        )

        x_next["dCaO_calcination"] = max(
            0.0,
            limiting * kinetic
        )
    
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

            arrhenius = np.exp(
                -200000 / (8.314 * (x_next["Ts_calcination"] + 273.15))
            )

            rate_term = 228000000 * arrhenius

            kinetic = 1 - np.exp(-rate_term *dt)

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
    # G: Kiln_solid_out
    # -------------------------
    if t == 0:
        x_next["Kiln_solid_out"] = 1e-6
    else:
        x_next["Kiln_solid_out"] = (
            0.15 * (x_next["Feed_rate"] - x_next["CO2"])
            + 0.85 * x["Kiln_solid_out"]
        )
        
    # -------------------------
    # Clinker_output
    # -------------------------
    if t == 0:
        x_next["Clinker_output"] = 1e-6
    else:
        x_next["Clinker_output"] = 0.9 * x_next["Kiln_solid_out"]
        
    # -------------------------
    # Clinker_yield
    # -------------------------
    feed_safe = max(x_next["Feed_rate"], 1e-6)

    x_next["Clinker_yield"] = (
        x_next["Clinker_output"]
        / feed_safe
    )    

    # -------------------------
    # H: Material_acc
    # -------------------------
    if t == 0:
        x_next["Material_acc"] = x_next["Feed_rate"]
    else:
        x_next["Material_acc"] = (
            x_next["Feed_rate"]
            - x_next["Kiln_solid_out"]
            - x_next["CO2"]
        )

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
    # kiln_rpm
    # -------------------------
    if t == 0:
        x_next["kiln_rpm"] = 0.1
    else:

        coeff = {
            "R1_HEATING_STABILIZATION": 0.55,
            "R2_EARLY_CALCINATION": 0.70,
            "R3_ACTIVE_CALCINATION": 0.80,
            "R4_TRANSITION_TO_CLINKERIZATION": 0.90,
            "R5_EARLY_CLINKERIZATION": 0.95,
            "R6_STEADY_CLINKERIZATION": 1.00,
            "R7_FUEL_SWITCH_TRANSIENT": 1.05,
            "R8_RESTABILIZATION": 0.85,
        }.get(regime, 1.0)

        coeff *= (1 - np.exp(-0.02 * t))

        x_next["kiln_rpm"] = (
            0.1
            + (
                x_next["Fuel_rate"]
                + (6 - x_next["Fuel_rate"])
                * (1 - np.exp(-0.00041 * t))
                * coeff
            )
            * 0.8
            * (1 - np.exp(-0.01 * t))
        )

    # -------------------------
    # Filling_rate
    # -------------------------
    x_next["Filling_rate"] = 0.1
    
    # -------------------------
    # Residence (min)
    # -------------------------

    if t == 0:
        x_next["Residence"] = 1e-6
    else:

        kiln_rpm = max(x_next["kiln_rpm"], 1e-6)
        feed = max(x_next["Feed_rate"], 1e-6)

        x_next["Residence"] = (
            (0.37 * 60)
            / (4.2 * kiln_rpm * math.tan(math.atan(3 / 100)))
            * (feed / 120) ** (-0.6)
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
    # Q_clinker
    # -------------------------
    x_next["Q_clinker"] = (
        x_next["Clinker_output"]
        * 1800
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
x_current["Tg_calcination"] = 846.531
x_current["Tg_burning"] = 1250.0
x_current["Tg_Cooling"] = 1590.0


x_current["Air_flow"] = 50000.0
x_current["Cooler_air_flow"] = 10000.0
x_current["ID_fan_speed"] = 900.0

# -------------------------
# SOLID PHASE INITIALIZATION
# -------------------------
x_current["Feed_rate"] = 41.50
x_current["Kiln_solid_out"] = 0.1
x_current["Material_acc"] = 0.0

x_current["Ts_preheater"] = 100.0
x_current["Ts_calcination"] = 750.0
x_current["Ts_burning"] = 1150.0
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
x_current["Petcoke"] = 0.03
x_current["Alternative_Fuel"] = 0.07
x_current["Lignite_Coal"] = 0.90

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

output_path = "kiln_simulation_output.csv"
df.to_csv(output_path, index=False)

print(f"Simülasyon başarıyla tamamlandı: {output_path}")