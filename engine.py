import pandas as pd
import numpy as np

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
    "Normalized_Energy_Index","Global_Energy_Closure","Energy_error",
    "SCALE"
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

# expand regime into time series
regime_list = []
for r, n in regime_series:
    regime_list.extend([r] * n)

# safety clamp (CRITICAL)
regime_list = regime_list[:N]

df["Regime"] = regime_list


def step(x, t, regime):

    x_next = x.copy()

    x_next["Lignite_Coal"] = max(
        0.0,
        1.0 - x_next["Petcoke"] - x_next["Alternative_Fuel"]
    )

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
    # I: Clinker_output
    # -------------------------
    if t == 0:
        x_next["Clinker_output"] = 1e-6
    else:
        x_next["Clinker_output"] = 0.9 * x_next["Kiln_solid_out"]

    # -------------------------
    # J: Air_flow (EXP RAMP)
    # -------------------------
    x_next["Air_flow"] = 45000 + (95000 - 45000) * (1 - np.exp(-t / 120))

    # -------------------------
    # K: Cooler_air_flow (EXP RAMP)
    # -------------------------
    x_next["Cooler_air_flow"] = 8000 + (83000 - 8000) * (1 - np.exp(-t / 140))

    # -------------------------
    # L: ID_fan_speed (EXP RAMP)
    # -------------------------
    x_next["ID_fan_speed"] = 900 + (2550 - 900) * (1 - np.exp(-t / 110))

    # -------------------------
    # M: Fuel_rate (ton/h)
    # -------------------------
    if t == 0:
        x_next["Fuel_rate"] = 2.5
    else:

        REGIME_FUEL_MULT = {
            "R1_HEATING_STABILIZATION": 0.6,
            "R2_EARLY_CALCINATION": 0.75,
            "R3_ACTIVE_CALCINATION": 0.85,
            "R4_TRANSITION_TO_CLINKERIZATION": 0.92,
            "R5_EARLY_CLINKERIZATION": 0.97,
            "R6_STEADY_CLINKERIZATION": 1.0,
            "R7_FUEL_SWITCH_TRANSIENT": 1.03,
            "R8_RESTABILIZATION": 0.9,
        }

        k = REGIME_FUEL_MULT[regime]

        x_next["Fuel_rate"] = (
            x["Fuel_rate"]
            + (6.0 - x["Fuel_rate"]) * (1 - np.exp(-0.00041 * t))
        ) * k

    # -------------------------
    # N: Tg_preheater (°C)
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
    # O: Ts_preheater (°C)
    # -------------------------
    if t == 0:
        x_next["Ts_preheater"] = 101.6
    else:
        x_next["Ts_preheater"] = (
            x["Ts_preheater"]
            + 0.0081 * (x["Tg_preheater"] - x["Ts_preheater"])
        )

    # -------------------------
    # P: Tg_calcination (°C)
    # -------------------------
    if t == 0:
        x_next["Tg_calcination"] = x_next["Tg_preheater"]
    else:

        if t <= 36:

            w = 1 / (1 + np.exp(-0.25 * (t - 36)))

            P_new = x["Tg_calcination"] + 0.0114 * (1300 - x["Tg_calcination"])
            noise = np.random.normal(0, 1.5)

            x_next["Tg_calcination"] = P_new + w * noise

        else:

            if t <= 218:

                noise = np.random.normal(0, 1.5)

                x_next["Tg_calcination"] = x["Tg_calcination"] + noise

            else:

                noise = np.random.normal(0, 1.5)

                ref = x["Tg_calcination"]  # approximation of INDEX(P:P,218)

                x_next["Tg_calcination"] = ref + noise
                
    # -------------------------
    # Q: Ts_calcination (°C)
    # -------------------------
    if t == 0:
        x_next["Ts_calcination"] = x_next["Ts_preheater"]
    else:
        x_next["Ts_calcination"] = (
            x["Ts_calcination"]
            - 0.0018217 * (x["Tg_calcination"] - x["Ts_calcination"])
        )
        
    # -------------------------
    # R: Tg_burning (°C)
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
    # S: Ts_burning (°C)
    # -------------------------
    if t == 0:
        x_next["Ts_burning"] = x_next["Ts_calcination"]
    else:

        w = 1 / (1 + np.exp(-0.25 * (t - 36)))

        S_new = x["Ts_burning"] + 0.0221 * (1455 - x["Ts_burning"])

        noise = np.random.normal(0, 1.5) * w

        x_next["Ts_burning"] = S_new + noise
        
    # =========================
    # T: Tg_Cooling
    # =========================
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
            
    # =========================
    # U: Ts_Cooling (°C)
    # =========================
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
    # =========================
    # V: O2 (%)
    # =========================
    if t == 0:
        x_next["O2"] = 6.0
    else:
        x_next["O2"] = (
            3.2
            + (6.0 - 3.2) * np.exp(-t / 10)
        )
    # =========================
    # W: CO ppm
    # =========================
    if t == 0:
        x_next["CO_ppm"] = 900.0
    else:
        x_next["CO_ppm"] = (
            36.0
            + (900.0 - 36.0) * np.exp(-t / 10)
        )
    # =========================
    # X: P_preheater (Pa)
    # =========================
    if t == 0:
        x_next["P_preheater"] = -120.0
    else:
        x_next["P_preheater"] = (
            -269.0
            + (-120.0 + 269.0) * np.exp(-t / 15)
        )
    # =========================
    # Y: P_calcination (Pa)
    # =========================
    if t == 0:
        x_next["P_calcination"] = -180.0
    else:
        x_next["P_calcination"] = (
            -406.0
            + (-180.0 + 406.0) * np.exp(-t / 20)
        )
    # =========================
    # Z: Damper_position (%)
    # =========================
    if t == 0:
        x_next["Damper_position"] = 85.0
    else:
        x_next["Damper_position"] = (
            33.0
            + (85.0 - 33.0) * np.exp(-t / 25)
        )
    # =========================
    # AB: CaCO3 (ton/h)
    # =========================
    if t == 0:
        x_next["CaCO3"] = 80.0
    else:

        k = (
            10000000
            * np.exp(-160000 / (8.314 * (x["Ts_calcination"] + 273.15)))
        )

        x_next["CaCO3"] = x["CaCO3"] * np.exp(-k *dt)
    # =========================
    # AC: CaO (ton/h)
    # =========================
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
    # =========================
    # AD: CO2 (ton/h)
    # =========================
    if t == 0:
        x_next["CO2"] = 1e-6
    else:
        x_next["CO2"] = (
            80.0
            - (x_next["CaCO3"] + x_next["CaO"])
        )
    # -------------------------
    # AE: SiO2
    # -------------------------
    if t == 0:
        x_next["SiO2"] = 13.0
    else:
        x_next["SiO2"] = max(
            0.0,
            x["SiO2"] - (x["C3A"] * 0.3488383838 + x["C4AF"] * 0.2631526466)
        )

    # -------------------------
    # AF: Al2O3
    # -------------------------
    if t == 0:
        x_next["Al2O3"] = 4.0
    else:
        x_next["Al2O3"] = max(
            0.0,
            x["Al2O3"] - (x["C3A"] * 0.3773565314 + x["C4AF"] * 0.209816997)
        )

    # -------------------------
    # AG: Fe2O3
    # -------------------------
    if t == 0:
        x_next["Fe2O3"] = 3.0
    else:
        x_next["Fe2O3"] = max(
            0.0,
            x["Fe2O3"] - (x["C4AF"] * 0.3286167885)
        )

    # -------------------------
    # AH: LSF
    # -------------------------
    x_next["LSF"] = x_next["CaO"] / (
        2.8 * x_next["SiO2"] +
        1.2 * x_next["Al2O3"] +
        0.65 * x_next["Fe2O3"] + 1e-9
    )

    # -------------------------
    # AI: C3A
    # -------------------------
    if t == 0:
        x_next["C3A"] = 1e-6
    else:
        x_next["C3A"] = x["C3A"] + x_next["Al2O3"]
    # -------------------------
    # AJ: C4AF
    # -------------------------
    if t == 0:
        x_next["C4AF"] = 1e-6
    else:
        x_next["C4AF"] = (
            x["C4AF"]
            + x_next["dC4AF"]
        )  
    # -------------------------
    # AK: C2S
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
    # AL: C3S
    # -------------------------
    if t == 0:
        x_next["C3S"] = 1e-6
    else:
        x_next["C3S"] = (
            x["C3S"]
            + x_next["dC3S"]
        )
    # -------------------------
    # AM: dC2S
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
    # AN: dC3S
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
    # AO: dC3A
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
    # AP: dC4AF
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
    # -------------------------
    # AQ: dCaO_calcination
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
    # AR: Mass_Balance_Error
    # -------------------------
    if t == 0:
        x_next["Mass_Balance_Error"] = 1e-6
    else:

        inputs = (
            x_next["CaCO3"]
            + x_next["SiO2"]
            + x_next["Al2O3"]
            + x_next["Fe2O3"]
        )

        outputs = (
            x_next["CaO"]
            + x_next["CO2"]
            + x_next["C3A"]
            + x_next["C4AF"]
            + x_next["C2S"]
            + x_next["C3S"]
        )

        x_next["Mass_Balance_Error"] = inputs - outputs
    # -------------------------
    # AS: kiln_rpm
    # -------------------------
    if t == 0:
        x_next["kiln_rpm"] = 0.1
    else:

        REGIME_RPM_COEFF = {
            "R1_HEATING_STABILIZATION": 0.55,
            "R2_EARLY_CALCINATION": 0.7,
            "R3_ACTIVE_CALCINATION": 0.8,
            "R4_TRANSITION_TO_CLINKERIZATION": 0.9,
            "R5_EARLY_CLINKERIZATION": 0.95,
            "R6_STEADY_CLINKERIZATION": 1.0,
            "R7_FUEL_SWITCH_TRANSIENT": 1.05,
            "R8_RESTABILIZATION": 0.85,
        }

        coeff = REGIME_RPM_COEFF[regime]

        feed_term = (
            x_next["Fuel_rate"]
            * 0.8
            * (1 - np.exp(-0.01 * t))
        )

        growth = (
            x_next["Fuel_rate"]
            + (6 - x_next["Fuel_rate"]) * (1 - np.exp(-0.00041 * t))
        )

        x_next["kiln_rpm"] = (
            0.1
            + growth * coeff * feed_term
        )
    # -------------------------
    # AT: Filling_rate
    # -------------------------
    x_next["Filling_rate"] = 0.1
    
    # -------------------------
    # AU: Residence (min)
    # -------------------------
    import math

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
    # AV: Q_in (kJ/kg)
    # -------------------------
    x_next["Q_in"] = (
        (x_next["Lignite_Coal"] * 15000
         + x_next["Petcoke"] * 30000
         + x_next["Alternative_Fuel"] * 18000)
        * x_next["Fuel_rate"]
        * np.exp(-((x_next["O2"] - 3.5) ** 2) / 25)
    )
    # -------------------------
    # AW: Q_out (kJ/kg)
    # -------------------------
    x_next["Q_out"] = (
        x_next["Q_acc"]
        + x_next["Q_loss"]
        + x_next["Q_reaction"]
    )
    # -------------------------
    # AX: Q_acc (kJ/kg)
    # -------------------------
    x_next["Q_acc"] = x_next["Clinker_output"] * 150
    
    # -------------------------
    # AY: Q_loss
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
    # AZ: Q_reaction
    # -------------------------
    x_next["Q_reaction"] = (
        x_next["Feed_rate"]
        * (1 - x_next["Clinker_yield"])
        * 3.2
    )
    # -------------------------
    # BA: Q_gas
    # -------------------------
    x_next["Q_gas"] = (
        x_next["Fuel_rate"]
        * 1.1
        * (x_next["Tg_burning"] - 25)
    )
    # -------------------------
    # BB: Q_clinker
    # -------------------------
    x_next["Q_clinker"] = (
        x_next["Clinker_output"]
        * 1800
    )
    # -------------------------
    # BC: Clinker_yield
    # -------------------------
    feed_safe = max(x_next["Feed_rate"], 1e-6)

    x_next["Clinker_yield"] = (
        x_next["Clinker_output"]
        / feed_safe
    )
    # -------------------------
    # BD: dTg_burning
    # -------------------------
    if t == 0:
        x_next["dTg_burning"] = 1e-6
    else:

        x_next["dTg_burning"] = (
            x_next["Tg_burning"]
            - x["Tg_burning"]
        ) / dt
        
    # -------------------------
    # BE: dFuel_rate
    # -------------------------
    if t == 0:
        x_next["dFuel_rate"] = 1e-6
    else:

        x_next["dFuel_rate"] = (
            x_next["Fuel_rate"]
            - x["Fuel_rate"]
        ) / dt
    # -------------------------
    # BF: Normalized_Energy_Index
    # -------------------------
    clinker_safe = max(x_next["Clinker_output"], 1e-6)

    x_next["Normalized_Energy_Index"] = (
        x_next["Q_in"]
        / clinker_safe
    )
    # -------------------------
    # BG: Global_Energy_Closure
    # -------------------------
    x_next["Global_Energy_Closure"] = (
        x_next["Q_in"]
        - x_next["Q_out"]
        + x_next["Q_reaction"]
        - x_next["Q_acc"]
    )
    # -------------------------
    # BH: Energy_error (%)
    # -------------------------
    q_in_safe = max(x_next["Q_in"], 1e-6)

    x_next["Energy_error"] = (
        x_next["Global_Energy_Closure"]
        / q_in_safe
    ) * 100 
        # -------------------------
    # BI: SCALE
    # -------------------------
    eps = 1e-9

    if t == 0:
        x_next["SCALE"] = 1.0
    else:

        denom1 = abs(3*x_next["dC3S"] + 2*x_next["dC2S"] + 3*x_next["dC3A"] + 4*x_next["dC4AF"]) + eps
        denom2 = abs(x_next["SiO2"] + x_next["dC2S"]) + eps
        denom3 = abs(x_next["Al2O3"] + x_next["dC3A"]) + eps
        denom4 = abs(x_next["Fe2O3"] + eps)

        scale = min(
            1.0,
            x_next["CaO"] / denom1,
            x_next["SiO2"] / denom2,
            x_next["Al2O3"] / denom3,
            x_next["Fe2O3"] / denom4
        )

        x_next["SCALE"] = max(0.0, scale)                                                                                                             
    return x_next

# 1. GLOBAL KONFİGÜRASYON
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

# 3. MAIN SIMULATION LOOP
sim_time = 0.0  # Fiziksel zaman başlangıcı (saat)

for t in range(N - 1):
    x_t = df.iloc[t].to_dict()
    
    # Zaman yönetimi: t indisi değil, sim_time zamanı raporlar
    x_t["t"] = sim_time 
    regime = x_t["Regime"]

    # --- INPUT LAYER ---
    # Not: Bazı girişler zamana (t indis) bağlıydı, fiziksel denge için 'sim_time' bazlı 
    # ölçekleme yapmak daha doğru olabilir, ancak mevcut formülleri korudum.
    D = 0.03 + (0.1 - 0.03) * (1 - np.exp(-t / 80))
    E = 0.07 + (0.14 - 0.07) * (1 - np.exp(-t / 100))
    x_t["Petcoke"] = D
    x_t["Alternative_Fuel"] = E
    x_t["Lignite_Coal"] = max(0.0, 1.0 - D - E)

    base_feed = (
        132 if t >= 72 else
        72 + 60 * ((1 / (1 + np.exp(-0.065 * (t - 36)))) - 0.09) / 0.82
    )
    x_t["Feed_rate"] = base_feed * REGIME_FEED_MULT.get(regime, 1.0)

    # --- STATE UPDATE (SUB-STEPPING) ---
    x_current = x_t.copy()
    
    for sub in range(STEPS_PER_REPORT):
        step_time = sim_time + (sub * dt)
        x_current = step(x_current, step_time, regime)

    # DataFrame'e fiziksel zamanı da içeren güncel veriyi yaz
    x_current["t"] = sim_time + reporting_dt
    df.iloc[t + 1] = pd.Series(x_current)
    
    # Zamanı raporlama periyodu kadar ilerlet
    sim_time += reporting_dt

# 6. SAVE TO CSV
output_path = "kiln_simulation_output.csv"
df.to_csv(output_path, index=False)
print(f"Simülasyon başarıyla tamamlandı: {output_path}")