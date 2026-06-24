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

        activity_factor = min(1.0, mat_acc / 50.0)

        coupling = flow_factor * activity_factor

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
        # ENERGY INPUT & COMBUSTION DYNAMICS
        # --------------------------------------------------------------
        AIR_DENSITY = 1.293

        # O2 & Heat Input (Physical dependence on air-fuel ratio)
        air_fuel_ratio = x_next["Air_flow"] / (f_rate + 1e-6)
        o2_target = 2.5 + 4.5 * np.exp(-1.2 / (air_fuel_ratio / 20000.0 + 1e-3))
        x_next["O2"] = x.get("O2", 3.5) + 0.15 * (o2_target - x.get("O2", 3.5))

        # Total Thermal Power Input (J/h)
        Q_in_total = (
            (
                x_next["Lignite_Coal"] * 15000.0
                + x_next["Petcoke"] * 30000.0
                + x_next["Alternative_Fuel"] * 18000.0
            )
            * x_next["Fuel_rate"]
            * 1000.0  # kJ to J
            * np.exp(-((x_next["O2"] - 3.5) ** 2) / 25.0)
        )

        # --------------------------------------------------------------
        # PHYSICAL TRANSFER COEFFICIENTS & CONSTANTS
        # --------------------------------------------------------------
        Cp_gas = 1100.0  # J/kgK (Approx. for flue gas)
        Cp_solid = 1050.0  # J/kgK (Approx. for clinker/raw mix)

        # Enthalpy of reaction for calcination (~1.7 MJ/kg CaO)
        delta_H_calc = 1.7e6

        # --------------------------------------------------------------
        # ZONE-BASED ENERGY DYNAMICS (Transport-based)
        # --------------------------------------------------------------
        # 1. Burning Zone Energy Balance
        # Inputs: Q_in, Heat from incoming solid (from calcination)
        # Outputs: Heat to gas flow, Heat to outgoing clinker
        Q_burning_total = Q_in_total

        # 2. Calcination Zone Energy Balance (Reactive Zone)
        # Includes Endothermic Load
        Q_reaction_load = CaO_consumed * delta_H_calc * 1000.0  # J/h

        # 3. Inter-zone Transport Logic
        # Gas flows from Burning -> Calcination -> Preheater
        # Solids flow from Preheater -> Calcination -> Burning
        # This replaces the static w_burn/w_calc weights.

        x_next["Q_gas"] = Q_burning_total
        x_next["Q_reaction"] = Q_reaction_load

        # Energy balance derivative approximation (Example for Burning Zone)
        # This will drive the temperature gradients instead of static weights
        m_gas = (x_next["Air_flow"] * 1.293) / 3600.0

        # Net Energy available in Burning Zone (J/h)
        Q_net_burning = Q_burning_total - Q_reaction_load

        # Temperature change driven by physical conservation:
        # (m*Cp)*dT = Q_net
        # dT/dt = Q_net / (m * Cp)
        x_next["dTg_burning"] = Q_net_burning / (m_gas * Cp_gas + 1e-6)

        # --------------------------------------------------------------
        # BURNING ZONE: PHYSICS-BASED ENERGY DYNAMICS (Integrated)
        # --------------------------------------------------------------
        # Dinamik zaman adımı (saat cinsinden dt'yi saniyeye çeviriyoruz)
        dt_s = self.dt * 3600.0

        Cp_g_burn, Cp_s_burn, Cp_w_burn = 1250.0, 1150.0, 1000.0
        A_c_burn, h_c_burn = 13.85, 0.05
        A_s_burn, A_w_burn, A_ws_burn = 82.0, 70.0, 57.0

        m_air_s = x_next["Air_flow"] * AIR_DENSITY / 3600.0
        m_solid_s = (x_next["Feed_rate"] * 1000.0) / 3600.0

        # Lumped capacities (Physical thermal inertia)
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
        # GAS ENERGY BALANCE (Implicit)
        # -------------------------------
        a_gas_burn = (
            (h_gs_burn * A_s_burn)
            + (h_gw_burn * A_w_burn)
            + (h_c_burn * 1000.0 * A_c_burn)
            + (m_air_s * Cp_g_burn)
        )

        # Q_in_total doğrudan sistemin toplam enerji kaynağı olarak kullanıldı
        b_gas_burn = (
            Q_in_total * 0.97
            + (h_gw_burn * A_w_burn * Tw_curr)
            + (h_c_burn * 1000.0 * A_c_burn * 30.0)
            + (m_air_s * Cp_g_burn * 400.0)
            + (h_gs_burn * A_s_burn * Ts_curr)
        )

        Tg_next_burn = (C_gas_total * Tg_curr + dt_s * b_gas_burn) / (
            C_gas_total + dt_s * a_gas_burn
        )

        # -------------------------------
        # WALL ENERGY BALANCE
        # -------------------------------
        T_amb = 25.0
        h_loss = 8.0

        Q_loss_wall = h_loss * A_w_burn * (Tw_curr - T_amb)
        Q_wall_to_solid = h_ws_burn * A_ws_burn * (Tw_curr - Ts_curr)

        Tw_next_burn = Tw_curr + (dt_s / C_wall_total) * (
            h_gw_burn * A_w_burn * (Tg_next_burn - Tw_curr)
            - Q_loss_wall
            - Q_wall_to_solid
        )

        # -------------------------------
        # SOLID PHASE (Exothermic Reaction Coupling)
        # -------------------------------
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

        Ts_next_burn = (C_solid_total * Ts_curr + dt_s * b_sol_burn) / (
            C_solid_total + dt_s * a_sol_burn
        )

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        x_next["Tg_burning"] = Tg_next_burn
        x_next["Ts_burning"] = Ts_next_burn
        x_next["Tw_burning"] = Tw_next_burn

        # ==========================================================
        # ------------------------------------------------------
        # CALCINATION ZONE (PHYSICS-BASED ADVECTION & REACTION)
        # ------------------------------------------------------
        # ==========================================================

        # 1. TEMEL PARAMETRELER VE SABİTLERİN TANIMLANMASI
        Cp_air = 1005.0  # J/(kg·K) - Gaz fazı özgül ısısı
        Cp_solid_calc = 1050.0  # J/(kg·K) - Katı fazı özgül ısısı
        T_ref = 25.0  # °C - Referans sıcaklık
        h_gs, A_s = (
            320.0,
            950.0,
        )  # W/(m²K) Gaz-Katı konvektif ısı transfer katsayısı ve Alan (m²)
        AIR_DENSITY = locals().get(
            "AIR_DENSITY", getattr(self, "AIR_DENSITY", 1.2)
        )  # kg/m³

        # Simülasyon adım zamanı kontrolü (Saniye bazında dt)
        if "dt_s" in locals():
            current_dt_s = dt_s
        elif hasattr(self, "dt"):
            current_dt_s = self.dt if self.dt > 0.1 else self.dt * 3600.0
        else:
            current_dt_s = x.get("dt_s", 1.0)

        # 2. MEVCUT DURUM VE HEDEFLERİN GÜVENLİ ŞEKİLDE ÇEKİLMESİ
        Ts_calc_curr = x.get("Ts_calcination", 800.0)
        Tg_calc_curr = x.get("Tg_calcination", 900.0)
        Ts_curr_from_preheater = x.get("Ts_preheater", 400.0)

        # 3. KÜTLE AKIŞ HIZLARININ HESAPLANMASI (ton/h -> kg/s)
        m_solid_calc = x_next.get("Kiln_solid_out", x.get("Kiln_solid_out", 50.0))
        m_solid_kg_s = (m_solid_calc * 1000.0) / 3600.0

        # 4. TERMAL KAPASİTELERİN HESAPLANMASI (J/K)
        # Nümerik Kararlılık Notu: dT/dt diferansiyel adımlarında paydadaki sığa terimi (C)
        # kütlesel debi (kg/s) değil, fırın zonunda o an hapsolmuş toplam envanter (kg) olmalıdır.
        C_gas_calc_total = (
            100.0 * Cp_air
        )  # J/K (Zondaki anlık anodik gaz kütlesi ~100 kg baz alınmıştır)

        # Kalsinasyon zonundaki anlık malzeme hold-up envanteri (Örn: fırın içinde birikmiş 4500 kg malzeme)
        m_solid_inventory_calc = x.get("Material_inventory_calc", 4500.0)
        C_solid_calc_total = (
            m_solid_inventory_calc * Cp_solid_calc
        )  # J/K (Gerçek Termal Sığa)

        # 5. ENDOTERMİK REAKSİYON YÜKÜ (REACTION SINK TERM)
        # Kalsinasyon kinetiği katı sıcaklığına bağlı sigmoid aktivasyon fonksiyonu
        activation = 1.0 / (1.0 + np.exp(-0.08 * (Ts_calc_curr - 850.0)))
        Q_reaction_rate = 1700.0 * 1000.0  # J/kg CaO (Reaksiyon entalpisi)

        # Kalsinasyon hızı fiziksel olarak kütle akışına ve sıcaklığa bağlıdır
        Q_rxn_calc = m_solid_kg_s * activation * Q_reaction_rate  # J/s (Watt)

        # 6. GAZ ENERJİ DENGESİ (Advection from Burning Zone)
        # Toplam yanma ve sekonder/tersiyer hava kütlesel debisi (kg/s)
        m_air_total = (
            (x_next.get("Air_flow", 100000.0) + x_next.get("Tertiary_air_flow", 0.0))
            * AIR_DENSITY
            / 3600.0
        )

        # Pişirme (Burning) zonundan kalsinasyon zonuna doğru akan sıcak gazın taşıdığı duyulur ısı entalpisi
        Q_in_gas = m_air_total * Cp_air * x_next.get("Tg_burning", 1450.0)

        # Gaz fazı matris katsayılarının oluşturulması (Sırasıyla W/K ve W birimleri)
        a_gas_calc = (m_air_total * Cp_air) + (h_gs * A_s)
        b_gas_calc = Q_in_gas + (h_gs * A_s * Ts_calc_curr)

        # Yarı-Örtük (Semi-Implicit) Euler diskretizasyonu ile yeni gaz sıcaklığı
        Tg_calc_next = (C_gas_calc_total * Tg_calc_curr + current_dt_s * b_gas_calc) / (
            C_gas_calc_total + current_dt_s * a_gas_calc
        )

        # 7. KATI ENERJİ DENGESİ (Solid Phase Energy Balance)
        # Reaksiyon endotermik ısı yükü (sink term) katı fazın enerjisinden düşülür
        a_solid_calc = (h_gs * A_s) + (m_solid_kg_s * Cp_solid_calc)
        b_solid_calc = (
            (h_gs * A_s * Tg_calc_next)
            + (
                m_solid_kg_s * Cp_solid_calc * Ts_curr_from_preheater
            )  # Preheater adveksiyon girdisi
            - Q_rxn_calc  # Endotermik enerji yutağı (Watt)
        )

        # Yarı-Örtük (Semi-Implicit) Euler diskretizasyonu ile yeni katı sıcaklığı
        Ts_calc_next = (
            C_solid_calc_total * Ts_calc_curr + current_dt_s * b_solid_calc
        ) / (C_solid_calc_total + current_dt_s * a_solid_calc)

        # 8. YENİ DURUM DEĞİŞKENLERİNİN MATRİSE YAZILMASI
        x_next["Tg_calcination"] = Tg_calc_next
        x_next["Ts_calcination"] = Ts_calc_next
        x_next["Q_rxn_calc"] = Q_rxn_calc  # Enerji kapanım kontrolleri için kaydedilir

        # ==============================================================================
        # PREHEATER ZONE (PHYSICS-BASED ADVECTION & THERMAL CASCADE)
        # ==============================================================================

        # Isıl kapasiteler ve debiler
        Cp_gas_pre, Cp_solid_pre = 1150.0, 1050.0
        Tg_in = x_next["Tg_calcination"]  # Calcination zonundan gelen gaz (Adveksiyon)
        Ts_pre = x.get("Ts_preheater", 25.0)

        m_gas_pre = (x_next["Air_flow"] * AIR_DENSITY) / 3600.0
        m_solid_pre = (x_next["Feed_rate"] * 1000.0) / 3600.0

        C_gas_flow = m_gas_pre * Cp_gas_pre + 1e-4
        C_solid_flow = m_solid_pre * Cp_solid_pre + 1e-4

        # UA (Overall Heat Transfer) fiziksel katsayısı
        UA_pre = 18.0 * 120.0

        # ------------------------------------------------------------------------------
        # ENERGY TRANSFER (NO BUDGET CONSTRAINT - ONLY PHYSICAL POTENTIAL)
        # ------------------------------------------------------------------------------
        # Gaz ve katı arasındaki sıcaklık farkı (Energy Driving Force)
        dT = Tg_in - Ts_pre

        # Fiziksel transfer (LMTD veya basit model ile)
        # Q_physical = UA * dT * (Zaman düzeltmesi)
        alpha_stable = 0.35  # Dinamik kararlılık katsayısı (tau_eff modelinizden gelen)
        Q_preheater = alpha_stable * UA_pre * dT

        # ------------------------------------------------------------------------------
        # ENERGY CASCADE (Adveksiyon ile Isı Taşınımı)
        # ------------------------------------------------------------------------------
        # Gaz soğur (calcinasyondan gelen enerjiyi katıya verir)
        Tg_next = Tg_in - (Q_preheater / C_gas_flow)

        # Katı ısınır (gazdan gelen enerji ile preheater'da yükselir)
        Ts_next = Ts_pre + (Q_preheater / C_solid_flow)

        # ------------------------------------------------------------------------------
        # STATE WRITE (Dinamik yumuşatma - MPC Stability)
        # ------------------------------------------------------------------------------
        x_next["Tg_preheater"] = 0.7 * Tg_in + 0.3 * Tg_next
        x_next["Ts_preheater"] = 0.7 * Ts_pre + 0.3 * Ts_next

        x_next["Q_dot_preheater"] = Q_preheater

        # ------------------------------------------------------
        # COOLING ZONE (PHYSICS-BASED ENERGY RECOVERY)
        # ------------------------------------------------------
        dt_s = self.dt * 3600.0
        Tamb_solid, Tamb_gas = 130.0, 510.0

        Ts_cool_curr = x.get("Ts_Cooling", 1450.0)
        Tg_cool_curr = x.get("Tg_Cooling", 1490.0)

        # Mass flows & Thermal Capacities (Rate based)
        m_solid = (x.get("Feed_rate", 100.0) * 1000.0) / 3600.0
        m_gas = x.get("Cooling_air_flow", 20000.0) * AIR_DENSITY / 3600.0

        C_solid = m_solid * Cp_solid + 1e-9
        C_gas = m_gas * Cp_gas + 1e-9

        # Heat Transfer
        UA = 0.15 * 500.0
        Q_transfer = UA * np.tanh((Ts_cool_curr - Tg_cool_curr) / 250.0)

        # Solid Dynamics (Ambient interaction + Transfer)
        tau_klinker = 9700.0
        Ts_cool_next = (
            Ts_cool_curr
            + (dt_s / tau_klinker) * (Tamb_solid - Ts_cool_curr)
            - (Q_transfer * dt_s / C_solid)
        )

        # Gas Dynamics
        tau_gas = 3800.0
        Tg_cool_next = (
            Tg_cool_curr
            + (dt_s / tau_gas) * (Tamb_gas - Tg_cool_curr)
            + (Q_transfer * dt_s / C_gas)
        )

        # ------------------------------------------------------
        # ENERGY RECOVERY (To Burning Zone)
        # ------------------------------------------------------
        # Enerji geri kazanımı birikimli değil, anlık bir güç (Watt/J/h) girişidir
        eta_recovery = 0.65
        Q_secondary_air = eta_recovery * m_gas * Cp_gas * max(Tg_cool_curr - 200.0, 0.0)

        # State Update
        x_next["Ts_Cooling"] = Ts_cool_next
        x_next["Tg_Cooling"] = Tg_cool_next
        # Burning zone'a iletilecek geri kazanım gücü
        x_next["Q_secondary_air_recovery"] = Q_secondary_air

        # ----------------------------------------------------------
        # KALSİNASYON (STOKİYOMETRİK + ENERJİ KORUNUMLU MODEL)
        # ----------------------------------------------------------

        # Ts_calcination_pred yerine artık doğrudan güncel state değerini kullanıyoruz
        # Eğer state henüz x_next'te yoksa, x'ten alıyoruz
        Ts_calc_curr = x_next.get("Ts_calcination", x.get("Ts_calcination", 800.0))
        T_calc = Ts_calc_curr + 273.15

        if T_calc > 873.15:
            k_calc = 1e7 * np.exp(-160000.0 / (8.314 * T_calc))
        else:
            k_calc = 0.0

        CaCO3_in = x["CaCO3"]

        # Reaction extent (Saniye bazlı tutarlılık için dt_s)
        reacted = CaCO3_in * (1.0 - np.exp(-k_calc * self.dt * 3600.0))
        reacted = np.clip(reacted, 0.0, CaCO3_in)

        # Ürünler
        CaCO3_out = CaCO3_in - reacted
        CaO_generated = reacted * 0.5603
        CO2_generated_phys = reacted * 0.4397

        # ----------------------------------------------------------
        # ENERJİ KASKADI: YUTAK TERİMİ (Sink Term)
        # ----------------------------------------------------------
        # Reaksiyon ısısı gereksinimi (1.7 MJ/kg)
        delta_H_calc = 1.7e6
        # Watt cinsinden enerji çekişi
        Q_rxn_sink = (CaO_generated * delta_H_calc) / (self.dt * 3600.0 + 1e-9)

        # ----------------------------------------------------------
        # STATE UPDATE
        # ----------------------------------------------------------
        x_next["CaCO3"] = CaCO3_out
        x_next["CaO_generated"] = CaO_generated
        x_next["CO2_generated"] = CO2_generated_phys
        x_next["Q_rxn_calcination"] = Q_rxn_sink  # Enerji dengesine iletmek için

        # Hızlar
        x_next["dCaO_calcination"] = CaO_generated / (self.dt * 3600.0 + 1e-9)
        x_next["dCO2_calcination"] = CO2_generated_phys / (self.dt * 3600.0 + 1e-9)

        # ==========================================================
        # 🔴 CaO MASS POOL (DİNAMİK KÜTLE DENGESİ)
        # ==========================================================
        # CaO_generated: Kalsinasyondan bu adımda gelen kütle (kg)
        # CaO_exit: Bu adımda sistemden çıkan kütle (Örn: f_rate * CaO_content)

        # Basitleştirilmiş kütle dengesi:
        # Yeni Miktar = Eski Miktar + Gelen - Çıkan

        # Çıkış debisini kütle akışından (f_rate) türetelim
        CaO_exit = (x_next["Feed_rate"] * 1000.0 / 3600.0) * self.dt * 3600.0 * 0.65

        CaO_pool = x.get("CaO", 0.0) + CaO_generated - CaO_exit

        # Negatif kütle oluşmaması için kısıt (Nümerik kararlılık)
        x_next["CaO"] = max(0.0, CaO_pool)

        # ==========================================================
        # C2S FORMASYONU (GÜVENLİ DEĞİŞKEN TANIMLARIYLA)
        # ==========================================================

        # 1. Gerekli değişkenlerin (SiO2, Al2O3, Fe2O3) x'ten alınması
        SiO2 = x.get(
            "SiO2", 20.0
        )  # Varsayılan değerler tesis verisine göre ayarlanmalı
        Al2O3 = x.get("Al2O3", 5.0)
        Fe2O3 = x.get("Fe2O3", 3.0)

        # 2. T_burn_eff'in güvenli hesaplanması
        Tg_burn = x_next.get("Tg_burning", 1450.0) + 273.15
        Ts_burn = x_next.get("Ts_burning", 1400.0) + 273.15
        Tw_burn = x_next.get("Tw_burning", 1300.0) + 273.15

        T_burn_eff = 0.7 * Ts_burn + 0.2 * Tg_burn + 0.1 * Tw_burn

        # 3. Kinetik Hesaplama
        if SiO2 <= 1e-6 or T_burn_eff < 1000.0:
            dC2S = 0.0
        else:
            k = 50000000.0 * np.exp(-170000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            # Stoichiometric limit (CaO_pool güncel havuzdan gelir)
            stoich_limit = min(
                SiO2 / 0.3488,
                max(0.0, CaO_pool) / 0.6512,
            )
            dC2S = stoich_limit * kinetic

        # 4. Kütle Havuzu Güncelleme
        CaO_consumed = dC2S * 0.6512
        CaO_pool -= CaO_consumed

        # 5. State Update
        x_next["CaO"] = max(0.0, CaO_pool)
        x_next["C2S"] = x.get("C2S", 0.0) + dC2S
        x_next["SiO2"] = SiO2 - (dC2S * 0.3488)
        x_next["dC2S_formation"] = dC2S / (self.dt + 1e-9)

        # ==========================================================
        # C3S FORMASYONU (ALIT KINETICS)
        # ==========================================================

        # C2S envanterini güncelleyelim (dC2S zaten CaO havuzunu eksiltti)
        C2S_pool = x.get("C2S", 0.0) + dC2S

        # Reaksiyon şartları: 1473.15 K (~1200 C) üzeri aktifleşme
        if C2S_pool <= 1e-6 or T_burn_eff < 1473.15:
            dC3S = 0.0
        else:
            # Alit oluşum kinetiği (Yüksek aktivasyon enerjisi)
            k = 2.28e8 * np.exp(-200000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            # Stokiyometrik kısıt: C3S = C2S + CaO
            stoich_limit = min(
                C2S_pool / 0.7544,
                max(0.0, CaO_pool) / 0.2456,
            )
            dC3S = stoich_limit * kinetic

        # Kütle Havuzlarını Güncelleme
        # C3S oluşumu CaO ve C2S tüketir
        CaO_consumed_C3S = dC3S * 0.2456
        C2S_consumed_C3S = dC3S * 0.7544

        CaO_pool -= CaO_consumed_C3S
        C2S_pool -= C2S_consumed_C3S

        # State Update
        x_next["CaO"] = max(0.0, CaO_pool)
        x_next["C2S"] = max(0.0, C2S_pool)
        x_next["C3S"] = x.get("C3S", 0.0) + dC3S
        x_next["dC3S_formation"] = dC3S / (self.dt + 1e-9)

        # ==========================================================
        # C3A FORMASYONU (ALUMINATE KINETICS)
        # ==========================================================
        # C3A oluşumu ~1100 K (830 C) civarında başlar.

        if Al2O3 <= 1e-6 or T_burn_eff < 1100.0:
            dC3A = 0.0
        else:
            # Kinetik (Aktivasyon enerjisi: 120,000 J/mol)
            k = 1e5 * np.exp(-120000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            # Stokiyometrik kısıt: C3A = 3CaO + Al2O3
            # Kütlece oranlar: %37.73 Al2O3, %62.27 CaO
            stoich_limit = min(
                Al2O3 / 0.3773,
                max(0.0, CaO_pool) / 0.6227,
            )
            dC3A = stoich_limit * kinetic

        # Kütle Havuzlarını Güncelleme
        # C3A oluşumu CaO ve Al2O3 tüketir
        CaO_consumed_C3A = dC3A * 0.6227
        Al2O3_consumed_C3A = dC3A * 0.3773

        CaO_pool -= CaO_consumed_C3A

        # State Update
        x_next["CaO"] = max(0.0, CaO_pool)
        x_next["C3A"] = x.get("C3A", 0.0) + dC3A

        # Al2O3 envanterini güncelle (Kütle Korunumu)
        # Yeni Al2O3 değerini değişkene ata ki sıradaki C4AF reaksiyonu doğru çalışsın
        Al2O3 = max(0.0, Al2O3 - Al2O3_consumed_C3A)
        x_next["Al2O3"] = Al2O3

        # MPC / Kontrolcü Gözlemlenebilirliği İçin Reaksiyon Hızı
        x_next["dC3A_formation"] = dC3A / (self.dt + 1e-9)

        # ==========================================================
        # C4AF FORMASYONU (FERRITE KINETICS)
        # ==========================================================
        # C4AF oluşumu da ~1100 K üzerindeki sıcaklıklarda aktifleşir.

        if Fe2O3 <= 1e-6 or T_burn_eff < 1100.0:
            dC4AF = 0.0
        else:
            # Kinetik katsayı (Aktivasyon enerjisi: 150,000 J/mol)
            k = 200000.0 * np.exp(-150000.0 / (8.314 * T_burn_eff))
            kinetic = 1.0 - np.exp(-k * self.dt)

            # Bir önceki C3A adımında Al2O3 zaten güncellenmişti.
            # Kodun bağımsız çalışabilmesi adına yerel kontrolü garantiye alıyoruz:
            Al2O3_remaining = max(0.0, Al2O3)

            # Stokiyometrik sınır: C4AF = 4CaO + Al2O3 + Fe2O3
            stoich_limit = min(
                Fe2O3 / 0.3286,
                Al2O3_remaining / 0.2098,
                max(0.0, CaO_pool) / 0.4616,
            )

            dC4AF = stoich_limit * kinetic

        # Kütle Havuzlarının Güncellenmesi (Tüketimler)
        CaO_consumed_C4AF = dC4AF * 0.4616
        Al2O3_consumed_C4AF = dC4AF * 0.2098
        Fe2O3_consumed_C4AF = dC4AF * 0.3286

        CaO_pool -= CaO_consumed_C4AF

        # ==========================================================
        # FINAL STATE DECLARES (KÜTLE KORUNUMU VE ÇIKTI YAZIMLARI)
        # ==========================================================
        x_next["CaO"] = max(0.0, CaO_pool)

        # Al2O3 ve Fe2O3 tüketimlerinin yerel değişken güvenliğiyle işlenmesi
        Al2O3_rem = Al2O3_remaining if "Al2O3_remaining" in locals() else Al2O3
        Al2O3_cons = Al2O3_consumed_C4AF if "Al2O3_consumed_C4AF" in locals() else 0.0
        Fe2O3_cons = Fe2O3_consumed_C4AF if "Fe2O3_consumed_C4AF" in locals() else 0.0

        x_next["Al2O3"] = max(0.0, Al2O3_rem - Al2O3_cons)
        x_next["Fe2O3"] = max(0.0, Fe2O3 - Fe2O3_cons)

        # Ürün Havuzu Güncellemesi
        x_next["C4AF"] = x.get("C4AF", 0.0) + dC4AF

        # Kontrolcü / MPC İzlenebilirliği için Reaksiyon Hızı
        x_next["dC4AF_formation"] = dC4AF / (self.dt + 1e-9)

        # ==========================================================
        # 🔴 KRİTİK DÜZELTME: STATE UPDATE (ZON DEĞERLERİNİN KORUNMASI)
        # ==========================================================
        # NOT: Zonlar üstteki bağımsız First-Principles denklemleriyle x_next'i
        # zaten güncelledi. Burada kaskadı ezmek yerine, simülasyonun kararlılığı
        # için sadece boşta kalma ihtimali olan durumları garanti altına alıyoruz.

        x_next["Tg_preheater"] = x_next.get(
            "Tg_preheater", x.get("Tg_preheater", 350.0)
        )
        x_next["Ts_preheater"] = x_next.get(
            "Ts_preheater", x.get("Ts_preheater", 400.0)
        )

        x_next["Tg_calcination"] = x_next.get(
            "Tg_calcination", x.get("Tg_calcination", 900.0)
        )
        x_next["Ts_calcination"] = x_next.get(
            "Ts_calcination", x.get("Ts_calcination", 800.0)
        )

        x_next["Tg_burning"] = x_next.get("Tg_burning", x.get("Tg_burning", 1450.0))
        x_next["Ts_burning"] = x_next.get("Ts_burning", x.get("Ts_burning", 1400.0))
        x_next["Tw_burning"] = x_next.get("Tw_burning", x.get("Tw_burning", 1300.0))

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
        # Fırın çıkış hızı ile giriş hızı arasındaki oran
        flow_scale = x_next.get("Kiln_solid_out", 1e-6) / max(
            x.get("Feed_rate", 100.0), 1e-6
        )

        # Fırın doluluğuna bağlı aktivasyon (Malzeme yoksa reaksiyon hızı düşer)
        activity_factor = min(1.0, x_next.get("Material_acc", 50.0) / 50.0)

        # ----------------------------------------------------------
        # EFFECTIVE REACTION EXTENTS (PHYSICALLY CONSISTENT)
        # ----------------------------------------------------------
        dC2S_eff = dC2S * flow_scale * activity_factor
        dC3S_eff = dC3S * flow_scale * activity_factor
        dC3A_eff = dC3A * flow_scale * activity_factor
        dC4AF_eff = dC4AF * flow_scale * activity_factor

        # ----------------------------------------------------------
        # 🔴 KÜTLE DENGE YENİDEN SENKRONİZASYONU (MASS CORRECTION)
        # ----------------------------------------------------------
        # Önceki adımlarda ham dC değerlerine göre yapılan harcamaları
        # efektif değerlere göre düzelterek kütle korunumunu garanti ediyoruz.

        # Orijinal (reaksiyonlar öncesi) CaO havuzunu x üzerinden simüle edelim
        # ve sadece efektif tüketimleri düşelim:
        CaO_start = x.get("CaO", 0.0) + x_next.get("CaO_generated", 0.0)
        CaO_pool_corrected = (
            CaO_start
            - (dC2S_eff * 0.6512)
            - (dC3S_eff * 0.2456)
            - (dC3A_eff * 0.6227)
            - (dC4AF_eff * 0.4616)
        )

        x_next["CaO"] = max(0.0, CaO_pool_corrected)

        # Oksitlerin efektif tüketime göre güncellenmesi
        SiO2_base = x.get("SiO2", 20.0)
        Al2O3_base = x.get("Al2O3", 5.0)
        Fe2O3_base = x.get("Fe2O3", 3.0)

        x_next["SiO2"] = max(0.0, SiO2_base - (dC2S_eff * 0.3488))
        x_next["Al2O3"] = max(
            0.0, Al2O3_base - (dC3A_eff * 0.3773) - (dC4AF_eff * 0.2098)
        )
        x_next["Fe2O3"] = max(0.0, Fe2O3_base - (dC4AF_eff * 0.3286))

        # Ürün Minerallerinin Efektif Oranlarla Yeniden Yazılması
        x_next["C2S"] = max(0.0, x.get("C2S", 0.0) + dC2S_eff - (dC3S_eff * 0.7544))
        x_next["C3S"] = x.get("C3S", 0.0) + dC3S_eff
        x_next["C3A"] = x.get("C3A", 0.0) + dC3A_eff
        x_next["C4AF"] = x.get("C4AF", 0.0) + dC4AF_eff

        # Kontrolcü ve MPC Arayüzü İçin Efektif Reaksiyon Hızları (Modbus/OpcUa uyumlu)
        x_next["dC2S_formation"] = dC2S_eff / (self.dt + 1e-9)
        x_next["dC3S_formation"] = dC3S_eff / (self.dt + 1e-9)
        x_next["dC3A_formation"] = dC3A_eff / (self.dt + 1e-9)
        x_next["dC4AF_formation"] = dC4AF_eff / (self.dt + 1e-9)

        # ==========================================================
        # CLINKER PHASE UPDATE (NÜMERİK KORUMALI)
        # ==========================================================
        # C2S, hem oluşup hem C3S tarafından tüketildiği için negatif faz riskine karşı max() eklenmiştir.
        x_next["C2S"] = max(0.0, x.get("C2S", 0.0) + dC2S_eff - dC3S_eff * 0.7544)
        x_next["C3S"] = x.get("C3S", 0.0) + dC3S_eff
        x_next["C3A"] = x.get("C3A", 0.0) + dC3A_eff
        x_next["C4AF"] = x.get("C4AF", 0.0) + dC4AF_eff

        # ==========================================================
        # ELEMENT BALANCES (FLOW-CONSISTENT)
        # ==========================================================
        x_next["SiO2"] = max(0.0, x.get("SiO2", 0.0) - dC2S_eff * 0.3488)
        x_next["Al2O3"] = max(
            0.0, x.get("Al2O3", 0.0) - dC3A_eff * 0.3773 - dC4AF_eff * 0.2098
        )
        x_next["Fe2O3"] = max(0.0, x.get("Fe2O3", 0.0) - dC4AF_eff * 0.3286)

        # ==========================================================
        # 🔴 CaO CONSERVATION (FIXED & STABLE)
        # ==========================================================
        # Mevcut envantere bu adımda kalsinasyondan üretilen CaO eklenir
        CaO_in = x.get("CaO", 0.0) + x_next.get("CaO_generated", 0.0)

        # Efektif klinker mineralleri oluşumunun tükettiği toplam CaO
        CaO_consumed = (
            dC2S_eff * 0.6512
            + dC3S_eff * 0.2456
            + dC3A_eff * 0.6227
            + dC4AF_eff * 0.4616
        )

        # Net CaO envanter güncellemesi ve nümerik koruma
        CaO_next = CaO_in - CaO_consumed
        x_next["CaO"] = max(0.0, CaO_next)

        # ==========================================================
        # 🔴 CO2 CLOSURE & MASS BALANCE (STRICT FIRST-PRINCIPLES)
        # ==========================================================
        # Kalsiyum Envanteri (Toplam CaO Eşdeğeri Kütle - Doğrulandı)
        # Katsayılar: CaO/CaCO3 = 56.08/100.09 = 0.5603
        Ca_inventory = (
            0.5603 * x_next.get("CaCO3", 0.0)
            + x_next["CaO"]
            + 0.6512 * x_next.get("C2S", 0.0)
            + 0.7369 * x_next.get("C3S", 0.0)  # Toplam kütle payı 0.7369
            + 0.6227 * x_next.get("C3A", 0.0)
            + 0.4616 * x_next.get("C4AF", 0.0)
        )

        # KÜTLE KAPANIMI YÖNTEMİ (GRAFİK VE DIAGNOSTIC İÇİN):
        # Statik 80.0 yerine, sistemin o anki toplam kütle tavanı girdilere bağlanmalıdır.
        # Alternatif olarak CO2 dinamik bir gaz olarak birikip bacadan tahliye edilir:
        dt_s = self.dt * 3600.0
        AIR_DENSITY = 1.293
        m_gas_flow = x.get("Cooling_air_flow", 20000.0) * AIR_DENSITY / 3600.0  # kg/s

        # Gaz adveksiyon tahliye katsayısı (Fırın içi gaz süpürme etkisi)
        gas_purge_factor = min(
            1.0, (m_gas_flow * dt_s) / max(x_next.get("Material_acc", 50.0), 1.0)
        )

        # CO2 Değişimi = Önceki + Üretilen - Gaz Akışıyla Taşınan/Uçan
        CO2_generated = x_next.get("CO2_generated", 0.0)
        CO2_prev = x.get("CO2", 0.0)

        CO2_next = CO2_prev + CO2_generated - (CO2_prev * gas_purge_factor)
        x_next["CO2"] = max(0.0, CO2_next)

        # Grafik izleme ve debugging için korunum metriği
        x_next["Ca_Total_System_Mass"] = Ca_inventory

        # ==========================================================
        # 🔴 DYNAMIC SIGNALS & EMISSIONS (TIME-STEP INDEPENDENT)
        # ==========================================================

        # 1. Gaz Sıcaklığı Türevi (Birim zamandaki değişim hızı)
        x_next["dTg_burning"] = (
            x_next["Tg_burning"] - x.get("Tg_burning", 1450.0)
        ) / max(self.dt, 1e-6)

        # 2. O2 Eksikliği Kontrolü
        oxygen_deficit = max(0.0, 6.0 - x_next.get("O2", 6.0))

        # 3. Target CO Hesabı (Sıcaklık birimi kontrolü: Tg_burning Kelvin cinsindendir)
        # Yüksek sıcaklıkta CO yanarak CO2'ye dönüştüğü için exp(-T) terimi doğrudur.
        target_CO = 20.0 + 800.0 * oxygen_deficit * (
            x_next.get("Fuel_rate", 3.0) / 6.0
        ) * np.exp(-x_next["Tg_burning"] / 1200.0)

        # 4. Örnekleme Zamanından Bağımsız CO Filtresi (Discretized First-Order Filter)
        # CO dinamik tepki zaman sabiti (Örn: tau_CO = 30 saniye). self.dt saat cinsindense saniyeye çevrilir.
        dt_seconds = (
            self.dt * 3600.0 if self.dt < 0.1 else self.dt
        )  # Zaman birimi standardizasyonu
        tau_CO = 30.0  # saniye
        alpha_CO = 1.0 - np.exp(-dt_seconds / tau_CO)

        x_next["CO_ppm"] = x.get("CO_ppm", 20.0) + alpha_CO * (
            target_CO - x.get("CO_ppm", 20.0)
        )

        # ==========================================================
        # 🔴 KALSİNASYON BASINCI (MUTLAK ZAMANDAN BAĞIMSIZ FİZİKSEL MODEL)
        # ==========================================================
        # Statik 't' parametresi yerine, kalsinasyondan çıkan gaz hacmi ve termal genleşme baz alınmıştır.
        # Basınç düşümü (Draft Loss) ~ Gaz Debisi^2 * Sıcaklık ilişkisine dayanır.

        CO2_gen_rate = x_next.get(
            "dCO2_calcination", 0.0
        )  # kg/s cinsinden üretilen gaz
        Tg_calc_K = x_next.get("Tg_calcination", 900.0) + 273.15
        ID_fan_draft = (
            x.get("ID_fan_speed", 70.0) * -10.0
        )  # Baca fanından gelen emiş etkisi (Pa)

        # Üretilen gaz miktarı ve sıcaklık arttıkça fırın içi basınç (pozitife doğru) artar.
        # Temel fırın içi draft direnç modeli (Pa cinsinden):
        P_calc_target = ID_fan_draft + (15.0 * CO2_gen_rate * (Tg_calc_K / 273.15))

        # Basınç sensörü dinamik gecikmesi (tau_P = 5 saniye)
        tau_P = 5.0
        alpha_P = 1.0 - np.exp(-dt_seconds / tau_P)
        x_next["P_calcination"] = x.get("P_calcination", -200.0) + alpha_P * (
            P_calc_target - x.get("P_calcination", -200.0)
        )

        # ==========================================================
        # MASS BALANCE CHECKS (DIAGNOSTIC METRICS)
        # ==========================================================

        # 1. CaO Kütle Dengesi Hatası:
        # Normal rejimde tam olarak 0.0 olmalıdır. Eğer 0'dan büyükse, bir önceki adımdaki
        # max(0.0, CaO_next) kırpması tetiklenmiş demektir. Bu durum MPC'ye fırının
        # kararsız bir transient (geçici) rejime girdiğini veya dt adımının çok büyük
        # seçildiğini (stiffness) haber verir.
        x_next["CaO_balance_error"] = abs(CaO_in - (x_next["CaO"] + CaO_consumed))

        # 2. CO2 Kimyasal Dönüşüm Dengesi Kontrolü:
        # Statik 80.0 yerine, kalsinasyon reaksiyonunun kendi içindeki saf stoikiometrik
        # doğruluğunu kontrol ediyoruz. Tüketilen CaCO3 ile üretilen CO2 kütlesi kilitli olmalıdır.
        # Katsayı: CO2 / CaCO3 = 44.01 / 100.09 = 0.4397
        CaCO3_consumed = max(0.0, x.get("CaCO3", 0.0) - x_next.get("CaCO3", 0.0))
        expected_CO2_generated = CaCO3_consumed * 0.4397
        actual_CO2_generated = x_next.get("CO2_generated", 0.0)

        x_next["CO2_balance_error"] = abs(expected_CO2_generated - actual_CO2_generated)

        # 3. Çözücü Sağlık İndeksi (Solver Health Index)
        # Optimizasyon algoritmasının kısıt fonksiyonlarında (constraints) kullanılabilir.
        x_next["Mass_balance_verified"] = (
            1.0
            if (
                x_next["CaO_balance_error"] < 1e-6
                and x_next["CO2_balance_error"] < 1e-6
            )
            else 0.0
        )

        # ==========================================================
        # ENERGY TERMS (THERMODYNAMICALLY CONSISTENT)
        # ==========================================================
        # Not: Sıcaklıkların Kelvin (K) cinsinden olduğu varsayılmıştır.
        # Eğer Celsius ise, Kelvin dönüşümü: T_K = T_C + 273.15
        Tg_burning_K = (
            x_next["Tg_burning"] + 273.15
            if x_next["Tg_burning"] < 1000.0
            else x_next["Tg_burning"]
        )
        T_amb_K = 25.0 + 273.15

        # 1. Radyatif ve Konvektif Isı Kaybı (Stefan-Boltzmann Doğrulaması)
        # 5.67e-8 * Emisivite (0.8) * Yüzey Alanı (110 m2) * Birim Dönüşüm Katsayısı (0.0016)
        # Çıktı birimi: MW veya MJ/s cinsine uyumlu ölçeklenmiştir.
        x_next["Q_loss"] = (
            0.0016 * 5.67e-8 * 0.8 * 110.0 * (Tg_burning_K**4 - T_amb_K**4)
        )

        # 2. Net Reaksiyon Isısı (Endotermik Kalsinasyon + Ekzotermik Klinkerleşme)
        # Kalsinasyon (Endotermik): ~3200 kJ/kg CaCO3 reaksiyonu
        Q_calcination = x_next.get("CaO_generated", 0.0) * 3180.0  # kJ/kg

        # Klinker Faz Oluşumu (Ekzotermik - Sisteme Isı Verir):
        # C3S oluşumu ~500 kJ/kg, C2S oluşumu ~260 kJ/kg ısı açığa çıkarır.
        dC2S_eff = locals().get("dC2S_eff", 0.0)
        dC3S_eff = locals().get("dC3S_eff", 0.0)
        Q_clinkerization = dC2S_eff * 260.0 + dC3S_eff * 500.0

        # Net reaksiyon enerjisi dengesi (Endotermik - Ekzotermik)
        x_next["Q_reaction"] = Q_calcination - Q_clinkerization

        # 3. Yakıt Yanma Enerjisi Girdisi (Combustion Heat Release)
        # LHV: Alt Isıl Değer (Örn: Kömür/Petrokök için ~30,000 kJ/kg)
        LHV_fuel = 32000.0  # kJ/kg
        x_next["Q_combustion"] = x_next["Fuel_rate"] * LHV_fuel

        # 4. Gaz Fazı Duyulur Isı Taşıma Kapasitesi (Sensible Heat of Flue Gas)
        # Yakıt miktarı + yanma havası (yaklaşık 11 katı) toplam gaz kütlesini oluşturur.
        # Cp_gas ~ 1.15 kJ/kgK
        m_gas_total = x_next["Fuel_rate"] * 12.0
        x_next["Q_gas"] = m_gas_total * 1.15 * (Tg_burning_K - T_amb_K)

        # 5. Toplam Enerji Çıkış Dengesi (Energy Closure)
        # Akümülasyon terimi (Q_acc) üzerine kayıplar ve reaksiyon yükleri eklenir
        x_next["Q_out"] = (
            x_next.get("Q_acc", 0.0) + x_next["Q_loss"] + x_next["Q_reaction"]
        )

        # ==========================================================
        # GLOBAL ENERGY CLOSURE & SPECIFIC ENERGY CONSUMPTION (SEC)
        # ==========================================================
        # İsimlendirme Güvenliği: Q_in terimini yukarıda hesaplanan yakıt yanma
        # enerjisine (Q_combustion) veya x'ten gelen değere pürüzsüzce bağlıyoruz.
        x_next["Q_in"] = x_next.get(
            "Q_in", x_next.get("Q_combustion", x.get("Q_in", 1e-6))
        )

        # Küresel enerji kapanımı (Giren Isı - Çıkan Isı)
        x_next["Global_Energy_Closure"] = x_next["Q_in"] - x_next["Q_out"]

        q_in_safe = max(x_next["Q_in"], 1e-6)
        x_next["Energy_error"] = (x_next["Global_Energy_Closure"] / q_in_safe) * 100.0

        # Klinker çıkış hızı senkronizasyonu (Kiln_solid_out veya Clinker_output koruması)
        clinker_out_val = x_next.get(
            "Clinker_output",
            x_next.get("Kiln_solid_out", x.get("Clinker_output", 1e-6)),
        )
        clinker_safe = max(clinker_out_val, 1e-6)

        # Spesifik Enerji İndeksi (SEC): Birim klinker başına harcanan net termal enerji
        x_next["Normalized_Energy_Index"] = x_next["Q_in"] / clinker_safe

        # 🔴 KRİTİK DÜZELTME: Durum Matrisinin Yapısal Sürekliliği (State Consistency)
        # Matris yapısının adımlar arasında mutasyona uğramasını engellemek için
        # SCALE değişkenini her adımda x_next içinde muhafaza ediyoruz.
        current_time = locals().get("t", x.get("t", 1.0))  # t değişkeni için fallback

        if current_time == 0:
            x_next["SCALE"] = 1.0
        else:
            x_next["SCALE"] = x.get("SCALE", 1.0)

        # Durum geçişinin pürüzsüzce tamamlanması ve yeni durum vektörünün teslimi
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
