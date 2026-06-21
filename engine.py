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

        x_next["P_preheater"] = -269.0 + 149.0 * np.exp(-t / 15.0)

        # =====================================================================
        # STATE-SPACE ENTEGRASYONU (Düzleştirilmiş Kütle ve Enerji Korunumu)
        # =====================================================================
        T_ref = 25.0  # [°C]
        R = 8.314  # [J/mol*K]
        dt_sec = self.dt * 3600.0

        # Birim Sistem Çevrimleri
        feed_rate_raw = x_next.get("Feed_rate", 120.0)
        m_solid_feed = (feed_rate_raw * 1000.0) / 3600.0  # [kg/s]

        air_flow_m3h = x_next.get("Air_flow", 65000.0)
        m_air_kiln_inlet = (air_flow_m3h * 1.293) / 3600.0  # [kg/s]

        t_air_flow_m3h = x_next.get("Tertiary_air_flow", 28000.0)
        m_air_tertiary_inlet = (t_air_flow_m3h * 1.293) / 3600.0  # [kg/s]

        cooling_air_m3h = x_next.get("Cooling_air_flow", 150000.0)
        m_cooling_air_total = (cooling_air_m3h * 1.293) / 3600.0  # [kg/s]

        Q_fuel_total_watts = x_next.get("Q_in", 35.0 * 1e6)
        scale_energy = 1e-6

        # Bütçe Dağılımı
        feed_factor_deviation = (feed_rate_raw - 100.0) / 100.0
        raw_frac_burn = self._mpc_smooth_fraction(0.56, -feed_factor_deviation)
        raw_frac_calc = self._mpc_smooth_fraction(0.26, feed_factor_deviation)
        raw_frac_pre = self._mpc_smooth_fraction(0.18, feed_factor_deviation)

        total_frac = raw_frac_burn + raw_frac_calc + raw_frac_pre
        frac_burn = raw_frac_burn / total_frac
        frac_calc = raw_frac_calc / total_frac
        frac_pre = raw_frac_pre / total_frac

        Q_burning_pool = Q_fuel_total_watts * frac_burn
        Q_calcination_budget = Q_fuel_total_watts * frac_calc
        Q_preheater_budget = Q_fuel_total_watts * frac_pre

        # Kalsinasyon Kinetiği
        Ts_calc_curr = x.get("Ts_calcination", 820.0)
        Tg_calc_curr = x.get("Tg_calcination", 950.0)

        Ea_calc = 165000.0
        A_arrhenius = 1.2e6
        Ts_kelvin = Ts_calc_curr + 273.15

        arrhenius_exponent = -Ea_calc / (R * Ts_kelvin)
        r_calc = A_arrhenius * np.exp(arrhenius_exponent)

        X_caoc3_max = 0.38
        m_CO2_evolved = (
            m_solid_feed
            * X_caoc3_max
            * (1.0 / (1.0 + np.exp(-0.05 * (Ts_calc_curr - 750.0))))
            * (1.0 - np.exp(-r_calc * dt_sec))
        )
        m_CO2_evolved = np.clip(m_CO2_evolved, 0.0, m_solid_feed * X_caoc3_max)

        m_solid_to_burning = m_solid_feed - m_CO2_evolved
        m_gas_to_preheater = m_air_kiln_inlet + m_air_tertiary_inlet + m_CO2_evolved

        Q_calc_reaction_enthalpy = m_CO2_evolved * 1780.0 * 1000.0

        # Soğutucu
        Ts_cool_curr = x.get("Ts_Cooling", 1350.0)
        Tg_cooler_last = x.get("Tg_Cooling", 750.0)
        Ts_burn_curr = x.get("Ts_burning", 1420.0)

        Cp_gas, Cp_solid, Cp_wall = 1080.0, 1150.0, 1000.0
        T_secondary = Tg_cooler_last
        T_tertiary = self._mpc_softplus(Tg_cooler_last - 20.0, width=5.0)

        # Burning Zone (Döner Fırın)
        Tg_burn_curr = x.get("Tg_burning", 1550.0)
        Tw_burn_curr = x.get("Tw_burning", 1250.0)

        A_s_burn, A_w_burn, A_ws_burn = 82.0, 70.0, 57.0
        h_gs_burn, h_gw_burn, h_ws_burn = 1450.0, 350.0, 400.0

        C_gas_burn = 250.0 * Cp_gas
        C_solid_burn = 7000.0 * Cp_solid
        C_wall_burn = 16000.0 * Cp_wall

        H_secondary_air = m_air_kiln_inlet * Cp_gas * (T_secondary - T_ref)

        a_gas_burn = (
            (h_gs_burn * A_s_burn)
            + (h_gw_burn * A_w_burn)
            + (m_air_kiln_inlet * Cp_gas)
        ) * scale_energy
        b_gas_burn = (
            Q_burning_pool
            + H_secondary_air
            + (h_gw_burn * A_w_burn * (Tw_burn_curr - T_ref))
            + (h_gs_burn * A_s_burn * (Ts_burn_curr - T_ref))
        ) * scale_energy

        Tg_next_burn = T_ref + (
            (C_gas_burn * scale_energy * (Tg_burn_curr - T_ref) + dt_sec * b_gas_burn)
            / (C_gas_burn * scale_energy + dt_sec * a_gas_burn)
        )

        Q_loss_wall = 12.0 * A_w_burn * (Tw_burn_curr - T_ref)
        Q_wall_to_solid = h_ws_burn * A_ws_burn * (Tw_burn_curr - Ts_burn_curr)
        Tw_next_burn = Tw_burn_curr + (dt_sec / C_wall_burn) * (
            h_gw_burn * A_w_burn * (Tg_next_burn - Tw_burn_curr)
            - Q_loss_wall
            - Q_wall_to_solid
        )

        exo_trigger = self._mpc_softplus(Ts_burn_curr - 1220.0, width=15.0)
        Q_exothermic_burn = (
            m_solid_to_burning * 480000.0 * (exo_trigger / (exo_trigger + 50.0))
        )
        H_solid_from_calc = m_solid_to_burning * Cp_solid * (Ts_calc_curr - T_ref)

        a_sol_burn = (
            (h_gs_burn * A_s_burn)
            + (h_ws_burn * A_ws_burn)
            + (m_solid_to_burning * Cp_solid)
        ) * scale_energy
        b_sol_burn = (
            Q_exothermic_burn
            + H_solid_from_calc
            + (h_gs_burn * A_s_burn * (Tg_next_burn - T_ref))
            + (h_ws_burn * A_ws_burn * (Tw_next_burn - T_ref))
        ) * scale_energy

        Ts_next_burn = T_ref + (
            (C_solid_burn * scale_energy * (Ts_burn_curr - T_ref) + dt_sec * b_sol_burn)
            / (C_solid_burn * scale_energy + dt_sec * a_sol_burn)
        )

        # Calcination Zone
        h_gs_calc, A_s_calc = 340.0, 980.0
        C_solid_calc = 13000.0 * Cp_solid
        C_gas_calc = 150.0 * Cp_gas
        Ts_pre_curr = x.get("Ts_preheater", 620.0)

        H_tertiary_air = m_air_tertiary_inlet * Cp_gas * (T_tertiary - T_ref)
        H_gas_from_burning = m_air_kiln_inlet * Cp_gas * (Tg_next_burn - T_ref)
        H_solid_from_preheater = m_solid_feed * Cp_solid * (Ts_pre_curr - T_ref)

        a_sol_calc = ((h_gs_calc * A_s_calc) + (m_solid_feed * Cp_solid)) * scale_energy
        b_sol_calc = (
            H_solid_from_preheater
            - Q_calc_reaction_enthalpy
            + (h_gs_calc * A_s_calc * (Tg_calc_curr - T_ref))
        ) * scale_energy

        Ts_next_calc = T_ref + (
            (C_solid_calc * scale_energy * (Ts_calc_curr - T_ref) + dt_sec * b_sol_calc)
            / (C_solid_calc * scale_energy + dt_sec * a_sol_calc)
        )

        H_CO2_generated = m_CO2_evolved * Cp_gas * (Ts_next_calc - T_ref)
        Q_total_gas_calc_in = (
            Q_calcination_budget + H_tertiary_air + H_gas_from_burning + H_CO2_generated
        )

        a_gas_calc = (
            (h_gs_calc * A_s_calc)
            + ((m_air_kiln_inlet + m_air_tertiary_inlet + m_CO2_evolved) * Cp_gas)
        ) * scale_energy
        b_gas_calc = (
            Q_total_gas_calc_in + (h_gs_calc * A_s_calc * (Ts_next_calc - T_ref))
        ) * scale_energy

        Tg_next_calc = T_ref + (
            (C_gas_calc * scale_energy * (Tg_calc_curr - T_ref) + dt_sec * b_gas_calc)
            / (C_gas_calc * scale_energy + dt_sec * a_gas_calc)
        )

        # Preheater Zone
        UA_preheater = 56000.0
        C_solid_pre = 4500.0 * Cp_solid

        W_gas_pre = m_gas_to_preheater * Cp_gas
        W_solid_pre = m_solid_feed * Cp_solid

        UA_eff_pre = (UA_preheater * W_gas_pre) / (UA_preheater + W_gas_pre + 1e-5)

        a_pre_sol = (UA_eff_pre + W_solid_pre) * scale_energy
        b_pre_sol = (
            Q_preheater_budget + (UA_eff_pre * (Tg_next_calc - T_ref))
        ) * scale_energy

        Ts_next_pre = T_ref + (
            (C_solid_pre * scale_energy * (Ts_pre_curr - T_ref) + dt_sec * b_pre_sol)
            / (C_solid_pre * scale_energy + dt_sec * a_pre_sol)
        )
        Tg_next_pre = Tg_next_calc - (
            (UA_eff_pre * (Tg_next_calc - Ts_next_pre)) / (W_gas_pre + 1e-5)
        )

        # Cooler Zone (NTU First Principles)
        W_solid_cool = m_solid_to_burning * Cp_solid
        W_gas_cool = m_cooling_air_total * Cp_gas

        UA_cooling = 34000.0
        ntu_cool = UA_cooling / (W_gas_cool + 1e-6)
        epsilon_cool = 1.0 - np.exp(-ntu_cool)

        M_clinker_cooler_inventory = 26000.0
        C_solid_cool_total = M_clinker_cooler_inventory * Cp_solid

        term_accum = C_solid_cool_total / dt_sec
        term_heat_transfer = W_gas_cool * epsilon_cool

        denom_cool = term_accum + W_solid_cool + term_heat_transfer
        num_cool = term_accum * (Ts_cool_curr - T_ref) + W_solid_cool * (
            Ts_next_burn - T_ref
        )

        Ts_next_cool = T_ref + (num_cool / denom_cool)
        Tg_next_cool_out = T_ref + epsilon_cool * (Ts_next_cool - T_ref)

        # Durum Enjeksiyonları (State Injection)
        x_next["Tg_burning"] = Tg_next_burn
        x_next["Ts_burning"] = Ts_next_burn
        x_next["Tw_burning"] = Tw_next_burn
        x_next["Ts_calcination"] = Ts_next_calc
        x_next["Tg_calcination"] = Tg_next_calc
        x_next["Ts_preheater"] = Ts_next_pre
        x_next["Tg_preheater"] = Tg_next_pre
        x_next["Ts_Cooling"] = Ts_next_cool
        x_next["Tg_Cooling"] = Tg_next_cool_out

        m_gas_pre_out = x_next.get("Air_flow", 65000.0) * 0.001293
        Q_in_total = x_next.get("Q_in", 35.0 * 1e6) * (3600.0 / 4184.0) / 1000.0

        x_next["m_gas_pre"] = m_gas_pre_out
        x_next["Q_in_total"] = Q_in_total
        x_next["t"] = t

        x_next["m_CO2_flow_kg_s"] = m_CO2_evolved
        x_next["T_secondary_air_calc"] = T_secondary
        x_next["T_tertiary_air_calc"] = T_tertiary

        # Global Denge & Klinker Verimi
        Q_total_out = m_gas_pre_out * 1150.0 * x_next["Tg_preheater"]
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

        # Kimyasal Reaksiyonlar
        T_calc = x["Ts_calcination"] + 273.15
        k_calc = 1e7 * np.exp(-160000.0 / (8.314 * T_calc)) if T_calc > 873.15 else 0.0
        x_next["CaCO3"] = x["CaCO3"] * np.exp(-k_calc * self.dt)

        dCaCO3 = x["CaCO3"] - x_next["CaCO3"]
        CaO_generated = dCaCO3 * 0.5603
        x_next["dCaO_calcination"] = CaO_generated / self.dt if self.dt > 0 else 0.0

        T_burn = x["Ts_burning"] + 273.15

        if x["SiO2"] <= 1e-6 or T_burn < 1000.0:
            x_next["dC2S"] = 0.0
        else:
            kinetic = 1.0 - np.exp(
                -(50000000.0 * np.exp(-170000.0 / (8.314 * T_burn))) * self.dt
            )
            x_next["dC2S"] = max(
                0.0,
                min(x["SiO2"] / 0.3488, (x["CaO"] + CaO_generated) / 0.6512) * kinetic,
            )

        if x["C2S"] <= 1e-6 or T_burn < 1473.15:
            x_next["dC3S"] = 0.0
        else:
            kinetic = 1.0 - np.exp(
                -(228000000.0 * np.exp(-200000.0 / (8.314 * T_burn))) * self.dt
            )
            x_next["dC3S"] = max(
                0.0,
                min(
                    x["C2S"] / 0.7544,
                    max(0.0, x["CaO"] + CaO_generated - x_next["dC2S"] * 0.6512)
                    / 0.2456,
                )
                * kinetic,
            )

        if x["Al2O3"] <= 1e-6 or T_burn < 1100.0:
            x_next["dC3A"] = 0.0
        else:
            kinetic = 1.0 - np.exp(
                -(100000.0 * np.exp(-120000.0 / (8.314 * T_burn))) * self.dt
            )
            lim_ca = (
                max(
                    0.0,
                    x["CaO"]
                    + CaO_generated
                    - x_next["dC2S"] * 0.6512
                    - x_next["dC3S"] * 0.2456,
                )
                / 0.6227
            )
            x_next["dC3A"] = max(0.0, min(x["Al2O3"] / 0.3773, lim_ca) * kinetic)

        if x["Fe2O3"] <= 1e-6 or T_burn < 1100.0:
            x_next["dC4AF"] = 0.0
        else:
            kinetic = 1.0 - np.exp(
                -(200000.0 * np.exp(-150000.0 / (8.314 * T_burn))) * self.dt
            )
            lim_al = max(0.0, x["Al2O3"] - x_next["dC3A"] * 0.3773) / 0.2098
            lim_ca = (
                max(
                    0.0,
                    x["CaO"]
                    + CaO_generated
                    - x_next["dC2S"] * 0.6512
                    - x_next["dC3S"] * 0.2456
                    - x_next["dC3A"] * 0.6227,
                )
                / 0.4616
            )
            x_next["dC4AF"] = max(
                0.0, min(x["Fe2O3"] / 0.3286, lim_al, lim_ca) * kinetic
            )

        x_next["C2S"] = max(0.0, x["C2S"] + x_next["dC2S"] - (x_next["dC3S"] * 0.7544))
        x_next["C3S"] = x["C3S"] + x_next["dC3S"]
        x_next["C3A"] = x["C3A"] + x_next["dC3A"]
        x_next["C4AF"] = x["C4AF"] + x_next["dC4AF"]

        x_next["SiO2"] = max(0.0, x["SiO2"] - (x_next["dC2S"] * 0.3488))
        x_next["Al2O3"] = max(
            0.0, x["Al2O3"] - (x_next["dC3A"] * 0.3773) - (x_next["dC4AF"] * 0.2098)
        )
        x_next["Fe2O3"] = max(0.0, x["Fe2O3"] - (x_next["dC4AF"] * 0.3286))

        CaO_consumed = (
            (x_next["dC2S"] * 0.6512)
            + (x_next["dC3S"] * 0.2456)
            + (x_next["dC3A"] * 0.6227)
            + (x_next["dC4AF"] * 0.4616)
        )
        x_next["CaO"] = max(0.0, x["CaO"] + CaO_generated - CaO_consumed)
        x_next["CO2"] = 80.0 - (x_next["CaCO3"] + x_next["CaO"])

        x_next["LSF"] = x_next["CaO"] / (
            2.8 * x_next["SiO2"] + 1.2 * x_next["Al2O3"] + 0.65 * x_next["Fe2O3"] + 1e-9
        )
        x_next["Mass_Balance_Error"] = abs(
            (x_next["CaO"] - x["CaO"]) - (CaO_generated - CaO_consumed)
        )

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
    print(
        f"Simülasyon başarıyla tamamlandı. t=0.0 başlangıçlı {len(df_results)} satır veri kaydedildi."
    )


if __name__ == "__main__":
    run_simulation()
