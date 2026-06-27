import math
import numpy as np


class Burning:

    def __init__(self):

        # ======================================================
        # FUEL PROPERTIES
        # ======================================================
        self.LHV_petcoke = 32000.0  # kJ/kg
        self.LHV_lignite = 18000.0
        self.LHV_alt = 25000.0

        # ======================================================
        # COMBUSTION MODEL PARAMETERS
        # ======================================================
        self.O2_opt = 3.5
        self.O2_sigma2 = 25.0

        # ======================================================
        # HEAT TRANSFER PARAMETERS
        # ======================================================
        self.radiation_coeff = 5e-8
        self.heat_transfer_coeff = 25.0

        # ======================================================
        # REACTION ENTHALPIES
        # ======================================================
        self.heat_c3s = -500.0
        self.heat_c2s = -400.0
        self.heat_c3a = -350.0
        self.heat_c4af = -300.0

        self.calcination_reversal = 178.0

    # ======================================================
    # COMBUSTION EFFICIENCY MODEL (FIXED)
    # ======================================================
    def combustion_efficiency(self, O2):

        return np.exp(-((O2 - self.O2_opt) ** 2) / self.O2_sigma2)

    # ======================================================
    # MAIN OPERATOR
    # ======================================================
    def apply(self, state, inputs, dt):

        # ======================================================
        # 0. FUEL MIX (LINEAR + NORMALIZED)
        # ======================================================
        petcoke = inputs.get("Petcoke", getattr(state, "Petcoke", 0.6))
        alt_fuel = inputs.get(
            "Alternative_Fuel", getattr(state, "Alternative_Fuel", 0.2)
        )

        lignite = 1.0 - petcoke - alt_fuel
        lignite = max(lignite, 0.0)

        total = petcoke + alt_fuel + lignite

        if total > 0:
            petcoke /= total
            alt_fuel /= total
            lignite /= total

        state.Petcoke = petcoke
        state.Alternative_Fuel = alt_fuel
        state.Lignite_Coal = lignite

        # ======================================================
        # 1. COMBUSTION HEAT RELEASE (FIXED + O2 COUPLED)
        # ======================================================
        fuel_rate = inputs.get("Fuel_rate", state.Fuel_rate)

        O2 = getattr(state, "O2", 3.5)
        comb_eff = self.combustion_efficiency(O2)

        LHV_mix = (
            petcoke * self.LHV_petcoke
            + lignite * self.LHV_lignite
            + alt_fuel * self.LHV_alt
        )

        Q_comb = fuel_rate * LHV_mix * comb_eff  # kJ/h (consistent rate form)

        # ✔ NO DOUBLE COUNTING: energy only tracked via enthalpy balance
        state.Tg_burning += 0.01 * Q_comb * dt

        # ======================================================
        # 2. RADIATION (gas → solid)
        # ======================================================
        T_g = state.Tg_burning + 273.15
        T_s = state.Ts_burning + 273.15

        Q_rad = self.radiation_coeff * (T_g**4 - T_s**4)

        state.Ts_burning += 0.00001 * Q_rad * dt
        state.Tg_burning -= 0.000005 * Q_rad * dt

        # ======================================================
        # 3. CONVECTION
        # ======================================================
        Q_conv = self.heat_transfer_coeff * (state.Tg_burning - state.Ts_burning)

        state.Ts_burning += 0.001 * Q_conv * dt
        state.Tg_burning -= 0.0005 * Q_conv * dt

        # ======================================================
        # 4. CLINKER FORMATION (EXOTHERMIC)
        # ======================================================
        dC3S = 0.01 * state.CaO * dt
        dC2S = 0.008 * state.CaO * dt
        dC3A = 0.005 * state.Al2O3 * dt
        dC4AF = 0.003 * state.Fe2O3 * dt

        state.C3S += dC3S
        state.C2S += dC2S
        state.C3A += dC3A
        state.C4AF += dC4AF

        Q_rxn = (
            dC3S * self.heat_c3s
            + dC2S * self.heat_c2s
            + dC3A * self.heat_c3a
            + dC4AF * self.heat_c4af
        )

        state.Ts_burning += 0.0001 * Q_rxn

        # ======================================================
        # 5. PARTIAL CALCINATION (ENDOTHERMIC)
        # ======================================================
        CaCO3_consumed = 0.002 * state.CaCO3 * dt

        state.CaCO3 -= CaCO3_consumed
        state.CaO += CaCO3_consumed

        Q_calc = CaCO3_consumed * self.calcination_reversal

        state.Ts_burning -= 0.0001 * Q_calc

        # ======================================================
        # 6. WALL LOSS
        # ======================================================
        Q_loss = 0.02 * (state.Ts_burning - 25.0)

        state.Ts_burning -= 0.001 * Q_loss * dt

        # ======================================================
        # 7. ENERGY CONSISTENCY (NO DOUBLE COUNT FIX)
        # ======================================================
        # NOTE:
        # Fuel energy artık burada tekrar eklenmiyor
        # çünkü Q_comb zaten enthalpy source olarak kullanıldı

        state.Fuel_energy += Q_comb * dt - Q_rad * dt - Q_conv * dt - Q_loss * dt

        # ======================================================
        # DEBUG OUTPUT (REAL-TIME MONITORING)
        # ======================================================
        print(
            f"[BURNING] Tg_burning = {state.Tg_burning:.2f} °C | "
            f"Ts_burning = {state.Ts_burning:.2f} °C"
        )

        return state


if __name__ == "__main__":

    from Kiln.Burning import Burning  # veya relative import

    model = Burning()
    from Kiln.GlobalState import GlobalState

    state = GlobalState()

    inputs = {"Fuel_rate": 3.0, "Petcoke": 0.6, "Alternative_Fuel": 0.2}

    for i in range(10):
        state = model.apply(state, inputs, 0.1)
