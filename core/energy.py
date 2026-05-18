import numpy as np

class EnergyModel:
    def __init__(self, config):

        gas_cfg = config.get("gas", {})
        mat_cfg = config.get("material", {})
        kiln_cfg = config.get("kiln", {})

        # =====================================================
        # CONVECTION (biraz daha fiziksel gerçekçi)
        # =====================================================
        self.base_h_gs = float(gas_cfg.get("h_gs", 220.0))  # ↑ 140 → 220

        # =====================================================
        # RADIATION CONSTANTS
        # =====================================================
        self.sigma = 5.670374419e-8

        self.eps_g = float(gas_cfg.get("emissivity_g", 0.6))
        self.eps_s = float(mat_cfg.get("emissivity_s", 0.85))
        self.eps_w = float(kiln_cfg.get("emissivity_w", 0.9))

        self.eps_gs = np.sqrt(self.eps_g * self.eps_s)
        self.eps_ws = 1.0 / (1.0 / self.eps_s + 1.0 / self.eps_w - 1.0)

        # =====================================================
        # FUEL
        # =====================================================
        self.lhv_fuel = float(gas_cfg.get("lhv_fuel", 32000.0))

    def calculate_combustion_source(self, fuel_rate_ton_h, nodes):

        fuel_rate_kg_s = (fuel_rate_ton_h * 1000.0) / 3600.0

        total_power = fuel_rate_kg_s * (self.lhv_fuel * 1e3)

        source_profile = np.zeros(nodes)

        # =====================================================
        # FIX 1: smoother burner distribution (Gaussian-like)
        # =====================================================
        burner_center = int(nodes * 0.92)
        width = max(2, int(nodes * 0.08))

        idx = np.arange(nodes)

        gaussian = np.exp(-0.5 * ((idx - burner_center) / width) ** 2)
        gaussian /= np.sum(gaussian) + 1e-12

        source_profile = gaussian * total_power

        return source_profile

    def calculate_convective_flux(self, Tg, Ts, area, h_gs):

        # =====================================================
        # FIX 2: nonlinear enhancement (high-temp boost)
        # =====================================================
        deltaT = Tg - Ts

        h_eff = h_gs * (1.0 + 0.0005 * np.maximum(Tg - 800.0, 0.0))

        q_conv = h_eff * area * deltaT

        return np.nan_to_num(q_conv, nan=0.0)

    def calculate_radiation_flux(self, Tg, Ts, Tw, area):

        Tg = np.clip(Tg, 300.0, 2600.0)
        Ts = np.clip(Ts, 300.0, 2600.0)
        Tw = np.clip(Tw, 300.0, 2600.0)

        f_gs = 0.75   # ↑ 0.65
        f_ws = 0.85   # ↑ 0.75

        q_gs = f_gs * self.eps_gs * self.sigma * area * (Tg**4 - Ts**4)
        q_ws = f_ws * self.eps_ws * self.sigma * area * (Tw**4 - Ts**4)

        return np.nan_to_num(q_gs + q_ws, nan=0.0)

    def get_reaction_heat(self, rates, m_dot_s, dH_vec):

        # FIX 3: sign-safe formulation
        q_rxn = np.sum(rates * dH_vec[:, None], axis=0) * m_dot_s

        return np.nan_to_num(q_rxn, nan=0.0)