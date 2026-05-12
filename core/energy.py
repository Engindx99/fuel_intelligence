import numpy as np

class EnergyModel:
    def __init__(self, config):
        # Config hiyerarşisine güvenli erişim
        gas_cfg = config.get("gas", {})
        mat_cfg = config.get("material", {})
        kiln_cfg = config.get("kiln", {})

        # --- Temel Isı Transfer Katsayıları ---
        self.base_h_gs = float(gas_cfg.get("h_gs", 140.0)) #

        # --- Stefan-Boltzmann ve Emisivite ---
        self.sigma = 5.670374419e-8
        self.eps_g = float(gas_cfg.get("emissivity_g", 0.55)) #
        self.eps_s = float(mat_cfg.get("emissivity_s", 0.85)) #
        self.eps_w = float(kiln_cfg.get("emissivity_w", 0.90)) #

        # Efektif emisivite hesaplamaları
        self.eps_gs = np.sqrt(self.eps_g * self.eps_s)
        self.eps_ws = 1.0 / (1.0 / self.eps_s + 1.0 / self.eps_w - 1.0)

        # --- Yakıt Özellikleri ---
        # LHV genellikle kJ/kg cinsindendir
        self.lhv_fuel = float(gas_cfg.get("lhv_fuel", 32000.0)) 

    def calculate_combustion_source(self, fuel_rate_ton_h, nodes):
        """
        Yakıt yanma gücünü hesaplar.
        fuel_rate_ton_h: Config'den gelen ton/saat değeri.
        """
        # Ton/h -> kg/s dönüşümü kritik!
        fuel_rate_kg_s = (fuel_rate_ton_h * 1000.0) / 3600.0
        
        # Power (W) = kg/s * J/kg
        total_power = fuel_rate_kg_s * (self.lhv_fuel * 1e3) #
        
        source_profile = np.zeros(nodes)
        # Brülör bölgesi (son %10)
        burning_zone_nodes = max(1, int(nodes * 0.1))
        source_profile[-burning_zone_nodes:] = total_power / burning_zone_nodes
        
        return source_profile

    def calculate_convective_flux(self, Tg, Ts, area, h_gs):
        """Gaz ve katı arasındaki konvektif ısı transferi."""
        # Isı akışı gazdan katıya doğrudur (Tg > Ts ise pozitif)
        q_conv = h_gs * area * (Tg - Ts)
        return np.nan_to_num(q_conv, nan=0.0)

    def calculate_radiation_flux(self, Tg, Ts, Tw, area):
        """Gaz, katı ve duvar arasındaki radyatif transfer."""
        Tg = np.clip(Tg, 300.0, 2600.0)
        Ts = np.clip(Ts, 300.0, 2600.0)
        Tw = np.clip(Tw, 300.0, 2600.0)

        # Görünürlük faktörleri (varsayılan basitleştirilmiş)
        f_gs = 0.65
        f_ws = 0.75

        q_gs = f_gs * self.eps_gs * self.sigma * area * (Tg**4 - Ts**4)
        q_ws = f_ws * self.eps_ws * self.sigma * area * (Tw**4 - Ts**4)

        return np.nan_to_num(q_gs + q_ws, nan=0.0)

    def get_reaction_heat(self, rates, m_dot_s, dH_vec):
        """
        Reaksiyon entalpisinden kaynaklanan ısı değişimi.
        NOT: dH_vec pozitifse (endotermik), q_rxn pozitiftir ve 
        katı sıcaklığından çıkarılmalıdır.
        """
        # rates: reaksiyon hızı, m_dot_s: katı kütle akışı
        q_rxn = np.sum(rates * dH_vec[:, None], axis=0) * m_dot_s
        return np.nan_to_num(q_rxn, nan=0.0)