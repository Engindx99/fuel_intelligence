import numpy as np
from numba import njit
from core.state import KilnState

@njit
def _numba_step_core(N, Ts, Tg, X, rho_g, dt, dz, v_s, m_dot_s, m_dot_g_inlet, 
                     cp_s, cp_g, dH_rxn, area_gas, exchange_area_per_dz, 
                     damping, effective_thermal_mass, h_gs, sigma, eps_eff,
                     k0, Ea, R_const, T_min):
    """CO2 kütle transferi eklenmiş vektörize termodinamik çekirdek."""
    
    # 1. Kinetik ve CO2 Üretimi
    k = k0 * np.exp(-Ea / (R_const * Ts))
    r = np.zeros(N)
    mask = (Ts >= T_min) & (X < 1.0)
    r[mask] = np.minimum(k[mask] * (1.0 - X[mask]), 0.01)
    
    # Kütle Kaybı Oranı: CO2 / CaCO3 ≈ 44.01 / 100.09 ≈ 0.44
    CO2_RATIO = 0.4397 
    
    # Gaz debisi fırın boyunca sabit değildir; CO2 eklenerek artar.
    # m_dot_g_inlet: Brülörden giren hava. Gaz sağdan sola (N-1 -> 0) akar.
    m_dot_co2 = r * m_dot_s * CO2_RATIO
    m_dot_g_local = np.full(N, m_dot_g_inlet)
    
    # Gaz fazına CO2 katılımı (Sağdan sola kümülatif toplam)
    current_co2_acc = 0.0
    for i in range(N - 1, -1, -1):
        current_co2_acc += m_dot_co2[i]
        m_dot_g_local[i] += current_co2_acc

    # 2. Isı Transferi
    q_conv = h_gs * (Tg - Ts) * exchange_area_per_dz
    q_rad = sigma * eps_eff * (Tg**4 - Ts**4) * exchange_area_per_dz
    q_net = q_conv + q_rad
    
    # 3. Enerji Dengesi Diferansiyelleri
    dTs = np.zeros(N)
    dTg = np.zeros(N)
    
    # Katı Faz Adveksiyon (Upwind: [i-1] - [i])
    dTs[1:] = (v_s * (Ts[:-1] - Ts[1:]) / dz)
    source_s = (q_net - r * m_dot_s * dH_rxn) / (effective_thermal_mass * cp_s)
    dTs += source_s
    dTs *= damping
    
    # Gaz Fazı Adveksiyon (Upwind: [i+1] - [i])
    # Gaz hızı, yerel gaz debisine (m_dot_g_local) bağlıdır
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    # Gazın soğuması yerel kütle akışına bölünür
    dTg -= q_net / (np.maximum(1.0, m_dot_g_local) * cp_g)
    dTg *= (damping * 0.75)
            
    return dTs, dTg, r

class KilnSolver:
    def __init__(self, config, kinetics, transport, energy):
        self.cfg = config
        self.kin = kinetics
        self.tra = transport
        self.en  = energy
        
        nodes = int(self._safe_f(config['kiln']['nodes']))
        self.state = KilnState(nodes)
        self.dz = self._safe_f(config['kiln']['length']) / nodes
        self.damping = 0.15
        
        # Stoikiometrik Katsayılar
        self.MW_CAO_RATIO = 0.5603 # CaO / CaCO3
        self.MW_CO2_RATIO = 0.4397 # CO2 / CaCO3

    def _safe_f(self, value):
        if isinstance(value, list): return float(value[0])
        return float(value)

    def _calculate_flame_temp(self, fuel_rate, fan_rate=None):
        base_temp = 20.0 
        gain_factor = 460.0
        flame_temp = base_temp + (np.sqrt(max(0, fuel_rate)) * gain_factor)
        if fan_rate is not None:
            dilution_factor = (850.0 / max(400, fan_rate))**0.40
            flame_temp *= dilution_factor
        return min(2400.0, flame_temp)

    def solve_step(self, dt, fuel_rate=None, feed_rate=None, kiln_rpm=None, fan_rate=None):
        # Parametre Güncellemeleri
        f_rate = fuel_rate if fuel_rate is not None else self._safe_f(self.cfg['gas'].get('fuel_rate', 2.0))
        cur_fan = fan_rate if fan_rate is not None else self.cfg.get('fan_rate_current', 850.0)
        self.cfg['gas']['temp_inlet'] = self._calculate_flame_temp(f_rate, cur_fan)
        
        fan_ratio = cur_fan / 850.0
        m_dot_g_inlet = self._safe_f(self.cfg['gas'].get('nominal_flow', 5.0)) * fan_ratio
        h_gs = 0.06 * (fan_ratio**0.4)
        
        current_rpm = kiln_rpm if kiln_rpm is not None else self._safe_f(self.cfg['kiln'].get('rpm', 3.0))
        v_s = self.tra.calculate_solid_velocity(current_rpm)
        dynamic_fill = self.tra.get_dynamic_filling_degree(current_rpm) 
        m_dot_s = feed_rate if feed_rate is not None else self._safe_f(self.cfg['material']['feed_rate'])
        
        node_mass = (m_dot_s / max(0.001, v_s)) * self.dz
        effective_thermal_mass = node_mass + 150.0
        
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        exchange_area_per_dz = diameter * self.dz * (dynamic_fill / 0.10)
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - dynamic_fill)
        eps_eff = (self._safe_f(self.cfg['gas']['emissivity_g']) + 0.85) / 2.0

        self.state.update_gas_density()

        # Çekirdek Hesaplama (CO2 kütle transferi dahil)
        dTs, dTg, dX = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.X, self.state.rho_g,
            float(dt), self.dz, v_s, m_dot_s, m_dot_g_inlet, 
            self._safe_f(self.cfg['material']['cp_s']), 
            self._safe_f(self.cfg['gas']['cp_g']), 
            float(self.kin.dH), area_gas, exchange_area_per_dz, self.damping, 
            effective_thermal_mass, h_gs, 5.67e-8, eps_eff,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min
        )

        # Durum Entegrasyonu
        self.state.X  += dX * dt
        self.state.Ts += dTs * dt
        self.state.Tg += dTg * dt
        
        # --- Vektörize Oksit Taşınımı ---
        flow_factor = min(1.0, v_s * dt / self.dz)
        for oxide in ['m_SiO2', 'm_Al2O3', 'm_Fe2O3']:
            arr = getattr(self.state, oxide)
            arr[1:] += flow_factor * (arr[:-1] - arr[1:])

        # Kütle Dönüşümü ve CO2 Kaybı (Vektörize)
        # CaCO3 azalırken katı kütlesinden CO2 çıkar, sadece CaO kalır.
        c0_CaCO3 = 0.78
        self.state.m_CaCO3 = c0_CaCO3 * (1.0 - self.state.X)
        self.state.m_CaO   = (c0_CaCO3 * self.state.X) * self.MW_CAO_RATIO

        # Sınır Koşulları ve Clipping
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        self.state.Tg[-1] = self._safe_f(self.cfg['gas']['temp_inlet'])
        
        self.state.X = np.clip(self.state.X, 0.0, 1.0)
        self.state.Ts = np.clip(self.state.Ts, 300.0, 2000.0)
        self.state.Tg = np.clip(self.state.Tg, 300.0, 2600.0)