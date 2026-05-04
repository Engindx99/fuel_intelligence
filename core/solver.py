import numpy as np
from numba import njit
from core.state import KilnState

@njit
def _numba_step_core(N, Ts, Tg, X, rho_g, dt, dz, v_s, m_dot_s, m_dot_g, 
                     cp_s, cp_g, dH_rxn, area_gas, exchange_area_per_dz, 
                     damping, effective_thermal_mass, h_gs, sigma, eps_eff,
                     k0, Ea, R_const, T_min):
    """Yüksek performanslı termodinamik çekirdek."""
    dTs = np.zeros(N)
    dTg = np.zeros(N)
    dX = np.zeros(N)

    for i in range(N):
        # 1. Kinetik (Kalsinasyon)
        r = 0.0
        if Ts[i] >= T_min and X[i] < 1.0:
            k = k0 * np.exp(-Ea / (R_const * Ts[i]))
            r = min(k * (1.0 - X[i]), 0.01)
        dX[i] = r
        
        # 2. Isı Transferi
        q_conv = h_gs * (Tg[i] - Ts[i]) * exchange_area_per_dz
        q_rad = sigma * eps_eff * (Tg[i]**4 - Ts[i]**4) * exchange_area_per_dz
        q_net = q_conv + q_rad
        
        # 3. Katı Faz Enerji Dengesi (Ağırlaştırılmış kütle)
        if i > 0:
            advection_s = v_s * (Ts[i-1] - Ts[i]) / dz
            source_s = (q_net - r * m_dot_s * dH_rxn) / (effective_thermal_mass * cp_s)
            dTs[i] = (advection_s + source_s) * damping
            
        # 4. Gaz Fazı Enerji Dengesi (Soğutma Duyarlı)
        if i < N - 1:
            v_g = m_dot_g / (max(0.1, rho_g[i]) * area_gas)
            advection_g = v_g * (Tg[i+1] - Tg[i]) / dz
            
            # Fan debisi (m_dot_g) arttıkça payda büyür, dış ısı gazı daha az ısıtır
            # Soğuk hava adveksiyonu (advection_g) baskın hale gelir
            dTg[i] = (advection_g - q_net / (max(1.0, m_dot_g) * cp_g)) * (damping * 0.75)
            
    return dTs, dTg, dX

class KilnSolver:
    def __init__(self, config, kinetics, transport, energy):
        self.cfg = config
        self.kin = kinetics
        self.tra = transport
        self.en  = energy
        
        nodes = int(self._safe_f(config['kiln']['nodes']))
        self.state = KilnState(nodes)
        self.dz = self._safe_f(config['kiln']['length']) / nodes
        self.damping = 0.15 # Atalet için biraz daha düşürüldü

    def _safe_f(self, value):
        if isinstance(value, list): return float(value[0])
        return float(value)

    def _calculate_flame_temp(self, fuel_rate, fan_rate=None):
        """Fan etkisi (seyreltme) güçlendirilmiş alev sıcaklığı."""
        base_temp = 20 
        gain_factor = 460
        flame_temp = base_temp + (np.sqrt(max(0, fuel_rate)) * gain_factor)
        
        if fan_rate is not None:
            nominal_fan = 850.0
            # Üs 0.15'ten 0.40'a çıkarıldı: 1200 RPM'de soğuma etkisi artık belirgin
            dilution_factor = (nominal_fan / max(400, fan_rate))**0.40
            flame_temp *= dilution_factor
            
        return min(2400.0, flame_temp)

    def solve_step(self, dt, fuel_rate=None, feed_rate=None, kiln_rpm=None, fan_rate=None):
        """Fiziksel tutarlılığı artırılmış solver adımı."""
        
        # 1. Girişler ve Alev Sıcaklığı
        if fan_rate is not None:
            self.cfg['fan_rate_current'] = fan_rate
        
        f_rate = fuel_rate if fuel_rate is not None else self._safe_f(self.cfg['gas'].get('fuel_rate', 2.0))
        cur_fan = self.cfg.get('fan_rate_current', 850.0)
        
        # Gaz giriş sıcaklığı artık fan hızına çok daha duyarlı
        self.cfg['gas']['temp_inlet'] = self._calculate_flame_temp(f_rate, cur_fan)
        
        # 2. Gaz Akış ve Konveksiyon Kalibrasyonu
        if fan_rate is not None:
            fan_ratio = fan_rate / 850.0
            base_flow = self._safe_f(self.cfg['gas'].get('nominal_flow', 5.0))
            self.cfg['gas']['flow_rate'] = base_flow * fan_ratio
            # h_gs artış hızı azaltıldı: Fan soğuk havayı getirmeli, gazı katıyla fazla ısıtmamalı
            self.cfg['gas']['h_gs'] = 0.06 * (fan_ratio**0.4)

        # 3. Parametreler ve Isıl Kütle (Atalet: 150)
        current_rpm = self._safe_f(self.cfg['kiln'].get('rpm', 3.0))
        v_s = self.tra.calculate_solid_velocity(current_rpm)
        dynamic_fill = self.tra.get_dynamic_filling_degree(current_rpm) 
        
        m_dot_s = self._safe_f(self.cfg['material']['feed_rate'])
        m_dot_g = self._safe_f(self.cfg['gas']['flow_rate'])
        
        node_mass = (m_dot_s / max(0.001, v_s)) * self.dz
        effective_thermal_mass = node_mass + 150.0 # Yüksek ısıl atalet korundu

        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        exchange_area_per_dz = diameter * self.dz * (dynamic_fill / 0.10)
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - dynamic_fill)
        
        h_gs = self._safe_f(self.cfg['gas']['h_gs'])
        eps_eff = (self._safe_f(self.cfg['gas']['emissivity_g']) + 0.85) / 2.0

        self.state.update_gas_density()

        # 4. Çekirdek Hesaplama
        dTs, dTg, dX = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.X, self.state.rho_g,
            float(dt), self.dz, v_s, m_dot_s, m_dot_g, 
            self._safe_f(self.cfg['material']['cp_s']), 
            self._safe_f(self.cfg['gas']['cp_g']), 
            float(self.kin.dH), area_gas, exchange_area_per_dz, self.damping, 
            effective_thermal_mass, h_gs, 5.67e-8, eps_eff,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min
        )

        # 5. Entegrasyon
        self.state.X  += dX * dt
        self.state.Ts += dTs * dt
        self.state.Tg += dTg * dt
        
        # Sınır Koşulları
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        self.state.Tg[-1] = self._safe_f(self.cfg['gas']['temp_inlet'])
        
        self.state.X = np.clip(self.state.X, 0.0, 1.0)
        self.state.Ts = np.clip(self.state.Ts, 300.0, 1950.0)
        self.state.Tg = np.clip(self.state.Tg, 300.0, 2600.0)