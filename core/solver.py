import numpy as np
from numba import njit
from core.state import KilnState

@njit
def _numba_step_core(N, Ts, Tg, X, m_CaO, m_SiO2, m_C2S, rho_g, dt, dz, v_s, m_dot_s, m_dot_g_inlet, 
                     cp_s, cp_g, dH_vec, area_gas, exchange_area_per_dz, 
                     damping, effective_thermal_mass, h_gs, sigma, eps_eff,
                     k0_vec, Ea_vec, R_const, T_min_vec):
    """
    Klinker fazlarını ve ekzotermik ısıları içeren gelişmiş termodinamik çekirdek.
    """
    rates = np.zeros((3, N))
    
    # Reaksiyon Maskeleri: Fiziksel limitlerde (hammadde bittiğinde) reaksiyonu durdurur
    mask_calc = (Ts >= T_min_vec[0]) & (X < 1.0)
    mask_c2s  = (Ts >= T_min_vec[1]) & (m_CaO > 0.001) & (m_SiO2 > 0.001)
    mask_c3s  = (Ts >= T_min_vec[2]) & (m_C2S > 0.001) & (m_CaO > 0.001)

    # Arrhenius Hız Denklemleri
    k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R_const * Ts))
    rates[0, mask_calc] = np.minimum(k_calc[mask_calc] * (1.0 - X[mask_calc]), 0.01)

    k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R_const * Ts))
    rates[1, mask_c2s] = np.minimum(k_c2s[mask_c2s] * (m_CaO[mask_c2s]**2) * m_SiO2[mask_c2s], 0.005)

    k_c3s = k0_vec[2] * np.exp(-Ea_vec[2] / (R_const * Ts))
    rates[2, mask_c3s] = np.minimum(k_c3s[mask_c3s] * m_C2S[mask_c3s] * m_CaO[mask_c3s], 0.002)

    # 2. Gaz Fazı Güncelleme (CO2 Katılımı)
    CO2_RATIO = 0.4397 
    m_dot_co2 = rates[0] * m_dot_s * CO2_RATIO
    m_dot_g_local = np.full(N, m_dot_g_inlet)
    
    current_co2_acc = 0.0
    for i in range(N - 1, -1, -1):
        current_co2_acc += m_dot_co2[i]
        m_dot_g_local[i] += current_co2_acc

    # 3. Isı Transferi
    q_conv = h_gs * (Tg - Ts) * exchange_area_per_dz
    q_rad = sigma * eps_eff * (Tg**4 - Ts**4) * exchange_area_per_dz
    q_net = q_conv + q_rad
    
    # 4. Enerji Dengesi
    q_reaction = (rates[0] * dH_vec[0] + rates[1] * dH_vec[1] + rates[2] * dH_vec[2]) * m_dot_s
    
    dTs = np.zeros(N)
    dTg = np.zeros(N)
    
    dTs[1:] = (v_s * (Ts[:-1] - Ts[1:]) / dz)
    dTs += (q_net - q_reaction) / (effective_thermal_mass * cp_s)
    dTs *= damping
    
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    dTg -= q_net / (np.maximum(1.0, m_dot_g_local) * cp_g)
    dTg *= (damping * 0.75)
            
    return dTs, dTg, rates

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
        self.MW_CAO_RATIO = 0.5603 

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
        dt_val = float(dt)
        
        f_rate = fuel_rate if fuel_rate is not None else self._safe_f(self.cfg['gas'].get('fuel_rate', 16.0))
        cur_fan = fan_rate if fan_rate is not None else self.cfg.get('fan_rate_current', 800.0)
        self.cfg['gas']['temp_inlet'] = self._calculate_flame_temp(f_rate, cur_fan)
        
        fan_ratio = cur_fan / 850.0
        m_dot_g_inlet = self._safe_f(self.cfg['gas'].get('nominal_flow', 8.5)) * fan_ratio
        h_gs = self._safe_f(self.cfg['gas'].get('h_gs', 400.0)) * (fan_ratio**0.4)
        
        current_rpm = kiln_rpm if kiln_rpm is not None else self._safe_f(self.cfg['kiln'].get('rpm', 2.0))
        v_s = self.tra.calculate_solid_velocity(current_rpm)
        m_dot_s = feed_rate if feed_rate is not None else self._safe_f(self.cfg['material']['feed_rate'])
        
        effective_thermal_mass = (m_dot_s / max(0.001, v_s)) * self.dz + 150.0
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        exchange_area_per_dz = diameter * self.dz * (self.tra.get_dynamic_filling_degree(current_rpm) / 0.10)
        area_gas = (np.pi * (diameter / 2)**2) * 0.90
        eps_eff = (self._safe_f(self.cfg['gas']['emissivity_g']) + 0.85) / 2.0

        self.state.update_gas_density()
        dH_vec = np.array([self.kin.dH_calc, self.kin.dH_c2s, self.kin.dH_c3s])

        dTs, dTg, rates = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.X, 
            self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S, self.state.rho_g,
            dt_val, self.dz, v_s, m_dot_s, m_dot_g_inlet, 
            self._safe_f(self.cfg['material']['cp_s']), 
            self._safe_f(self.cfg['gas']['cp_g']), 
            dH_vec, area_gas, exchange_area_per_dz, self.damping, 
            effective_thermal_mass, h_gs, 5.67e-8, eps_eff,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min
        )

        # --- DURUM GÜNCELLEME ---
        self.state.X  += rates[0] * dt_val
        self.state.Ts += dTs * dt_val
        self.state.Tg += dTg * dt_val
        
        # --- STOİKİOMETRİK KÜTLE DENGESİ (FINAL REVISION) ---
        c0 = self.cfg['raw_meal_composition']
        CO2_LOSS_FACTOR = 0.44 # CaCO3 kütlesinin uçucu CO2 oranı
        
        # 1. Faz Entegrasyonu
        self.state.m_C2S += rates[1] * dt_val
        self.state.m_C3S += rates[2] * dt_val
        
        # 2. Hammadde Stok Kontrolü ve Yan Fazlar
        self.state.m_CaCO3 = np.maximum(0.0, c0['CaCO3'] * (1.0 - self.state.X))
        
        sinter_mask = self.state.Ts > 1530.0 
        self.state.m_C3A[sinter_mask] = np.minimum(self.state.m_C3A[sinter_mask] + 0.05 * dt_val, c0['Al2O3'] * 2.6)
        self.state.m_C4AF[sinter_mask] = np.minimum(self.state.m_C4AF[sinter_mask] + 0.05 * dt_val, c0['Fe2O3'] * 3.0)

        # 3. Toplam Katı Kütlesi (Grafikteki siyah çizgi zıplamasını düzeltir)
        # Sadece CO2 kaybı kütleyi azaltır, klinkerleşme kütleyi korur.
        self.state.total_mass = 1.0 - (self.state.X * c0['CaCO3'] * CO2_LOSS_FACTOR)
        
        # 4. CaO ve SiO2 Dengesi (Net Tüketim Hesabı)
        produced_CaO = self.state.X * c0['CaCO3'] * self.MW_CAO_RATIO
        consumed_CaO = (self.state.m_C2S * 0.651) + (self.state.m_C3S * 0.737) + \
                       (self.state.m_C3A * 0.623) + (self.state.m_C4AF * 0.461)
        
        self.state.m_CaO = np.maximum(0.0, produced_CaO - consumed_CaO)
        self.state.m_SiO2 = np.maximum(0.0, c0['SiO2'] - (self.state.m_C2S * 0.349) - (self.state.m_C3S * 0.263))

        # --- Fiziksel Taşınım (Flow) ---
        flow_factor = min(1.0, v_s * dt_val / self.dz)
        phases = ['m_SiO2', 'm_CaO', 'm_C2S', 'm_C3S', 'm_C3A', 'm_C4AF', 'X', 'total_mass']
        for p in phases:
            arr = getattr(self.state, p)
            arr[1:] += flow_factor * (arr[:-1] - arr[1:])
            # Klinker fazları için doyum limitleri
            limit = 0.85 if p == 'm_C3S' else 0.4 if p in ['m_C2S', 'm_C3A', 'm_C4AF'] else 1.0
            setattr(self.state, p, np.clip(arr, 0.0, limit))

        # Sınır Koşulları
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        self.state.Tg[-1] = self._safe_f(self.cfg['gas']['temp_inlet'])
        self.state.Ts = np.clip(self.state.Ts, 300.0, self.cfg['solver']['max_temp_s'])
        self.state.Tg = np.clip(self.state.Tg, 300.0, self.cfg['solver']['max_temp_g'])