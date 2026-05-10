import numpy as np
from numba import njit
from core.state import KilnState

@njit
def get_smooth_step(T, T_min, span=50.0):
    """Sayısal süreksizliği ve sert geçişleri engellemek için düzleştirme fonksiyonu."""
    val = (T - T_min) / span
    if val < 0: return 0.0
    if val > 1: return 1.0
    return val

@njit
def _numba_step_core(N, Ts, Tg, Tw, X, m_CaO, m_SiO2, m_C2S, m_Al2O3, m_Fe2O3, m_C3A, m_C4AF,
                     rho_g, dt, dz, v_s, m_dot_s_kg_s, m_dot_g_inlet_kg_s, 
                     cp_s, cp_g, dH_vec, area_gas, q_gs_vec, q_rad_net_vec,
                     eff_mass_node, exchange_area_per_dz,
                     k0_vec, Ea_vec, R_const, T_min_vec,
                     k_flux_c3a, k_flux_c4af, loss_coeff):
    
    rates = np.zeros((5, N))
    
    for i in range(N):
        # 1. Kalsinasyon (Kütle kaybının kaynağı) - Sayısal yumuşatma eklendi
        f_calc = get_smooth_step(Ts[i], T_min_vec[0], 40.0)
        if f_calc > 0 and X[i] < 1.0:
            k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R_const * Ts[i]))
            rates[0, i] = min(k_calc * (1.0 - X[i]) * f_calc, 0.010)

        # 2. Belit (C2S) Oluşumu
        f_c2s = get_smooth_step(Ts[i], T_min_vec[1], 50.0)
        if f_c2s > 0 and m_CaO[i] > 1e-6 and m_SiO2[i] > 1e-6:
            k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R_const * Ts[i]))
            rates[1, i] = min(k_c2s * m_CaO[i] * m_SiO2[i], 0.025) * f_c2s

        # 3. Alit (C3S) Oluşumu
        f_c3s = get_smooth_step(Ts[i], T_min_vec[2], 75.0)
        c2s_safety_margin = get_smooth_step(m_C2S[i], 0.15, span=0.01)
        liquid_factor = 1.0 + 10.0 * (m_C3A[i] + m_C4AF[i])
        
        if f_c3s > 0 and m_C2S[i] > 1e-6 and m_CaO[i] > 1e-6:
            k_c3s = k0_vec[2] * np.exp(-Ea_vec[2] / (R_const * Ts[i]))
            rates[2, i] = k_c3s * m_C2S[i] * m_CaO[i] * f_c3s * c2s_safety_margin * liquid_factor

        # 4. Sıvı Fazlar (Flux)
        if Ts[i] >= 1400.0 and m_Al2O3[i] > 1e-6: 
            rates[3, i] = k_flux_c3a * m_Al2O3[i]
        if Ts[i] >= 1350.0 and m_Fe2O3[i] > 1e-6: 
            rates[4, i] = k_flux_c4af * m_Fe2O3[i]

    # Enerji Dengesi
    q_rxn = (rates[0]*dH_vec[0] + rates[1]*dH_vec[1] + rates[2]*dH_vec[2] + 
             rates[3]*dH_vec[3] + rates[4]*dH_vec[4]) * m_dot_s_kg_s
    
    dTs, dTg = np.zeros(N), np.zeros(N)
    thermal_capacity_s = eff_mass_node * cp_s
    
    dTs[1:] = -v_s * (Ts[1:] - Ts[:-1]) / dz
    q_loss_wall = loss_coeff * (Ts - Tw) * exchange_area_per_dz
    dTs += (q_gs_vec + q_rad_net_vec - q_loss_wall - q_rxn) / thermal_capacity_s
    
    m_dot_g_local = np.full(N, m_dot_g_inlet_kg_s)
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    dTg -= (q_gs_vec + q_rad_net_vec) / (np.maximum(1.0, m_dot_g_local) * cp_g)
            
    return dTs, dTg, rates

class KilnSolver:
    def __init__(self, config, kinetics, transport, energy):
        self.cfg = config
        self.kin = kinetics
        self.tra = transport
        self.en = energy
        nodes = int(self._safe_f(config['kiln']['nodes']))
        self.state = KilnState(nodes)
        self.dz = self._safe_f(config['kiln']['length']) / nodes
        self.total_sim_time = 0.0
        # Daha sıkı sıcaklık değişim limiti (Osilasyon önleme)
        self.max_dt_temp_change = 3.0 
        self._log_timer = 0

    def _safe_f(self, v): 
        return float(v[0]) if isinstance(v, list) else float(v)

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        dt_v = float(dt)
        self.total_sim_time += dt_v
        
        fuel_kg_s = (fuel_rate * 1000.0) / 3600.0
        m_dot_s_kg_s = (feed_rate * 1000.0) / 3600.0 
        m_dot_g_inlet_kg_s = self._safe_f(self.cfg['gas']['nominal_flow']) * (fan_rate / 800.0)
        
        # --- Dinamik Taşınım Parametreleri ---
        v_s = self.tra.calculate_solid_velocity(kiln_rpm)
        fill_degree = self.tra.get_dynamic_filling_degree(kiln_rpm)
        
        # Termal Atalet (Hıza bağlı kütle birikimi)
        eff_mass_node = ((m_dot_s_kg_s / max(1e-4, v_s)) * self.dz)
        
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - fill_degree)
        exchange_area_dz = diameter * np.pi * self.dz

        # Sönümleme ve Isı Transfer Katsayıları
        z_coords = np.linspace(0, self.cfg['kiln']['length'], self.state.N)
        dist_damp = 0.2 + 0.8 * np.clip(z_coords / 12.0, 0.0, 1.0)
        
        rampa = max(0.05, 1.0 - np.exp(-1.34 * self.total_sim_time / 86400.0))
        h_gs = self.en.calculate_convection_coeff(fan_rate) * rampa * dist_damp
        
        # Isı transferinde fill_degree entegrasyonu
        q_gs_vec = h_gs * (self.state.Tg - self.state.Ts) * (exchange_area_dz * fill_degree * 2.0)
        q_rad_net_vec = self.en.calculate_radiation_flux(self.state.Tg, self.state.Ts, self.state.Tw, exchange_area_dz) * rampa * dist_damp

        # Taşınım (Upwind Advection) - v_s ile tam bağlı
        f_red = np.clip(v_s * dt_v / self.dz, 0.0, 1.0) * 0.5 
        params = ['m_SiO2', 'm_SiO2_locked', 'm_Al2O3', 'm_Fe2O3', 'm_CaO', 'm_C2S', 'm_C3S', 'm_C3A', 'm_C4AF', 'X', 'total_solid_mass', 'm_co2_released']
        for p in params:
            if hasattr(self.state, p):
                arr = getattr(self.state, p)
                arr[1:] = (1.0 - f_red) * arr[1:] + f_red * arr[:-1]

        # Silica Unlocking
        unlock_mask = (self.state.m_SiO2 < 0.002) & (self.state.m_SiO2_locked > 0)
        if np.any(unlock_mask):
            self.state.m_SiO2[unlock_mask] += self.state.m_SiO2_locked[unlock_mask]
            self.state.m_SiO2_locked[unlock_mask] = 0.0

        # Numba Core Çağrısı
        dTs, dTg, rates = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.Tw,
            self.state.X, self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S,
            self.state.m_Al2O3, self.state.m_Fe2O3, self.state.m_C3A, self.state.m_C4AF,
            self.state.rho_g, dt_v, self.dz, v_s, m_dot_s_kg_s, m_dot_g_inlet_kg_s,
            self._safe_f(self.cfg['material']['cp_s']), self._safe_f(self.cfg['gas']['cp_g']),
            np.array([self.kin.dH_calc, self.kin.dH_c2s, self.kin.dH_c3s, -125.0, -100.0]), 
            area_gas, q_gs_vec, q_rad_net_vec, eff_mass_node, exchange_area_dz,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min, 0.0015, 0.0010, 12.0
        )

        # Reaksiyon hızı ve sıcaklık değişimine göre adaptif zaman adımı kontrolü
        max_conv_change = 0.01 # Bir adımda maksimum %1 reaksiyon değişimi
        actual_dt_reac = dt_v * min(1.0, max_conv_change / (np.max(np.abs(rates[0] * dt_v)) + 1e-6))
        actual_dt_temp = dt_v * min(1.0, self.max_dt_temp_change / (np.max(np.abs(dTs * dt_v)) + 1e-6))
        actual_dt = min(actual_dt_reac, actual_dt_temp)

        # --- GÜNCELLEME VE KÜTLE KAYBI ENTEGRASYONU ---
        co2_fraction = 0.44  
        delta_co2 = (rates[0] * co2_fraction) * actual_dt
        self.state.m_co2_released += delta_co2
        self.state.total_solid_mass -= delta_co2 

        self.state.Ts += dTs * actual_dt
        self.state.Tg += dTg * actual_dt
        self.state.X  += rates[0] * actual_dt
        self.state.m_C3S += rates[2] * actual_dt
        self.state.m_C2S += (rates[1] - rates[2] * 0.744) * actual_dt
        self.state.m_SiO2 -= (rates[1] * 0.349) * actual_dt
        self.state.m_CaO  += (rates[0] * 0.56 - rates[1] * 0.651 - rates[2] * 0.246) * actual_dt
        self.state.m_C3A  += rates[3] * actual_dt
        self.state.m_C4AF += rates[4] * actual_dt
        self.state.m_Al2O3 -= rates[3] * actual_dt
        self.state.m_Fe2O3 -= rates[4] * actual_dt

        # Negatif değer koruması
        for s in ['m_SiO2', 'm_CaO', 'm_C2S', 'm_C3S', 'X', 'm_Al2O3', 'm_Fe2O3', 'total_solid_mass']:
            if hasattr(self.state, s):
                setattr(self.state, s, np.maximum(getattr(self.state, s), 0.0))

        # Alev Modeli ve Sınır Şartları
        lhv = self._safe_f(self.cfg['gas'].get('lhv_fuel', 32000.0))
        t_flame = 1200.0 + (fuel_kg_s * lhv * rampa) / (max(0.1, m_dot_g_inlet_kg_s) * self._safe_f(self.cfg['gas']['cp_g']) / 1000.0)
        t_flame = min(t_flame, 2350.0)
        for i in range(12):
            idx = (self.state.N - 1) - i
            self.state.Tg[idx] = max(self.state.Tg[idx], 400.0 + (t_flame - 400.0) * ((12-i)/12))

        self.state.Tg[0] = min(self.state.Tg[0], 900.0)
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])

        # Loglama
        self._log_timer += dt_v
        if self._log_timer >= 600:
            avg_mass = np.mean(self.state.total_solid_mass)
            print(f"[MASS] Avg Total Mass: {avg_mass:.4f} | Released CO2: {np.max(self.state.m_co2_released):.4f}")
            self._log_timer = 0

        return actual_dt