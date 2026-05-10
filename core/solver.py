import numpy as np
from numba import njit
from core.state import KilnState

@njit
def get_smooth_step(T, T_min, span=50.0):
    """Sayısal süreksizliği önlemek için yumuşak geçiş fonksiyonu."""
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
        # 1. Kalsinasyon
        if Ts[i] >= T_min_vec[0] and X[i] < 1.0:
            k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R_const * Ts[i]))
            rates[0, i] = min(k_calc * (1.0 - X[i]), 0.05)

        # 2. Belit (C2S) Oluşumu
        f_c2s = get_smooth_step(Ts[i], T_min_vec[1], 50.0)
        if f_c2s > 0 and m_CaO[i] > 1e-4 and m_SiO2[i] > 1e-4:
            k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R_const * Ts[i]))
            rates[1, i] = min(k_c2s * m_CaO[i] * m_SiO2[i], 0.02) * f_c2s

        # 3. Alit (C3S) Oluşumu
        f_c3s = get_smooth_step(Ts[i], T_min_vec[2], 75.0)
        if f_c3s > 0 and m_C2S[i] > 0.01 and m_CaO[i] > 0.01:
            k_c3s = k0_vec[2] * np.exp(-Ea_vec[2] / (R_const * Ts[i]))
            rates[2, i] = k_c3s * m_C2S[i] * m_CaO[i] * f_c3s

        # 4. Sıvı Fazlar
        if Ts[i] >= 1420.0: rates[3, i] = k_flux_c3a * m_Al2O3[i]
        if Ts[i] >= 1380.0: rates[4, i] = k_flux_c4af * m_Fe2O3[i]

    q_rxn = (rates[0]*dH_vec[0] + rates[1]*dH_vec[1] + rates[2]*dH_vec[2] + 
             rates[3]*dH_vec[3] + rates[4]*dH_vec[4]) * m_dot_s_kg_s
    
    dTs, dTg = np.zeros(N), np.zeros(N)
    thermal_capacity_s = eff_mass_node * cp_s
    
    dTs[1:] = -v_s * (Ts[1:] - Ts[:-1]) / dz
    
    # Isıl kayıp katsayısı artık dışarıdan geliyor
    q_loss_wall = loss_coeff * (Ts - Tw) * exchange_area_per_dz
    
    q_net_s = (q_gs_vec + q_rad_net_vec - q_loss_wall - q_rxn)
    dTs += q_net_s / thermal_capacity_s
    
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
        self.warmup_duration = 10800.0 
        self.max_dt_temp_change = 1.0
        self._log_timer = 0
        
        # --- Cooldown State ---
        self.cooldown_timer = 0.0

    def _safe_f(self, v): 
        return float(v[0]) if isinstance(v, list) else float(v)

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        dt_v = float(dt)
        self.total_sim_time += dt_v
        
        fuel_kg_s = (fuel_rate * 1000.0) / 3600.0
        m_dot_s_kg_s = (feed_rate * 1000.0) / 3600.0 
        m_dot_g_inlet_kg_s = self._safe_f(self.cfg['gas']['nominal_flow']) * (fan_rate / 800.0)
        
        # 1. Dinamik Rampa ve Cooldown (Sönümlenme)
        base_rampa = 1.0 - np.exp(-0.53 * self.total_sim_time / self.warmup_duration)
        
        if fuel_kg_s < 0.01: # Yakıt kesildiyse
            self.cooldown_timer += dt_v
            # 3 saatlik (10800s) sönümlenme eğrisi
            decay = np.exp(-self.cooldown_timer / 10800.0)
            rampa = base_rampa * decay
            loss_coeff = 35.0  # Soğuma fazında kabuk kayıplarını artırıyoruz
        else:
            self.cooldown_timer = 0.0
            rampa = base_rampa
            loss_coeff = 12.0  # Normal çalışma şartı kayıpları
            
        rampa = max(0.05, rampa)

        v_s = self.tra.calculate_solid_velocity(kiln_rpm)
        fill_degree = self.tra.get_dynamic_filling_degree(kiln_rpm)
        
        inertia_factor = 0.51
        eff_mass_node = ((m_dot_s_kg_s / max(1e-4, v_s)) * self.dz) * inertia_factor
        
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - fill_degree)
        exchange_area_dz = diameter * np.pi * self.dz
        efektif_temas_alani = exchange_area_dz * (fill_degree * 2.0)

        # 2. Isı Akıları (Rampa/Decay Uygulanmış)
        h_gs = self.en.calculate_convection_coeff(fan_rate) * rampa
        q_gs_vec = h_gs * (self.state.Tg - self.state.Ts) * efektif_temas_alani
        
        q_rad_net_vec = self.en.calculate_radiation_flux(
            self.state.Tg, self.state.Ts, self.state.Tw, efektif_temas_alani
        ) * rampa

        # 3. Taşınım
        flow_factor = np.clip(v_s * dt_v / self.dz, 0.0, 1.0)
        params = ['m_SiO2', 'm_Al2O3', 'm_Fe2O3', 'm_CaO', 'm_C2S', 'm_C3S', 'm_C3A', 'm_C4AF', 'X']
        for p in params:
            arr = getattr(self.state, p)
            arr[1:] = (1.0 - flow_factor) * arr[1:] + flow_factor * arr[:-1]

        # 4. Numba Çekirdek Çözücü
        dTs, dTg, rates = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.Tw,
            self.state.X, self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S,
            self.state.m_Al2O3, self.state.m_Fe2O3, self.state.m_C3A, self.state.m_C4AF,
            self.state.rho_g, dt_v, self.dz, v_s, m_dot_s_kg_s, m_dot_g_inlet_kg_s,
            self._safe_f(self.cfg['material']['cp_s']), self._safe_f(self.cfg['gas']['cp_g']),
            np.array([self.kin.dH_calc, self.kin.dH_c2s, self.kin.dH_c3s, -125.0, -100.0]), 
            area_gas, q_gs_vec, q_rad_net_vec, eff_mass_node, exchange_area_dz,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min, 0.0015, 0.0010, loss_coeff
        )

        max_jump = np.max(np.abs(dTs * dt_v))
        actual_dt = dt_v * min(1.0, self.max_dt_temp_change / (max_jump + 1e-6))

        self.state.Ts += dTs * actual_dt
        self.state.Tg += dTg * actual_dt
        self.state.X += rates[0] * actual_dt
        self.state.m_C3A += rates[3] * actual_dt
        self.state.m_C4AF += rates[4] * actual_dt
        self.state.m_C3S += rates[2] * actual_dt
        self.state.m_C2S += (rates[1] - rates[2]*0.75) * actual_dt
        self.state.m_CaO += (rates[0]*0.56 - rates[1]*0.65 - rates[2]*0.24) * actual_dt
        
        # 5. Alev Şartları (Yakıt Kesintisi Duyarlı)
        lhv = self._safe_f(self.cfg['gas'].get('lhv_fuel', 32000.0))
        t_flame_base = 1200.0 + (fuel_kg_s * lhv * rampa) / (max(0.1, m_dot_g_inlet_kg_s) * self._safe_f(self.cfg['gas']['cp_g']) / 1000.0)
        t_flame_max = min(t_flame_base, 2350.0)
        
        flame_nodes = 12
        for i in range(flame_nodes):
            idx = (self.state.N - 1) - i
            weight = (flame_nodes - i) / flame_nodes
            tg_min_dynamic = 400.0 + (700.0 * rampa)
            self.state.Tg[idx] = max(self.state.Tg[idx], tg_min_dynamic + (t_flame_max - tg_min_dynamic) * weight)

        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])

        # Loglama
        self._log_timer += dt_v
        if self._log_timer >= 600:
            status = "COOLDOWN" if fuel_kg_s < 0.01 else "HEATING"
            print(f"[{status}] Rampa: %{rampa*100:.1f} | Ts_max: {np.max(self.state.Ts):.1f}K")
            self._log_timer = 0