import numpy as np
from numba import njit
from core.state import KilnState

@njit
def _numba_step_core(N, Ts, Tg, Tw, X, m_CaO, m_SiO2, m_C2S, m_Al2O3, m_Fe2O3, m_C3A, m_C4AF,
                     rho_g, dt, dz, v_s, m_dot_s_kg_s, m_dot_g_inlet_kg_s, 
                     cp_s, cp_g, dH_vec, area_gas, q_gs_vec, q_rad_net_vec,
                     eff_mass_node, exchange_area_per_dz,
                     k0_vec, Ea_vec, R_const, T_min_vec,
                     pre_factors, loss_coeff, 
                     clip_min, clip_max, burning_zone_factor,
                     q_rxn_prev):
    
    rates = np.zeros((5, N))
    x_coords = np.linspace(0, 1, N)
    
    # Isı transferi katsayıları
    dynamic_boost = 1.0 + 0.4 / (1.0 + np.exp(-25 * (x_coords - 0.82)))
    rad_multiplier = 1.05
    
    for i in range(N):
        T_eval = max(300.0, Ts[i])
        
        # 1. Kalsinasyon
        if x_coords[i] >= 0.25:
            if T_eval >= T_min_vec[0] and X[i] < 1.0:
                arg = -Ea_vec[0] / (R_const * T_eval)
                rates[0, i] = k0_vec[0] * np.exp(max(clip_min, min(clip_max, arg))) * (1.0 - X[i]) 

        # 2. Belit (Diffusion-Limited)
        if x_coords[i] >= 0.50 and T_eval >= 1050.0:
            cao_sat = 1.0 - np.exp(-m_CaO[i] / 0.05)
            sio2_sat = 1.0 - np.exp(-m_SiO2[i] / 0.03)
            diffusion_limiter = 1.0 - np.exp(-4.0 * (m_C2S[i] + 0.005))
            k_c2s = k0_vec[1] * np.exp(max(clip_min, min(clip_max, -Ea_vec[1]/(R_const*T_eval))))
            rates[1, i] = k_c2s * cao_sat * sio2_sat * diffusion_limiter

        # 3. Alit
        if x_coords[i] >= 0.75 and T_eval >= 1300.0:
            c2s_sat = 1.0 - np.exp(-m_C2S[i] / 0.10)
            cao_sat_c3s = 1.0 - np.exp(-m_CaO[i] / 0.04)
            liquid_sat = 1.0 - np.exp(-(m_C3A[i] + m_C4AF[i]) / 0.15)
            k_c3s = k0_vec[2] * np.exp(max(clip_min, min(clip_max, -Ea_vec[2]/(R_const*T_eval))))
            rates[2, i] = k_c3s * c2s_sat * cao_sat_c3s * liquid_sat * burning_zone_factor

    # Enerji EMA
    q_rxn_instant = (rates[0]*dH_vec[0] + rates[1]*dH_vec[1] + rates[2]*dH_vec[2]) * m_dot_s_kg_s
    q_rxn = 0.85 * q_rxn_prev + 0.15 * q_rxn_instant

    thermal_capacity_s = np.maximum(10.0, eff_mass_node * cp_s) 
    dTs, dTg = np.zeros(N), np.zeros(N)
    
    dTs[1:] = -v_s * (Ts[1:] - Ts[:-1]) / dz
    q_loss_wall = loss_coeff * (Ts - Tw) * exchange_area_per_dz
    rad_amp = np.where(Ts > 1450.0, 1.0 + (Ts - 1450.0) / 400.0, 1.0)
    
    net_heat_to_s = (dynamic_boost * (q_gs_vec + q_rad_net_vec * rad_multiplier * rad_amp)) - q_loss_wall - q_rxn
    dTs += np.clip(net_heat_to_s / thermal_capacity_s, -3000.0, 3000.0) 
    
    m_dot_g_local = np.maximum(0.5, m_dot_g_inlet_kg_s)
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    dTg += np.clip(-(q_gs_vec + q_rad_net_vec) / (m_dot_g_local * cp_g), -6000.0, 6000.0)
            
    return dTs, dTg, rates, q_rxn

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
        self.q_rxn_ema = np.zeros(nodes)
        self.m_CaO_pool = np.zeros(nodes)
        
        # Başlangıç durumu
        self.state.Tg = np.full(nodes, 350.0)
        self.state.Ts = np.full(nodes, 300.0)
        self.state.Tw = np.full(nodes, 300.0)

    def _safe_f(self, v): 
        return float(v[0]) if isinstance(v, list) else float(v)

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        dt_v = min(float(dt), 0.25)
        sim_hour = self.total_sim_time / 3600.0 
        
        # --- HIZLANDIRILMIŞ RAMPA (3 SAAT) ---
        # 1 - exp(-t/tau) formu ilk başta daha agresif ısınma sağlar
        # tau = 1.0 (yaklaşık 3 saatte %95'e ulaşır)
        ramp_factor = 1.0 - np.exp(-sim_hour / 12) 
        
        nodes = self.state.N
        x = np.linspace(0, 1, nodes)
        
        # Hedef Profiller
        target_Tg = 450 + 1800 * np.exp(-(x-1.0)**2 / 0.15)
        target_Tw = 350 + 850 * x 

        # Rampa Uygulaması
        self.state.Tg = 350.0 * (1 - ramp_factor) + target_Tg * ramp_factor
        self.state.Tw = 300.0 * (1 - ramp_factor) + target_Tw * ramp_factor

        # --- ADVECTION ---
        v_s = self.tra.calculate_solid_velocity(kiln_rpm)
        f_red = np.clip(v_s * dt_v / self.dz, 0.0, 0.4)
        params = ['m_CaO', 'm_C2S', 'm_C3S', 'X', 'm_Al2O3', 'm_Fe2O3', 'm_C3A', 'm_C4AF', 'm_SiO2']
        for p in params:
            arr = getattr(self.state, p)
            arr[1:] = (1.0 - f_red) * arr[1:] + f_red * arr[:-1]
        
        self.m_CaO_pool[1:] = (1.0 - f_red) * self.m_CaO_pool[1:] + f_red * self.m_CaO_pool[:-1]
        self.q_rxn_ema[1:] = (1.0 - f_red) * self.q_rxn_ema[1:] + f_red * self.q_rxn_ema[:-1]

        # --- PHYSICS & REACTION ---
        m_dot_s_kg_s = (feed_rate * 1000.0) / 3600.0 
        m_dot_g_inlet_kg_s = self._safe_f(self.cfg['gas']['nominal_flow']) * (fan_rate / 800.0)
        
        fill_degree = self.tra.get_dynamic_filling_degree(kiln_rpm)
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - fill_degree)
        exchange_area_dz = diameter * np.pi * self.dz
        
        h_gs = self.en.calculate_convection_coeff(fan_rate)
        q_gs_vec = h_gs * (self.state.Tg - self.state.Ts) * exchange_area_dz
        q_rad_net_vec = self.en.calculate_radiation_flux(self.state.Tg, self.state.Ts, self.state.Tw, exchange_area_dz)

        dTs, dTg, rates, self.q_rxn_ema = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.Tw,
            self.state.X, self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S,
            self.state.m_Al2O3, self.state.m_Fe2O3, self.state.m_C3A, self.state.m_C4AF,
            self.state.rho_g, dt_v, self.dz, v_s, m_dot_s_kg_s, m_dot_g_inlet_kg_s,
            self._safe_f(self.cfg['material']['cp_s']), self._safe_f(self.cfg['gas']['cp_g']),
            self.kin.get_enthalpy_vector(), area_gas, q_gs_vec, q_rad_net_vec, 
            max(0.5, (m_dot_s_kg_s / max(1e-4, v_s)) * self.dz), exchange_area_dz,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min,
            self.kin.pre_factors, 0.05, -80.0, 20.0, self.kin.burning_zone_factor,
            self.q_rxn_ema
        )

        self.state.Ts += dTs * dt_v
        self.state.Ts = np.clip(self.state.Ts, 300.0, 1850.0)

        # --- REACTION UPDATES ---
        delta_X = rates[0] * dt_v
        self.state.X = np.clip(self.state.X + delta_X, 0.0, 1.0)
        produced_CaO = delta_X * self._safe_f(self.cfg['raw_meal_composition']['CaCO3']) * 0.56 
        self.m_CaO_pool += produced_CaO

        # Tüketim Sınırı
        max_safe_cons = self.state.m_CaO * 0.12
        r_c2s = np.minimum(rates[1] * dt_v, max_safe_cons / 0.65)
        r_c2s = np.minimum(r_c2s, self.state.m_SiO2 / 0.28)
        
        r_c3s = np.minimum(rates[2] * dt_v, (max_safe_cons - r_c2s*0.65) / 0.25)
        r_c3s = np.minimum(r_c3s, self.state.m_C2S / 0.75)

        # Pool release hızı artırıldı (0.4 -> 0.6)
        release_to_state = self.m_CaO_pool * 0.6
        self.state.m_CaO += release_to_state
        self.m_CaO_pool -= release_to_state

        # Kütle Güncellemeleri
        self.state.m_C3S += r_c3s
        self.state.m_C2S += (r_c2s - r_c3s * 0.75) 
        self.state.m_SiO2 = np.maximum(0.0, self.state.m_SiO2 - (r_c2s * 0.28))
        self.state.m_CaO = np.maximum(1e-6, self.state.m_CaO - (r_c2s * 0.65) - (r_c3s * 0.25))
        
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        self.total_sim_time += dt_v
        return dt_v