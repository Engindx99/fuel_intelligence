import numpy as np
from numba import njit
from core.state import KilnState

@njit
def get_smooth_step(T, T_min, span=50.0):
    """Sert eşikler yerine 0 ile 1 arasında yumuşak geçiş sağlar."""
    val = (T - T_min) / span
    if val < 0: return 0.0
    if val > 1: return 1.0
    return val

@njit
def _numba_step_core(N, Ts, Tg, Tw, X, m_CaO, m_SiO2, m_C2S, m_Al2O3, m_Fe2O3, m_C3A, m_C4AF,
                     rho_g, dt, dz, v_s, m_dot_s, m_dot_g_inlet, 
                     cp_s, cp_g, dH_vec, area_gas, q_gs_vec, q_rad_gs_vec,
                     damping, effective_thermal_mass, exchange_area_per_dz,
                     k0_vec, Ea_vec, R_const, T_min_vec,
                     k_flux_c3a, k_flux_c4af):
    
    rates = np.zeros((5, N))
    
    for i in range(N):
        # 1. Kalsinasyon (X)
        if Ts[i] >= T_min_vec[0] and X[i] < 1.0:
            k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R_const * Ts[i]))
            rates[0, i] = min(k_calc * (1.0 - X[i]), 0.04)

        # 2. Belit (C2S)
        f_c2s = get_smooth_step(Ts[i], T_min_vec[1], 50.0)
        if f_c2s > 0 and m_CaO[i] > 1e-4 and m_SiO2[i] > 1e-4:
            k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R_const * Ts[i]))
            rates[1, i] = min(k_c2s * m_CaO[i] * m_SiO2[i], 0.02) * f_c2s

        # 3. Alit (C3S)
        f_c3s = get_smooth_step(Ts[i], T_min_vec[2], 75.0)
        belite_factor = min(max((m_C2S[i] - 0.10) / 0.05, 0.0), 1.0)
        cao_factor = min(max((m_CaO[i] - 0.002) / 0.03, 0.0), 1.0)
        
        if f_c3s > 0 and m_C2S[i] > 0.01:
            k_c3s = (k0_vec[2] * 0.75) * np.exp(-Ea_vec[2] / (R_const * Ts[i]))
            rates[2, i] = k_c3s * m_C2S[i] * m_CaO[i] * belite_factor * cao_factor * f_c3s

        # 4. Flux Fazları
        if Ts[i] >= 1450.0: rates[3, i] = k_flux_c3a * m_Al2O3[i]
        if Ts[i] >= 1400.0: rates[4, i] = k_flux_c4af * m_Fe2O3[i]

    # Gaz Kütle Akış Güncellemesi (CO2 salınımı)
    m_dot_co2 = rates[0] * m_dot_s * 0.4397
    m_dot_g_local = np.full(N, m_dot_g_inlet)
    acc = 0.0
    for i in range(N - 1, -1, -1):
        acc += m_dot_co2[i]
        m_dot_g_local[i] += acc

    # Isı Dengesi
    q_rxn = (rates[0]*dH_vec[0] + rates[1]*dH_vec[1] + rates[2]*dH_vec[2] + rates[3]*dH_vec[3] + rates[4]*dH_vec[4]) * m_dot_s
    
    dTs, dTg = np.zeros(N), np.zeros(N)
    
    # Katı Enerji Dengesi
    dTs[1:] = (v_s * (Ts[:-1] - Ts[1:]) / dz)
    # Mevcut energy.py'ye göre q_rad_gs_vec doğrudan eklenir
    dTs += (q_gs_vec + q_rad_gs_vec - 15.0*(Ts-Tw)*exchange_area_per_dz - q_rxn) / (effective_thermal_mass * cp_s)
    
    # Gaz Enerji Dengesi
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    dTg -= (q_gs_vec + q_rad_gs_vec) / (np.maximum(1.0, m_dot_g_local) * cp_g)
            
    return dTs * damping, dTg * damping * 0.8, rates

class KilnSolver:
    def __init__(self, config, kinetics, transport, energy):
        self.cfg, self.kin, self.tra, self.en = config, kinetics, transport, energy
        nodes = int(self._safe_f(config['kiln']['nodes']))
        self.state = KilnState(nodes)
        self.dz = self._safe_f(config['kiln']['length']) / nodes
        self.damping = 0.16 

    def _safe_f(self, v): return float(v[0]) if isinstance(v, list) else float(v)

    def solve_step(self, dt, fuel_rate=None, feed_rate=None, kiln_rpm=None, fan_rate=None):
        dt_v = float(dt)
        c0 = self.cfg['raw_meal_composition']
        
        # Giriş Değişkenleri
        cur_fuel = fuel_rate if fuel_rate is not None else self._safe_f(self.cfg['gas']['fuel_rate'])
        cur_fan = fan_rate if fan_rate is not None else self.cfg.get('fan_rate_current', 800.0)
        
        m_dot_g_inlet = self._safe_f(self.cfg['gas'].get('nominal_flow', 8.5)) * (cur_fan / 800.0)
        v_s = self.tra.calculate_solid_velocity(kiln_rpm or self._safe_f(self.cfg['kiln']['rpm']))
        m_dot_s = feed_rate or self._safe_f(self.cfg['material']['feed_rate'])
        
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        area_gas = (np.pi * (diameter / 2)**2) * 0.90
        exchange_area_dz = diameter * np.pi * self.dz

        # --- ENERGYMODEL ENTEGRASYONU ---
        # 1. Konveksiyon
        h_gs = self.en.calculate_convection_coeff(cur_fan)
        q_gs_vec = h_gs * (self.state.Tg - self.state.Ts) * exchange_area_dz
        
        # 2. Radyasyon (energy.py içindeki sadeleşmiş metod çağrılıyor)
        q_rad_gs_vec = self.en.calculate_radiation_flux(self.state.Tg, self.state.Ts, exchange_area_dz)

        # --- TAŞINIM ---
        flow_factor = min(1.0, v_s * dt_v / self.dz)
        params = ['m_SiO2', 'm_SiO2_locked', 'm_Al2O3', 'm_Fe2O3', 'm_CaO', 'm_C2S', 'm_C3S', 'm_C3A', 'm_C4AF', 'X', 'total_mass']
        for p in params:
            if hasattr(self.state, p):
                arr = getattr(self.state, p)
                arr[1:] = (1 - flow_factor) * arr[1:] + flow_factor * arr[:-1]
                if p == 'm_SiO2': arr[0] = float(c0['SiO2']) * 0.75
                elif p == 'm_SiO2_locked': arr[0] = float(c0['SiO2']) * 0.25
                elif p in ['m_Al2O3', 'm_Fe2O3']: arr[0] = float(c0[p[2:]])
                elif p == 'total_mass': arr[0] = 1.0
                else: arr[0] = 0.0

        # --- SİLİS AKTİVASYONU ---
        activation_factor = np.clip((self.state.Ts - 1350.0) / 150.0, 0.0, 1.0)
        unlocked_mass = np.minimum(self.state.m_SiO2_locked * 0.40 * activation_factor * dt_v, self.state.m_SiO2_locked)
        self.state.m_SiO2_locked -= unlocked_mass
        self.state.m_SiO2 += unlocked_mass

        # --- REAKSİYONLAR (NUMBA) ---
        eff_mass = (m_dot_s / max(0.001, v_s)) * self.dz + 20.0
        
        dTs, dTg, rates = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.Tw,
            self.state.X, self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S,
            self.state.m_Al2O3, self.state.m_Fe2O3, self.state.m_C3A, self.state.m_C4AF,
            self.state.rho_g, dt_v, self.dz, v_s, m_dot_s, m_dot_g_inlet,
            self._safe_f(self.cfg['material']['cp_s']), self._safe_f(self.cfg['gas']['cp_g']),
            np.array([self.kin.dH_calc, self.kin.dH_c2s, self.kin.dH_c3s, -125.0, -100.0]), 
            area_gas, q_gs_vec, q_rad_gs_vec, self.damping, eff_mass, exchange_area_dz,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min,
            self.cfg['kinetics'].get('pre_factor_c3a', 0.0015),
            self.cfg['kinetics'].get('pre_factor_c4af', 0.0010)
        )

        # --- GÜNCELLEME VE KÜTLE DENGESİ ---
        self.state.Ts += dTs * dt_v
        self.state.Tg += dTg * dt_v
        self.state.X += rates[0] * dt_v

        # Faz kütle güncellemeleri
        self.state.m_C3A += np.minimum(rates[3] * dt_v, self.state.m_Al2O3 / 0.377)
        self.state.m_C4AF += np.minimum(rates[4] * dt_v, self.state.m_Fe2O3 / 0.329)
        self.state.m_Al2O3 = np.maximum(1e-6, self.state.m_Al2O3 - (rates[3]*dt_v*0.377 + rates[4]*dt_v*0.210))
        self.state.m_Fe2O3 = np.maximum(1e-6, self.state.m_Fe2O3 - (rates[4]*dt_v*0.329))
        sio2_cons = np.minimum(rates[1] * dt_v * 0.349, self.state.m_SiO2 * 0.98)
        self.state.m_SiO2 -= sio2_cons
        c3s_inc = np.minimum(rates[2] * dt_v, self.state.m_C2S * 0.5)
        self.state.m_C3S += c3s_inc
        self.state.m_C2S += (sio2_cons / 0.349) - (c3s_inc * 0.754)
        self.state.m_CaO += (rates[0]*dt_v*0.561) - ((sio2_cons/0.349)*0.651 + c3s_inc*0.246 + rates[3]*dt_v*0.623 + rates[4]*dt_v*0.461)
        self.state.total_mass = 1.0 - (np.minimum(self.state.X, 1.0) * float(c0.get('CaCO3', 0.80)) * 0.4397)

        # Guard & Boundary Conditions
        for p in ['m_SiO2', 'm_Al2O3', 'm_Fe2O3', 'm_CaO', 'm_C2S', 'm_C3S', 'm_C3A', 'm_C4AF']:
            setattr(self.state, p, np.maximum(1e-6, getattr(self.state, p)))
        
        # --- DINAMIK GAZ GIRIS SICAKLIGI ---
        lhv_fuel = 40000.0 # kJ/kg
        fuel_kg_s = cur_fuel / 3.6
        cp_g = self._safe_f(self.cfg['gas']['cp_g'])
        t_sec_air = 1100.0
        if m_dot_g_inlet > 0.1:
            delta_T_flame = (fuel_kg_s * lhv_fuel) / (m_dot_g_inlet * cp_g / 1000.0)
            t_gas_inlet = t_sec_air + delta_T_flame
        else:
            t_gas_inlet = t_sec_air
            
        self.state.Tg[-1] = min(t_gas_inlet, 2500.0) 
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])