import numpy as np
from numba import njit

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
                     cp_s, cp_g, dH_vec, area_gas, exchange_area_per_dz, 
                     damping, effective_thermal_mass, h_gs, sigma, eps_eff,
                     k0_vec, Ea_vec, R_const, T_min_vec,
                     k_flux_c3a, k_flux_c4af):
    
    rates = np.zeros((5, N))
    
    for i in range(N):
        # 1. Kalsinasyon (X)
        if Ts[i] >= T_min_vec[0] and X[i] < 1.0:
            k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R_const * Ts[i]))
            rates[0, i] = min(k_calc * (1.0 - X[i]), 0.04)

        # 2. Belit (C2S) - Yumuşak Başlangıç
        f_c2s = get_smooth_step(Ts[i], T_min_vec[1], 50.0)
        if f_c2s > 0 and m_CaO[i] > 1e-4 and m_SiO2[i] > 1e-4:
            k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R_const * Ts[i]))
            rates[1, i] = min(k_c2s * m_CaO[i] * m_SiO2[i], 0.02) * f_c2s

        # 3. Alit (C3S)
        f_c3s = get_smooth_step(Ts[i], T_min_vec[2], 75.0)
        
        # Belit Koruma (%10)
        b_val = (m_C2S[i] - 0.10) / 0.05
        belite_factor = min(max(b_val, 0.0), 1.0)
        
        # Kireç Sönümleme
        c_val = (m_CaO[i] - 0.002) / 0.03
        cao_factor = min(max(c_val, 0.0), 1.0)
        
        if f_c3s > 0 and m_C2S[i] > 0.01:
            k_c3s = (k0_vec[2] * 0.75) * np.exp(-Ea_vec[2] / (R_const * Ts[i]))
            rates[2, i] = k_c3s * m_C2S[i] * m_CaO[i] * belite_factor * cao_factor * f_c3s

        # 4. Flux Fazları
        if Ts[i] >= 1450.0: rates[3, i] = k_flux_c3a * m_Al2O3[i]
        if Ts[i] >= 1400.0: rates[4, i] = k_flux_c4af * m_Fe2O3[i]

    # Termal ve Gaz Dinamiği
    m_dot_co2 = rates[0] * m_dot_s * 0.4397
    m_dot_g_local = np.full(N, m_dot_g_inlet)
    acc = 0.0
    for i in range(N - 1, -1, -1):
        acc += m_dot_co2[i]
        m_dot_g_local[i] += acc

    # Isı Transferi
    q_gs = h_gs * (Tg - Ts) * exchange_area_per_dz
    q_rad_gs = sigma * eps_eff * (Tg**4 - Ts**4) * exchange_area_per_dz
    q_rxn = (rates[0]*dH_vec[0] + rates[1]*dH_vec[1] + rates[2]*dH_vec[2] + 
             rates[3]*dH_vec[3] + rates[4]*dH_vec[4]) * m_dot_s
    
    dTs, dTg = np.zeros(N), np.zeros(N)
    
    # Solid Sıcaklık Değişimi
    dTs[1:] = (v_s * (Ts[:-1] - Ts[1:]) / dz)
    dTs += (q_gs + q_rad_gs - 15.0*(Ts-Tw)*exchange_area_per_dz - q_rxn) / (effective_thermal_mass * cp_s)
    
    # Gaz Sıcaklık Değişimi
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    dTg -= (q_gs + q_rad_gs) / (np.maximum(1.0, m_dot_g_local) * cp_g)
            
    # Damping etkisi kaldırıldı: Doğrudan türevler döndürülüyor
    return dTs, dTg, rates

class KilnSolver:
    def __init__(self, config, kinetics, transport, energy):
        self.cfg = config
        self.kin = kinetics
        self.tra = transport
        self.en = energy
        
        nodes = int(self._safe_f(config['kiln']['nodes']))
        from core.state import KilnState
        self.state = KilnState(nodes)
        self.dz = self._safe_f(config['kiln']['length']) / nodes
        self.damping = 1.0 # Damping artık 1.0 (etkisiz)

    def _safe_f(self, v): 
        if isinstance(v, list): return float(v[0])
        return float(v)

    def solve_step(self, dt, fuel_rate=None, feed_rate=None, kiln_rpm=None, fan_rate=None):
        dt_v = float(dt)
        c0 = self.cfg['raw_meal_composition']
        
        cur_fan = fan_rate if fan_rate is not None else self.cfg.get('fan_rate_current', 850.0)
        m_dot_g_inlet = self._safe_f(self.cfg['gas'].get('nominal_flow', 8.5)) * (cur_fan / 850.0)
        h_gs = self._safe_f(self.cfg['gas'].get('h_gs', 350.0)) * ((cur_fan / 850.0)**0.4)
        v_s = self.tra.calculate_solid_velocity(kiln_rpm or self._safe_f(self.cfg['kiln']['rpm']))
        m_dot_s = feed_rate or self._safe_f(self.cfg['material']['feed_rate'])
        
        eff_mass = (m_dot_s / max(0.001, v_s)) * self.dz + 20.0
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        area_gas = (np.pi * (diameter / 2)**2) * 0.90

        if 'energy' in self.cfg and 'eps_eff' in self.cfg['energy']:
            eps_eff = self._safe_f(self.cfg['energy']['eps_eff'])
        elif hasattr(self.en, 'eps_eff'):
            eps_eff = self.en.eps_eff
        else:
            eg = self._safe_f(self.cfg['gas'].get('emissivity_g', 0.65))
            es = self._safe_f(self.cfg['material'].get('emissivity_s', 0.85))
            eps_eff = (eg + es) / 2.0

        # --- ADIM A: TAŞINIM ---
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

        # --- ADIM B: SİLİS AKTİVASYONU ---
        activation_factor = np.clip((self.state.Ts - 1350.0) / 150.0, 0.0, 1.0)
        unlocked_mass = self.state.m_SiO2_locked * 0.40 * activation_factor * dt_v
        unlocked_mass = np.minimum(unlocked_mass, self.state.m_SiO2_locked)
        self.state.m_SiO2_locked -= unlocked_mass
        self.state.m_SiO2 += unlocked_mass

        # --- ADIM C: REAKSİYONLAR (NUMBA) ---
        dTs, dTg, rates = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.Tw,
            self.state.X, self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S,
            self.state.m_Al2O3, self.state.m_Fe2O3, self.state.m_C3A, self.state.m_C4AF,
            self.state.rho_g, dt_v, self.dz, v_s, m_dot_s, m_dot_g_inlet,
            self._safe_f(self.cfg['material']['cp_s']), self._safe_f(self.cfg['gas']['cp_g']),
            np.array([self.kin.dH_calc, self.kin.dH_c2s, self.kin.dH_c3s, -125.0, -100.0]), 
            area_gas, diameter * self.dz, 1.0, # Damping 1.0 olarak geçildi
            eff_mass, h_gs, 5.67e-8, eps_eff,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min,
            self.cfg['kinetics'].get('pre_factor_c3a', 0.0015),
            self.cfg['kinetics'].get('pre_factor_c4af', 0.0010)
        )

        # --- ADIM D: GÜNCELLEME VE KÜTLE DENGESİ ---
        self.state.Ts += dTs * dt_v
        self.state.Tg += dTg * dt_v
        self.state.X += rates[0] * dt_v

        c3a_inc = np.minimum(rates[3] * dt_v, self.state.m_Al2O3 / 0.377)
        c4af_inc = np.minimum(rates[4] * dt_v, self.state.m_Fe2O3 / 0.329)
        self.state.m_C3A += c3a_inc; self.state.m_C4AF += c4af_inc
        self.state.m_Al2O3 -= (c3a_inc * 0.377 + c4af_inc * 0.210)
        self.state.m_Fe2O3 -= (c4af_inc * 0.329)
        
        sio2_cons = np.minimum(rates[1] * dt_v * 0.349, self.state.m_SiO2 * 0.98)
        self.state.m_SiO2 -= sio2_cons
        c3s_inc = np.minimum(rates[2] * dt_v, self.state.m_C2S * 0.5)
        self.state.m_C3S += c3s_inc
        self.state.m_C2S += (sio2_cons / 0.349) - (c3s_inc * 0.754)
        
        cao_gain = rates[0] * dt_v * 0.561
        cao_loss = ((sio2_cons/0.349)*0.651 + c3s_inc*0.246 + c3a_inc*0.623 + c4af_inc*0.461)
        self.state.m_CaO += (cao_gain - cao_loss)

        loi_factor = float(c0.get('CaCO3', 0.80)) * 0.4397
        self.state.total_mass = 1.0 - (np.minimum(self.state.X, 1.0) * loi_factor)

        for p in ['m_SiO2', 'm_Al2O3', 'm_Fe2O3', 'm_CaO', 'm_C2S', 'm_C3S', 'm_C3A', 'm_C4AF']:
            setattr(self.state, p, np.maximum(1e-6, getattr(self.state, p)))
        
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        self.state.Tg[-1] = self._safe_f(self.cfg['gas']['temp_inlet'])