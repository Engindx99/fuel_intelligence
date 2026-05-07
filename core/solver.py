import numpy as np
from numba import njit
from core.state import KilnState

@njit
def _numba_step_core(N, Ts, Tg, Tw, X, m_CaO, m_SiO2, m_C2S, rho_g, dt, dz, v_s, m_dot_s, m_dot_g_inlet, 
                     cp_s, cp_g, dH_vec, area_gas, exchange_area_per_dz, 
                     damping, effective_thermal_mass, h_gs, sigma, eps_eff,
                     k0_vec, Ea_vec, R_const, T_min_vec):
    rates = np.zeros((3, N))
    
    # 1. Reaksiyon Maskeleri
    mask_calc = (Ts >= T_min_vec[0]) & (X < 1.0)
    # SiO2 ve CaO varsa Belit (C2S) oluşabilir
    mask_c2s  = (Ts >= T_min_vec[1]) & (m_CaO > 1e-4) & (m_SiO2 > 1e-4)
    # C2S ve CaO varsa Alit (C3S) oluşabilir
    mask_c3s  = (Ts >= T_min_vec[2]) & (m_C2S > 0.01) & (m_CaO > 1e-4)

    # 2. Kinetik Hızlar
    k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R_const * Ts))
    rates[0, mask_calc] = np.minimum(k_calc[mask_calc] * (1.0 - X[mask_calc]), 0.05)

    k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R_const * Ts))
    rates[1, mask_c2s] = np.minimum(k_c2s[mask_c2s] * (m_CaO[mask_c2s]**2) * m_SiO2[mask_c2s], 0.005)

    k_c3s = k0_vec[2] * np.exp(-Ea_vec[2] / (R_const * Ts))
    # Belit %5'in altına indikçe Alit oluşumu yavaşlar (Sayısal kararlılık için)
    c2s_resistance = np.clip((m_C2S - 0.05) / (0.25 - 0.05), 0.0, 1.0)
    rates[2, mask_c3s] = np.minimum(k_c3s[mask_c3s] * m_C2S[mask_c3s] * m_CaO[mask_c3s] * c2s_resistance[mask_c3s], 0.003)

    # 3. Gaz ve Enerji Dengesi
    CO2_RATIO = 0.4397
    m_dot_co2 = rates[0] * m_dot_s * CO2_RATIO
    m_dot_g_local = np.full(N, m_dot_g_inlet)
    acc = 0.0
    for i in range(N - 1, -1, -1):
        acc += m_dot_co2[i]
        m_dot_g_local[i] += acc

    q_gs = h_gs * (Tg - Ts) * exchange_area_per_dz
    q_rad_gs = sigma * eps_eff * (Tg**4 - Ts**4) * exchange_area_per_dz
    q_sw = 15.0 * (Ts - Tw) * exchange_area_per_dz
    q_rxn = (rates[0] * dH_vec[0] + rates[1] * dH_vec[1] + rates[2] * dH_vec[2]) * m_dot_s
    
    dTs, dTg = np.zeros(N), np.zeros(N)
    dTs[1:] = (v_s * (Ts[:-1] - Ts[1:]) / dz)
    dTs += (q_gs + q_rad_gs - q_sw - q_rxn) / (effective_thermal_mass * cp_s)
    dTs *= damping 
    
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    dTg -= (q_gs + q_rad_gs) / (np.maximum(1.0, m_dot_g_local) * cp_g)
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
        self.damping = 0.20 

    def _safe_f(self, value):
        if isinstance(value, list): return float(value[0])
        return float(value)

    def _calculate_flame_temp(self, fuel_rate, fan_rate=None):
        base, gain = 20.0, 460.0
        flame = base + (np.sqrt(max(0, fuel_rate)) * gain)
        if fan_rate is not None:
            flame *= (850.0 / max(400, fan_rate))**0.40
        return min(2400.0, flame)

    def solve_step(self, dt, fuel_rate=None, feed_rate=None, kiln_rpm=None, fan_rate=None):
        dt_v = float(dt)
        c0 = self.cfg['raw_meal_composition']
        
        # Giriş Parametreleri
        f_rate = fuel_rate if fuel_rate is not None else self._safe_f(self.cfg['gas'].get('fuel_rate', 16.0))
        cur_fan = fan_rate if fan_rate is not None else self.cfg.get('fan_rate_current', 800.0)
        self.cfg['gas']['temp_inlet'] = self._calculate_flame_temp(f_rate, cur_fan)
        m_dot_g_inlet = self._safe_f(self.cfg['gas'].get('nominal_flow', 8.5)) * (cur_fan / 850.0)
        h_gs = self._safe_f(self.cfg['gas'].get('h_gs', 400.0)) * ((cur_fan / 850.0)**0.4)
        current_rpm = kiln_rpm if kiln_rpm is not None else self._safe_f(self.cfg['kiln'].get('rpm', 2.0))
        v_s = self.tra.calculate_solid_velocity(current_rpm)
        m_dot_s = feed_rate if feed_rate is not None else self._safe_f(self.cfg['material']['feed_rate'])
        
        eff_mass = (m_dot_s / max(0.001, v_s)) * self.dz + 20.0
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        area_gs_dz = diameter * self.dz * (self.tra.get_dynamic_filling_degree(current_rpm) / 0.10)
        area_gas = (np.pi * (diameter / 2)**2) * 0.90
        eps_eff = (self._safe_f(self.cfg['gas'].get('emissivity_g', 0.3)) + 0.85) / 2.0

        # --- KRİTİK DEĞİŞİKLİK: ÖNCE TAŞINIM (SÜREKLİ BESLEME) ---
        # Fırın boyunca malzemeyi kaydırıyoruz. Bu, her düğüme taze SiO2 girişi sağlar.
        flow_factor = min(1.0, v_s * dt_v / self.dz)
        phases = ['m_SiO2', 'm_Al2O3', 'm_Fe2O3', 'm_CaO', 'm_C2S', 'm_C3S', 'm_C3A', 'm_C4AF', 'X']
        for p in phases:
            arr = getattr(self.state, p)
            arr[1:] = (1 - flow_factor) * arr[1:] + flow_factor * arr[:-1]
            
            # FIRIN GİRİŞİ (Sınır Koşulu - Taze Hammadde)
            if p == 'm_SiO2': arr[0] = float(c0['SiO2'])
            elif p == 'm_Al2O3': arr[0] = float(c0['Al2O3'])
            elif p == 'm_Fe2O3': arr[0] = float(c0['Fe2O3'])
            elif p == 'X': arr[0] = 0.0
            elif p == 'm_CaO': arr[0] = 0.0
            else: arr[0] = 0.0

        self.state.update_gas_density()

        # --- REAKSİYONLARIN HESAPLANMASI ---
        dTs, dTg, rates = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.Tw,
            self.state.X, self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S,
            self.state.rho_g, dt_v, self.dz, v_s, m_dot_s, m_dot_g_inlet,
            self._safe_f(self.cfg['material']['cp_s']), self._safe_f(self.cfg['gas']['cp_g']),
            np.array([self.kin.dH_calc, self.kin.dH_c2s, self.kin.dH_c3s]), 
            area_gas, area_gs_dz, self.damping, eff_mass, h_gs, 5.67e-8, eps_eff,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min
        )

        # --- DURUM GÜNCELLEME VE STOKİYOMETRİ ---
        self.state.X += rates[0] * dt_v
        self.state.Ts += dTs * dt_v
        self.state.Tg += dTg * dt_v

        # SiO2 Tüketimi (Gelen miktar kadar harcanabilir)
        sio2_needed = (rates[1] * dt_v) * 0.349
        actual_sio2_cons = np.minimum(self.state.m_SiO2 - 1e-6, sio2_needed)
        self.state.m_SiO2 -= actual_sio2_cons
        
        # Belit Oluşumu (Tüketilen SiO2'ye orantılı)
        c2s_formed = actual_sio2_cons / 0.349
        actual_c3s_inc = rates[2] * dt_v
        
        self.state.m_C3S += actual_c3s_inc
        # Belit Değişimi = Oluşan - (Alit'e Dönüşen)
        self.state.m_C2S += c2s_formed - (actual_c3s_inc * 0.754)
        
        # CaO Dengesi: Üretilen (Kalsinasyon) - Tüketilen (C2S ve C3S oluşumu)
        cao_prod = rates[0] * dt_v * 0.5607
        cao_cons = (c2s_formed * 0.651) + (actual_c3s_inc * 0.737)
        self.state.m_CaO += (cao_prod - cao_cons)

        # Emniyet Sınırları
        self.state.m_SiO2 = np.maximum(0.0001, self.state.m_SiO2)
        self.state.m_CaO = np.maximum(1e-5, self.state.m_CaO)
        self.state.m_C2S = np.maximum(0.0, self.state.m_C2S)

        # Flux Fazlar (C3A ve C4AF)
        sinter_mask = self.state.Ts > 1523.0
        L = 3.0 * c0['Al2O3'] + 2.25 * c0['Fe2O3']
        c4af_inc = np.minimum((c0['Fe2O3'] * 2.85 - self.state.m_C4AF), 0.05 * L) * dt_v
        self.state.m_C4AF[sinter_mask] += c4af_inc[sinter_mask]
        c3a_inc = np.minimum(((c0['Al2O3'] - (self.state.m_C4AF * 0.21)) * 2.45 - self.state.m_C3A), 0.08 * L) * dt_v
        self.state.m_C3A[sinter_mask] += c3a_inc[sinter_mask]

        self.state.total_mass = 1.0 - (self.state.X * c0['CaCO3'] * 0.4397)
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        self.state.Tg[-1] = self._safe_f(self.cfg['gas']['temp_inlet'])