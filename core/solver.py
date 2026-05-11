import numpy as np
from numba import njit
from core.state import KilnState

@njit
def _numba_step_core(N, Ts, Tg, Tw, X, m_CaO, m_SiO2, m_C2S, m_Al2O3, m_Fe2O3, m_C3A, m_C4AF,
                     rho_g, dt, dz, v_s, m_dot_s_kg_s, m_dot_g_inlet_kg_s, 
                     cp_s, cp_g, dH_vec, area_gas, q_gs_vec, q_rad_net_vec,
                     eff_mass_node, exchange_area_per_dz,
                     k0_vec, Ea_vec, R_const, T_min_vec,
                     pre_factors, loss_coeff, kiln_length,
                     clip_min, clip_max, burning_zone_factor):
    
    rates = np.zeros((5, N))
    x_coords = np.linspace(0, 1, N)
    
    # 1. AGRESİF BURNING ZONE BOOST (Sigmoid)
    # Çıkış bölgesinde radyasyon ve alev etkisiyle transfer katsayısını 4.5 katına çıkarır
    dynamic_boost = 1.0 + 3.5 / (1.0 + np.exp(-25 * (x_coords - 0.82)))
    
    for i in range(N):
        T_eval = max(300.0, Ts[i])
        
        # Kalsinasyon Reaksiyonu
        if x_coords[i] >= 0.30:
            if T_eval >= T_min_vec[0] and X[i] < 1.0:
                arg = -Ea_vec[0] / (R_const * T_eval)
                exp_arg = max(clip_min, min(clip_max, arg))
                rates[0, i] = k0_vec[0] * np.exp(exp_arg) * (1.0 - X[i]) 

        # Belit (C2S) Oluşumu (Ekzotermik)
        if x_coords[i] >= 0.55 and T_eval >= 1100.0:
            lock_c2s = 1.0 / (1.0 + np.exp(-12.0 * (X[i] - 0.75)))
            k_c2s = k0_vec[1] * np.exp(max(clip_min, min(clip_max, -Ea_vec[1]/(R_const*T_eval))))
            rates[1, i] = k_c2s * m_CaO[i] * m_SiO2[i] * lock_c2s

        # Alit (C3S) Oluşumu
        if x_coords[i] >= 0.75 and T_eval >= 1350.0:
            liquid_effect = (m_C3A[i] + m_C4AF[i] + 0.06) * 10.0
            k_c3s = k0_vec[2] * np.exp(max(clip_min, min(clip_max, -Ea_vec[2]/(R_const*T_eval))))
            rates[2, i] = k_c3s * m_C2S[i] * m_CaO[i] * liquid_effect * burning_zone_factor

        # Yardımcı Fazlar (Flux)
        if x_coords[i] >= 0.60:
            if T_eval >= 1280.0: rates[3, i] = pre_factors[0] * m_Al2O3[i]
            if T_eval >= 1220.0: rates[4, i] = pre_factors[1] * m_Fe2O3[i]

    # ENERJİ TÜREVLERİ
    q_rxn = (rates[0]*dH_vec[0] + rates[1]*dH_vec[1] + rates[2]*dH_vec[2]) * m_dot_s_kg_s
    
    # FİZİKSEL ATALET KORUMASI: Patlamayı engellemek için alt sınır 5.0
    thermal_capacity_s = np.maximum(5.0, eff_mass_node * cp_s) 
    
    dTs, dTg = np.zeros(N), np.zeros(N)
    
    # Malzeme Taşınımı (Advection)
    dTs[1:] = -v_s * (Ts[1:] - Ts[:-1]) / dz
    
    # Isı Kayıpları
    q_loss_wall = loss_coeff * (Ts - Tw) * exchange_area_per_dz
    
    # RADIATION AMPLIFICATION: 1400K üstünde non-lineer radyasyon baskınlığı
    rad_amp = np.where(Ts > 1400.0, 1.0 + (Ts - 1400.0) / 350.0, 1.0)
    
    # Toplam Isı Akısı
    net_heat_to_s = (dynamic_boost * (q_gs_vec + q_rad_net_vec * rad_amp)) - q_loss_wall - q_rxn
    
    # Sayısal Koruma Clipping (Runaway serbest bırakıldı)
    dTs += np.clip(net_heat_to_s / thermal_capacity_s, -5000.0, 5000.0) 
    
    # Gaz Fazı Enerji Dengesi
    m_dot_g_local = np.maximum(0.5, m_dot_g_inlet_kg_s)
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    dTg += np.clip(-(q_gs_vec + q_rad_net_vec) / (m_dot_g_local * cp_g), -8000.0, 8000.0)
            
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
        self.use_adaptive_dt = True 
        
        # Başlangıçta gaz ve duvar sıcaklıklarını düşük tutuyoruz (Rampa başlangıcı)
        self.state.Tg = np.full(nodes, 350.0)
        self.state.Tw = np.full(nodes, 300.0)

    def _safe_f(self, v): 
        return float(v[0]) if isinstance(v, list) else float(v)

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        dt_v = float(dt)
        
        # RAMPA HESABI: İlk 2 saat (7200 saniye) boyunca 0'dan 1'e rampa
        sim_hour = self.total_sim_time / 3600.0 
        ramp_factor = min(1.0, sim_hour / 12.0) 
        
        # HEDEF GAUSSIAN ALEV PROFİLİ
        nodes = self.state.N
        x = np.linspace(0, 1, nodes)
        target_Tg = 750 + 1750 * np.exp(-(x-1.0)**2 / 0.18)
        target_Tw = 350 + 750 * x 
        
        # Rampa uygulanmış güncel profil
        self.state.Tg = 350.0 + (target_Tg - 350.0) * ramp_factor
        self.state.Tw = 300.0 + (target_Tw - 300.0) * ramp_factor

        m_dot_s_kg_s = (feed_rate * 1000.0) / 3600.0 
        m_dot_g_inlet_kg_s = self._safe_f(self.cfg['gas']['nominal_flow']) * (fan_rate / 800.0)
        v_s = self.tra.calculate_solid_velocity(kiln_rpm)
        fill_degree = self.tra.get_dynamic_filling_degree(kiln_rpm)
        
        # KÜTLE HESABI VE ATALET KORUMASI (Kritik Düzeltme)
        eff_mass_node = max(0.5, (m_dot_s_kg_s / max(1e-4, v_s)) * self.dz)
        
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - fill_degree)
        exchange_area_dz = diameter * np.pi * self.dz

        # AYRI KALİBRE EDİLMİŞ ISI TRANSFER GÜÇLERİ (Katsayılar korundu)
        h_gs = self.en.calculate_convection_coeff(fan_rate)
        q_gs_vec = h_gs * (self.state.Tg - self.state.Ts) * exchange_area_dz * 0.024
        q_rad_net_vec = self.en.calculate_radiation_flux(self.state.Tg, self.state.Ts, self.state.Tw, exchange_area_dz) * 2.4

        # Malzeme Kaydırma (Advection Step)
        f_red = np.clip(v_s * dt_v / self.dz, 0.0, 0.5)
        params = ['m_SiO2', 'm_CaO', 'm_C2S', 'm_C3S', 'X', 'm_Al2O3', 'm_Fe2O3', 'm_C3A', 'm_C4AF']
        for p in params:
            arr = getattr(self.state, p)
            arr[1:] = (1.0 - f_red) * arr[1:] + f_red * arr[:-1]

        # Numba Core Çağrısı
        dTs, dTg, rates = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.Tw,
            self.state.X, self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S,
            self.state.m_Al2O3, self.state.m_Fe2O3, self.state.m_C3A, self.state.m_C4AF,
            self.state.rho_g, dt_v, self.dz, v_s, m_dot_s_kg_s, m_dot_g_inlet_kg_s,
            self._safe_f(self.cfg['material']['cp_s']), self._safe_f(self.cfg['gas']['cp_g']),
            self.kin.get_enthalpy_vector(), area_gas, q_gs_vec, q_rad_net_vec, 
            eff_mass_node, exchange_area_dz,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min,
            self.kin.pre_factors, 0.05, 
            self._safe_f(self.cfg['kiln']['length']), -80.0, 20.0,
            self.kin.burning_zone_factor
        )

        # ADAPTIVE TIMESTEP (CFL-LIKE)
        if self.use_adaptive_dt:
            max_dTs = np.max(np.abs(dTs))
            actual_dt = min(dt_v, 1.5 / (max_dTs + 1e-6))
        else:
            actual_dt = dt_v

        # Durum Güncellemeleri
        self.state.Ts += dTs * actual_dt
        # Tg ve Tw rampa ile dışarıdan set ediliyor, dTg takip amaçlı bırakılabilir.
        
        # Flame Persistence (Zorlamalı Sıcaklık Alanı) - Rampa ile uyumlu olması için katsayı ile çarpıldı
        bz_idx = int(self.state.N * 0.85)
        self.state.Tg[bz_idx:] = np.maximum(self.state.Tg[bz_idx:], 2250.0 * ramp_factor + 350.0 * (1-ramp_factor))
        
        # Kimyasal Faz Güncellemeleri
        self.state.X = np.clip(self.state.X + rates[0] * actual_dt, 0.0, 1.0)
        self.state.m_C3S += rates[2] * actual_dt
        self.state.m_C2S += (rates[1] - rates[2] * 0.75) * actual_dt
        self.state.m_SiO2 = np.maximum(0.0, self.state.m_SiO2 - (rates[1] * 0.35) * actual_dt)
        
        m_CaCO3_init = self.cfg['raw_meal_composition']['CaCO3']
        dn_calc = rates[0] * m_CaCO3_init * actual_dt
        self.state.m_CaO = np.maximum(0.0, self.state.m_CaO + (dn_calc * 0.56 - rates[1] * 0.65 - rates[2] * 0.25) * actual_dt)
        
        self.state.m_C3A += rates[3] * actual_dt
        self.state.m_C4AF += rates[4] * actual_dt
        self.state.m_Al2O3 = np.maximum(0.0, self.state.m_Al2O3 - rates[3] * actual_dt)
        self.state.m_Fe2O3 = np.maximum(0.0, self.state.m_Fe2O3 - rates[4] * actual_dt)

        # Giriş Sınır Koşulu
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        
        self.total_sim_time += actual_dt
        return actual_dt