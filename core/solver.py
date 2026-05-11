import numpy as np
from numba import njit
from core.state import KilnState

@njit
def get_zone_params(x_ratio):
    """
    Fırın konumuna göre hedef sıcaklık ve bölge tanımını döndürür.
    """
    if x_ratio < 0.35:
        return 900.0, 0  # PREHEAT
    elif x_ratio < 0.60:
        return 1250.0, 1 # CALCINATION
    elif x_ratio < 0.80:
        return 1400.0, 2 # TRANSITION
    else:
        return 1550.0, 3 # BURNING

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
                     k_flux_c3a, k_flux_c4af, loss_coeff, kiln_length):
    
    rates = np.zeros((5, N))
    
    for i in range(N):
        x_ratio = (i * dz) / kiln_length
        _, zone_id = get_zone_params(x_ratio)
        
        # Arrhenius koruması: Sıcaklık çok düşükse kinetik hesaplamayı atla
        T_eval = max(400.0, Ts[i])
        
        # 1. KİNETİK REAKSİYONLAR
        if zone_id >= 1:
            f_calc = get_smooth_step(T_eval, T_min_vec[0], 40.0)
            if f_calc > 0 and X[i] < 1.0:
                k_calc = k0_vec[0] * np.exp(-Ea_vec[0] / (R_const * T_eval))
                rates[0, i] = min(k_calc * (1.0 - X[i]) * f_calc, 0.05)

        if zone_id >= 2:
            f_c2s = get_smooth_step(T_eval, T_min_vec[1], 50.0)
            if f_c2s > 0 and m_CaO[i] > 1e-5 and m_SiO2[i] > 1e-5:
                k_c2s = k0_vec[1] * np.exp(-Ea_vec[1] / (R_const * T_eval))
                rates[1, i] = min(k_c2s * m_CaO[i] * m_SiO2[i], 0.040) * f_c2s

        if zone_id == 3:
            f_c3s = get_smooth_step(T_eval, T_min_vec[2], 60.0)
            if f_c3s > 0 and m_C2S[i] > 1e-5 and m_CaO[i] > 1e-5:
                k_c3s = k0_vec[2] * np.exp(-Ea_vec[2] / (R_const * T_eval))
                rates[2, i] = k_c3s * m_C2S[i] * m_CaO[i] * f_c3s * 2.5

    # 2. ENERJİ TÜREVLERİ
    q_rxn = (rates[0]*dH_vec[0] + rates[1]*dH_vec[1] + rates[2]*dH_vec[2]) * m_dot_s_kg_s
    
    dTs, dTg = np.zeros(N), np.zeros(N)
    # thermal_capacity_s için koruma
    thermal_capacity_s = max(100.0, eff_mass_node * cp_s)
    
    # Taşınım (Upwind)
    dTs[1:] = -v_s * (Ts[1:] - Ts[:-1]) / dz
    
    # Isı Kaynakları
    # Isı kaybı (loss_coeff) sistem ısınana kadar düşük tutulmalı
    q_loss_wall = loss_coeff * (Ts - Tw) * exchange_area_per_dz
    
    # dTs hesaplama ve sınırlama (Overflow koruması)
    raw_dTs = (q_gs_vec + q_rad_net_vec - q_loss_wall - q_rxn) / thermal_capacity_s
    dTs += np.clip(raw_dTs, -1000.0, 1000.0) # Saniyede max 1000K değişim
    
    # Gaz fazı ısı türevleri (Gaz çıkıştan girişe/zıt yöne akar)
    m_dot_g_local = np.maximum(1.0, m_dot_g_inlet_kg_s)
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    dTg -= (q_gs_vec + q_rad_net_vec) / (m_dot_g_local * cp_g)
            
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
        self.max_dt_temp_change = 10.0 # Daha hızlı ısınma için 10K tolerans
        
        # SİSTEMİ ATEŞLEME: Başlangıçta gazı sıcak tanımlıyoruz
        # Burner tarafı (son node) 1800K, Giriş tarafı 800K
        self.state.Tg[:] = np.linspace(800.0, 1800.0, nodes)
        self.state.Tw[:] = 1000.0 # Refrakter başlangıç ısısı

    def _safe_f(self, v): 
        return float(v[0]) if isinstance(v, list) else float(v)

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        dt_v = float(dt)
        
        m_dot_s_kg_s = (feed_rate * 1000.0) / 3600.0 
        m_dot_g_inlet_kg_s = self._safe_f(self.cfg['gas']['nominal_flow']) * (fan_rate / 800.0)
        
        v_s = self.tra.calculate_solid_velocity(kiln_rpm)
        fill_degree = self.tra.get_dynamic_filling_degree(kiln_rpm)
        
        # ISIL ATALET (Düşük tutuldu ki ısınma başlasın)
        # image_947b74'deki nan'ı engellemek için 1.0 yerine 8.0 güvenli bir orta yoldur.
        eff_mass_node = ((m_dot_s_kg_s / max(1e-4, v_s)) * self.dz) * 8.0
        
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - fill_degree)
        exchange_area_dz = diameter * np.pi * self.dz

        # Isı transfer katsayıları
        h_gs = self.en.calculate_convection_coeff(fan_rate)
        # Isınmayı tetiklemek için ısı geçiş alanını 3.0 çarpanı ile artırıyoruz
        q_gs_vec = h_gs * (self.state.Tg - self.state.Ts) * (exchange_area_dz * 3.0)
        q_rad_net_vec = self.en.calculate_radiation_flux(self.state.Tg, self.state.Ts, self.state.Tw, exchange_area_dz)

        # 3. TAŞINIM (Upwind) - %50 Kararlılık Sınırı
        f_red = np.clip(v_s * dt_v / self.dz, 0.0, 0.5)
        params = ['m_SiO2', 'm_CaO', 'm_C2S', 'm_C3S', 'X']
        for p in params:
            if hasattr(self.state, p):
                arr = getattr(self.state, p)
                arr[1:] = (1.0 - f_red) * arr[1:] + f_red * arr[:-1]

        # 4. NUMBA ÇEKİRDEĞİ
        dTs, dTg, rates = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.Tw,
            self.state.X, self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S,
            self.state.m_Al2O3, self.state.m_Fe2O3, self.state.m_C3A, self.state.m_C4AF,
            self.state.rho_g, dt_v, self.dz, v_s, m_dot_s_kg_s, m_dot_g_inlet_kg_s,
            self._safe_f(self.cfg['material']['cp_s']), self._safe_f(self.cfg['gas']['cp_g']),
            np.array([self.kin.dH_calc, self.kin.dH_c2s, self.kin.dH_c3s, -125.0, -100.0]), 
            area_gas, q_gs_vec, q_rad_net_vec, eff_mass_node, exchange_area_dz,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min, 0.0015, 0.0010, 
            0.0, # Loss coeff geçici olarak 0.0 yapıldı (ısınma garantisi için)
            self._safe_f(self.cfg['kiln']['length'])
        )

        # 5. ADAPTİF ZAMAN ADIMI (Floor: 0.01)
        temp_variation = np.max(np.abs(dTs * dt_v))
        step_factor = self.max_dt_temp_change / (temp_variation + 1e-6)
        actual_dt = dt_v * max(0.01, min(1.0, step_factor)) 

        # 6. GÜNCELLEMELER
        self.state.Ts = np.nan_to_num(self.state.Ts + dTs * actual_dt, nan=300.0)
        # Gaz sıcaklığının 800K altına düşmesini engelle
        self.state.Tg = np.clip(self.state.Tg + dTg * actual_dt, 800.0, 2200.0)
        
        # Kütle korumalı güncellemeler
        self.state.X = np.clip(self.state.X + rates[0] * actual_dt, 0.0, 1.0)
        self.state.m_C3S += rates[2] * actual_dt
        self.state.m_C2S += (rates[1] - rates[2] * 0.744) * actual_dt
        self.state.m_SiO2 = np.maximum(0.0, self.state.m_SiO2 - (rates[1] * 0.349) * actual_dt)
        self.state.m_CaO = np.maximum(0.0, self.state.m_CaO + (rates[0] * 0.56 - rates[1] * 0.651 - rates[2] * 0.246) * actual_dt)

        # Giriş Sınır Şartı
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])

        self.total_sim_time += actual_dt
        return actual_dt