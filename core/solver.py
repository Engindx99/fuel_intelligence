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
    
    # Isı transferi boost faktörleri
    dynamic_boost = 1.0 + 0.4 / (1.0 + np.exp(-25 * (x_coords - 0.82)))
    rad_multiplier = 1.05
    
    for i in range(N):
        T_eval = max(300.0, Ts[i])
        
        # 1. KALSİNASYON (Endotermik)
        if x_coords[i] >= 0.20:
            if T_eval >= T_min_vec[0] and X[i] < 1.0:
                arg = -Ea_vec[0] / (R_const * T_eval)
                rates[0, i] = k0_vec[0] * np.exp(max(clip_min, min(clip_max, arg))) * (1.0 - X[i]) 

        # 2. BELİT (C2S) OLUŞUMU - Difüzyon Kontrollü
        # Katı-katı reaksiyonu olduğu için oluşan Belit tabakası difüzyon direnci yaratır.
        if x_coords[i] >= 0.40 and T_eval >= T_min_vec[1]:
            # Ginstling-Brounshtein tipi difüzyon direnci (m_C2S arttıkça hız düşer)
            diffusion_inhibition = np.exp(-4.5 * m_C2S[i])
            # Hammadde mevcudiyet doygunluğu
            chemical_driving_force = m_CaO[i] * m_SiO2[i]
            
            k_c2s = k0_vec[1] * np.exp(max(clip_min, min(clip_max, -Ea_vec[1]/(R_const*T_eval))))
            rates[1, i] = k_c2s * chemical_driving_force * diffusion_inhibition

        # 3. ALİT (C3S) OLUŞUMU - Sıvı Faz Hareketliliği Kontrollü
        # Alit oluşumu için C3A ve C4AF kaynaklı sıvı faz (melt) elzemdir.
        if x_coords[i] >= 0.65 and T_eval >= T_min_vec[2]:
            liquid_content = m_C3A[i] + m_C4AF[i]
            # Sıvı faz eşiği: Sıvı faz %5'in altındaysa reaksiyon kinetiği çok yavaştır.
            # Bu "yapay" değil, kimyasal bir difüzyon kısıtıdır.
            liquid_mobility = 1.0 / (1.0 + np.exp(-30.0 * (liquid_content - 0.05)))
            
            # Reaksiyon hızı Belit ve CaO konsantrasyonuna bağlıdır.
            k_c3s = k0_vec[2] * np.exp(max(clip_min, min(clip_max, -Ea_vec[2]/(R_const*T_eval))))
            rates[2, i] = k_c3s * m_C2S[i] * m_CaO[i] * liquid_mobility * burning_zone_factor

        # 4. YARDIMCI FAZLAR (C3A & C4AF) - Flux/Sıvı Faz Oluşumu
        if x_coords[i] >= 0.55:
            if T_eval >= 1250.0 and m_Al2O3[i] > 1e-6:
                rates[3, i] = pre_factors[0] * m_Al2O3[i]
            if T_eval >= 1200.0 and m_Fe2O3[i] > 1e-6:
                rates[4, i] = pre_factors[1] * m_Fe2O3[i]

    # Enerji Dengesi (EMA ile kararlılık sağlanıyor)
    q_rxn_instant = (rates[0]*dH_vec[0] + rates[1]*dH_vec[1] + rates[2]*dH_vec[2]) * m_dot_s_kg_s
    q_rxn = 0.80 * q_rxn_prev + 0.20 * q_rxn_instant

    thermal_capacity_s = np.maximum(10.0, eff_mass_node * cp_s) 
    dTs, dTg = np.zeros(N), np.zeros(N)
    
    # Katı Sıcaklık Değişimi (Advection + Heat Transfer - Reaction)
    dTs[1:] = -v_s * (Ts[1:] - Ts[:-1]) / dz
    q_loss_wall = loss_coeff * (Ts - Tw) * exchange_area_per_dz
    rad_amp = np.where(Ts > 1450.0, 1.0 + (Ts - 1450.0) / 400.0, 1.0)
    
    net_heat_to_s = (dynamic_boost * (q_gs_vec + q_rad_net_vec * rad_multiplier * rad_amp)) - q_loss_wall - q_rxn
    dTs += np.clip(net_heat_to_s / thermal_capacity_s, -4000.0, 4000.0) 
    
    # Gaz Sıcaklık Değişimi
    m_dot_g_local = np.maximum(0.5, m_dot_g_inlet_kg_s)
    v_g = m_dot_g_local / (np.maximum(0.1, rho_g) * area_gas)
    dTg[:-1] = (v_g[:-1] * (Tg[1:] - Tg[:-1]) / dz)
    dTg += np.clip(-(q_gs_vec + q_rad_net_vec) / (m_dot_g_local * cp_g), -7000.0, 7000.0)
            
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
        
        # Başlangıç Koşulları
        self.state.Tg = np.full(nodes, 400.0)
        self.state.Ts = np.full(nodes, 300.0)
        self.state.Tw = np.full(nodes, 300.0)

    def _safe_f(self, v): 
        return float(v[0]) if isinstance(v, list) else float(v)

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        dt_v = min(float(dt), 0.10) # Sayısal kararlılık için CFL kısıtı
        sim_hour = self.total_sim_time / 3600.0 
        
        # Termal Rampa (Kararlı rejime geçiş)
        ramp_factor = 1.0 - np.exp(-sim_hour / 10.0) 
        nodes = self.state.N
        x = np.linspace(0, 1, nodes)
        target_Tg = 450 + 1850 * np.exp(-(x-1.0)**2 / 0.12)
        target_Tw = 350 + 900 * x 

        self.state.Tg = 400.0 * (1 - ramp_factor) + target_Tg * ramp_factor
        self.state.Tw = 300.0 * (1 - ramp_factor) + target_Tw * ramp_factor

        # --- ADVECTION (Kütle Taşınımı) ---
        v_s = self.tra.calculate_solid_velocity(kiln_rpm)
        f_red = np.clip(v_s * dt_v / self.dz, 0.0, 0.45)
        params = ['m_CaO', 'm_C2S', 'm_C3S', 'X', 'm_Al2O3', 'm_Fe2O3', 'm_C3A', 'm_C4AF', 'm_SiO2']
        for p in params:
            arr = getattr(self.state, p)
            arr[1:] = (1.0 - f_red) * arr[1:] + f_red * arr[:-1]
        
        self.m_CaO_pool[1:] = (1.0 - f_red) * self.m_CaO_pool[1:] + f_red * self.m_CaO_pool[:-1]
        self.q_rxn_ema[1:] = (1.0 - f_red) * self.q_rxn_ema[1:] + f_red * self.q_rxn_ema[:-1]

        # --- PHYSICS & REACTION CORE ---
        m_dot_s_kg_s = (feed_rate * 1000.0) / 3600.0 
        m_dot_g_inlet_kg_s = self._safe_f(self.cfg['gas']['nominal_flow']) * (fan_rate / 800.0)
        
        fill_degree = self.tra.get_dynamic_filling_degree(kiln_rpm)
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - fill_degree)
        exchange_area_dz = diameter * np.pi * self.dz
        
        h_gs = self.en.calculate_convection_coeff(fan_rate)
        q_gs_vec = h_gs * (self.state.Tg - self.state.Ts) * exchange_area_dz
        q_rad_net_vec = self.en.calculate_radiation_flux(self.state.Tg, self.state.Ts, self.state.Tw, exchange_area_dz)

        # Numba Call
        dTs, dTg, rates, self.q_rxn_ema = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.Tw,
            self.state.X, self.state.m_CaO, self.state.m_SiO2, self.state.m_C2S,
            self.state.m_Al2O3, self.state.m_Fe2O3, self.state.m_C3A, self.state.m_C4AF,
            self.state.rho_g, dt_v, self.dz, v_s, m_dot_s_kg_s, m_dot_g_inlet_kg_s,
            self._safe_f(self.cfg['material']['cp_s']), self._safe_f(self.cfg['gas']['cp_g']),
            self.kin.get_enthalpy_vector(), area_gas, q_gs_vec, q_rad_net_vec, 
            max(0.5, (m_dot_s_kg_s / max(1e-4, v_s)) * self.dz), exchange_area_dz,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min,
            self.kin.pre_factors, 0.05, -80.0, 25.0, self.kin.burning_zone_factor,
            self.q_rxn_ema
        )

        self.state.Ts += dTs * dt_v
        self.state.Ts = np.clip(self.state.Ts, 300.0, 1900.0)

        # --- KİMYASAL STOK GÜNCELLEMELERİ ---
        delta_X = rates[0] * dt_v
        self.state.X = np.clip(self.state.X + delta_X, 0.0, 1.0)
        
        caco3_frac = self._safe_f(self.cfg['raw_meal_composition']['CaCO3'])
        produced_CaO = delta_X * caco3_frac * 0.56 
        self.m_CaO_pool += produced_CaO

        # Flux fazları (Sıvı Faz oluşumu)
        r_c3a = rates[3] * dt_v
        r_c4af = rates[4] * dt_v
        self.state.m_C3A += r_c3a
        self.state.m_C4AF += r_c4af
        self.state.m_Al2O3 = np.maximum(0.0, self.state.m_Al2O3 - r_c3a)
        self.state.m_Fe2O3 = np.maximum(0.0, self.state.m_Fe2O3 - r_c4af)

        # Klinkere Dönüşüm (Kimyasal Sınırlandırmalı)
        # CaO Pool'dan ana kütleye geçiş (Dissolution hızı)
        release_rate = self.m_CaO_pool * 0.5
        self.state.m_CaO += release_rate
        self.m_CaO_pool -= release_rate

        # Reaksiyon miktarları
        r_c2s = rates[1] * dt_v
        r_c3s = rates[2] * dt_v

        # Kütle Korunumu: Stokiyometrik Limitler
        # C2S oluşumu için SiO2 ve CaO kısıtı
        r_c2s_final = np.minimum(r_c2s, self.state.m_SiO2 / 0.28)
        r_c2s_final = np.minimum(r_c2s_final, self.state.m_CaO / 0.65)

        # C3S (Alit) oluşumu için Belit ve CaO kısıtı
        r_c3s_final = np.minimum(r_c3s, self.state.m_C2S / 0.75)
        r_c3s_final = np.minimum(r_c3s_final, self.state.m_CaO / 0.25)

        # Kütlelerin nihai güncellenmesi
        self.state.m_C3S += r_c3s_final
        self.state.m_C2S += (r_c2s_final - (r_c3s_final * 0.75))
        self.state.m_SiO2 = np.maximum(0.0, self.state.m_SiO2 - (r_c2s_final * 0.28))
        self.state.m_CaO = np.maximum(1e-8, self.state.m_CaO - (r_c2s_final * 0.65) - (r_c3s_final * 0.25))
        
        # Inlet Boundary
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        self.total_sim_time += dt_v
        
        return dt_v