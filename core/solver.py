import numpy as np
from numba import njit

# ==============================================================
# NUMBA KERNEL: Fiziksel Limitli ve Kararlı Çekirdek
# ==============================================================
@njit(fastmath=True, cache=True)
def _physics_step_core(
    Ts, Tg, CaCO3, CaO, SiO2, C2S, C3S, Al2O3, Fe2O3, C3A, C4AF, CO2,
    dt, dz, v_s, gas_vel, m_dot_s_in, area_solid, area_gas,
    rho_s, cp_s, cp_g, exchange_area, h_conv,
    k0_v, Ea_v, R_gas, enthalpies, fuel_source
):
    nodes = Ts.shape[0]
    # Diferansiyel denklemin sertliğini (stiffness) kırmak için sub-stepping yüksek tutuldu
    sub_steps = 50 
    dt_sub = dt / sub_steps

    for _ in range(sub_steps):
        # 1. ADVEKSİYON (Upwind Fark Şeması)
        # Kararlılık için CFL sınırını %1 ile kısıtlıyoruz (Sayısal gürültüyü önler)
        cfl_s = np.minimum(v_s * dt_sub / dz, 0.01)

        for i in range(nodes - 1, 0, -1):
            Ts[i] = (1 - cfl_s) * Ts[i] + cfl_s * Ts[i-1]
            CaCO3[i] = (1 - cfl_s) * CaCO3[i] + cfl_s * CaCO3[i-1]
            CaO[i]   = (1 - cfl_s) * CaO[i]   + cfl_s * CaO[i-1]
            SiO2[i]  = (1 - cfl_s) * SiO2[i]  + cfl_s * SiO2[i-1]
            C2S[i]   = (1 - cfl_s) * C2S[i]   + cfl_s * C2S[i-1]
            C3S[i]   = (1 - cfl_s) * C3S[i]   + cfl_s * C3S[i-1]

        # Gaz Fazı Adveksiyonu (Karşıt Akış)
        for i in range(nodes - 1):
            cfl_g = np.minimum(gas_vel[i] * dt_sub / dz, 0.01)
            Tg[i] = (1 - cfl_g) * Tg[i] + cfl_g * Tg[i+1]

        # 2. ENERJİ VE REAKSİYON DENGESİ
        for i in range(nodes):
            T_s_curr = np.maximum(Ts[i], 300.0)
            T_g_curr = np.maximum(Tg[i], 300.0)
            inv_rt = 1.0 / (R_gas * T_s_curr)
            
            # --- Kalsinasyon Kinetiği (r0) ---
            r0 = 0.0
            if T_s_curr > 850.0 and CaCO3[i] > 1e-4:
                e_calc = -Ea_v[0] * inv_rt
                k_calc = k0_v[0] * np.exp(np.maximum(e_calc, -60.0))
                r_limit_calc = 0.005 
                r0 = np.minimum(k_calc * CaCO3[i], r_limit_calc)

            # --- Belit (C2S) Oluşumu (r1: 2CaO + SiO2 -> C2S) ---
            r1 = 0.0
            if T_s_curr > 1000.0 and CaO[i] > 1e-4 and SiO2[i] > 1e-4:
                e_c2s = -Ea_v[1] * inv_rt
                k_c2s = k0_v[1] * np.exp(np.maximum(e_c2s, -60.0))
                r_limit_c2s = 0.002
                r1 = np.minimum(k_c2s * CaO[i] * SiO2[i], r_limit_c2s)

            # --- Alit (C3S) Oluşumu (r2: C2S + CaO -> C3S) ---
            r2 = 0.0
            if T_s_curr > 1300.0 and C2S[i] > 1e-4 and CaO[i] > 1e-4:
                e_c3s = -Ea_v[2] * inv_rt
                k_c3s = k0_v[2] * np.exp(np.maximum(e_c3s, -60.0))
                r_limit_c3s = 0.001
                r2 = np.minimum(k_c3s * C2S[i] * CaO[i], r_limit_c3s)

            # Isı Transfer Terimleri
            q_conv = h_conv * (T_g_curr - T_s_curr) * exchange_area
            q_loss = 5.67e-8 * 0.9 * exchange_area * (T_s_curr**4 - 320.0**4) * 0.1
            
            # Toplam Reaksiyon Isısı
            q_rxn = (r0 * enthalpies[0] + r1 * enthalpies[1] + r2 * enthalpies[2]) * m_dot_s_in
            
            # Gaz termal kütlesini stabilize etmek için taban değer yükseltildi
            m_cp_s = np.maximum(area_solid * dz * rho_s * cp_s, 1000.0)
            m_cp_g = np.maximum(area_gas * dz * 1.2 * cp_g, 500.0)

            # Isı kaynağı (fuel_source) alev sıcaklığına bağlı olarak limitleyici içerir
            # Eğer gaz zaten çok sıcaksa enerji transferi azalır (Termodinamik limit)
            eff_fuel_source = fuel_source[i] * np.maximum(0.0, (2500.0 - T_g_curr) / 2500.0)

            dT_s = ((q_conv - q_rxn - q_loss) / m_cp_s) * dt_sub
            dT_g = ((eff_fuel_source - q_conv) / m_cp_g) * dt_sub

            # Dinamik Emniyet Limitleyicisi
            safe_limit = 0.2 * dt_sub
            dT_s = np.maximum(np.minimum(dT_s, safe_limit), -safe_limit)
            dT_g = np.maximum(np.minimum(dT_g, safe_limit * 2), -safe_limit * 2)

            Ts[i] += dT_s
            Tg[i] += dT_g

            # 3. KÜTLE GÜNCELLEME (Stokiyometri)
            r0_dt = r0 * dt_sub
            r1_dt = r1 * dt_sub
            r2_dt = r2 * dt_sub

            if r0_dt > CaCO3[i]: r0_dt = CaCO3[i]
            CaCO3[i] -= r0_dt
            CaO[i]   += r0_dt * 0.56
            CO2[i]   += r0_dt * 0.44

            if CaO[i] > (r1_dt * 0.65 + r2_dt * 0.25) and SiO2[i] > (r1_dt * 0.35):
                CaO[i]  -= (r1_dt * 0.65 + r2_dt * 0.25)
                SiO2[i] -= (r1_dt * 0.35)
                C2S[i]  += (r1_dt - r2_dt)
                C3S[i]  += r2_dt

    return Ts, Tg, CaCO3, CaO, SiO2, C2S, C3S, CO2

# ==============================================================
# STATE VE SOLVER SINIFLARI
# ==============================================================
class KilnState:
    """Fırın içindeki tüm fiziksel değişkenlerin durumunu tutar."""
    def __init__(self, N):
        self.N = N  
        self.Ts = np.full(N, 300.0, dtype=np.float64)
        self.Tg = np.full(N, 1000.0, dtype=np.float64)
        self.Tw = np.full(N, 450.0, dtype=np.float64)
        
        self.CaCO3 = np.zeros(N, dtype=np.float64)
        self.CaO   = np.zeros(N, dtype=np.float64)
        self.SiO2  = np.zeros(N, dtype=np.float64)
        self.Al2O3 = np.zeros(N, dtype=np.float64)
        self.Fe2O3 = np.zeros(N, dtype=np.float64)
        self.C2S   = np.zeros(N, dtype=np.float64)
        self.C3S   = np.zeros(N, dtype=np.float64)
        self.C3A   = np.zeros(N, dtype=np.float64)
        self.C4AF  = np.zeros(N, dtype=np.float64)
        self.CO2   = np.zeros(N, dtype=np.float64)

    def initialize_profiles(self, config):
        """Hata veren eksik metot: Profil başlangıç değerlerini atar."""
        feed = config.get("feed", {})
        self.CaCO3.fill(feed.get("CaCO3", 0.82))
        self.SiO2.fill(feed.get("SiO2", 0.14))
        self.Al2O3.fill(feed.get("Al2O3", 0.02))
        self.Fe2O3.fill(feed.get("Fe2O3", 0.02))
        
        self.Ts.fill(300.0)
        self.Tg.fill(1000.0)

class KilnSolver:
    """Fizik motorunu ve simülasyon adımlarını yöneten ana sınıf."""
    def __init__(self, config, kinetics_fn, transport, energy):
        self.cfg = config
        self.tra = transport
        self.en = energy
        
        self.nodes = int(self._extract_val(config["kiln"]["nodes"]))
        self.state = KilnState(self.nodes)
        self.length = self._extract_val(config["kiln"]["length"])
        self.dz = self.length / self.nodes
        self.elapsed_time = 0.0
        

    def _extract_val(self, x):
        if isinstance(x, list): return float(x[0])
        return float(x)

    def _calculate_exponential_flame(self, fuel_rate, current_time):
        
        x = np.linspace(0, self.length, self.nodes)
        L_flame = self.length * 0.15
        
        dist_from_burner = (self.length - x)
        flame_profile = np.exp(-dist_from_burner / L_flame)
        
        # ramp
        time_scaling = 1.0 - np.exp(-current_time / 1e6)
        
        # Yakıtın alt ısıl değeri üzerinden toplam enerji
        lhv = float(self.cfg["gas"]["lhv_fuel"])
        total_energy = fuel_rate * lhv
        raw_source = (flame_profile / np.sum(flame_profile)) * total_energy * time_scaling
        max_node_energy = 5.0e6 
        flame_source = max_node_energy * np.tanh(raw_source / max_node_energy)
        
        return flame_source

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        s = self.state
        k = self.cfg["kinetics"]
        self.elapsed_time += dt
        
        k0_v = np.array([k["k0"], k["k0_c2s"], k["k0_c3s"]], dtype=np.float64)
        Ea_v = np.array([k["Ea"], k["Ea_c2s"], k["Ea_c3s"]], dtype=np.float64)
        enthalpies = np.array([1780e3, -590e3, 500e3], dtype=np.float64)
        
        d = self._extract_val(self.cfg["kiln"]["diameter"])
        area_total = np.pi * (d / 2)**2
        fill = self.tra.get_dynamic_filling_degree(kiln_rpm, feed_rate)
        
        fuel_source_dist = self._calculate_exponential_flame(fuel_rate, self.elapsed_time)
        
        res = _physics_step_core(
            s.Ts, s.Tg, s.CaCO3, s.CaO, s.SiO2, s.C2S, s.C3S, 
            s.Al2O3, s.Fe2O3, s.C3A, s.C4AF, s.CO2,
            dt, self.dz, 
            self.tra.calculate_solid_velocity(kiln_rpm),
            np.full(self.nodes, 6.5 + (fan_rate / 200.0)),
            (feed_rate * 1000.0) / 3600.0,
            area_total * fill, area_total * (1.0 - fill),
            self._extract_val(self.cfg["material"]["rho_s"]),
            self._extract_val(self.cfg["material"]["cp_s"]),
            1600.0, 
            np.pi * d * self.dz, 18.0 + 0.05 * fan_rate,
            k0_v, Ea_v, 8.314, enthalpies,
            fuel_source_dist
        )
        
        (s.Ts, s.Tg, s.CaCO3, s.CaO, s.SiO2, s.C2S, s.C3S, s.CO2) = res
        return dt