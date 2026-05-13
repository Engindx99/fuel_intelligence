import numpy as np
from numba import njit

# ============================================================== 
# NUMBA KERNEL: Fiziksel Limitli ve Kararlı Çekirdek (UPDATED)
# ==============================================================

@njit(fastmath=True, cache=True)
def _physics_step_core(
    Ts, Tg, CaCO3, CaO, SiO2, C2S, C3S, Al2O3, Fe2O3, C3A, C4AF, CO2,
    dt, dz, v_s, gas_vel, m_dot_s_in, area_solid, area_gas,
    rho_s, cp_s, cp_g, exchange_area, h_conv,
    k0_v, Ea_v, R_gas, enthalpies, fuel_source
):
    nodes = Ts.shape[0]
    sub_steps = 50
    dt_sub = dt / sub_steps
    eps = 1e-12

    for _ in range(sub_steps):
        # --- 1. TAŞINIM (ADVECTION) ---
        cfl_s = np.minimum(v_s * dt_sub / dz, 0.2)
        for i in range(nodes - 1, 0, -1):
            Ts[i] = (1 - cfl_s) * Ts[i] + cfl_s * Ts[i-1]
            CaCO3[i] = (1 - cfl_s) * CaCO3[i] + cfl_s * CaCO3[i-1]
            CaO[i]   = (1 - cfl_s) * CaO[i]   + cfl_s * CaO[i-1]
            SiO2[i]  = (1 - cfl_s) * SiO2[i]  + cfl_s * SiO2[i-1]
            C2S[i]   = (1 - cfl_s) * C2S[i]   + cfl_s * C2S[i-1]
            C3S[i]   = (1 - cfl_s) * C3S[i]   + cfl_s * C3S[i-1]
            C3A[i]   = (1 - cfl_s) * C3A[i]   + cfl_s * C3A[i-1]
            C4AF[i]  = (1 - cfl_s) * C4AF[i]  + cfl_s * C4AF[i-1]
            Fe2O3[i] = (1 - cfl_s) * Fe2O3[i] + cfl_s * Fe2O3[i-1]
            Al2O3[i] = (1 - cfl_s) * Al2O3[i] + cfl_s * Al2O3[i-1]

        # Gaz taşınımı: Çıkıştan (nodes-1) girişe (0) - Karşıt Akış
        for i in range(nodes - 1):
            cfl_g = np.minimum(gas_vel[i] * dt_sub / dz, 0.2)
            Tg[i] = (1 - cfl_g) * Tg[i] + cfl_g * Tg[i+1]

        # --- 2. REAKSİYON VE ENERJİ DENGESİ ---
        for i in range(nodes):
            T_s_curr = max(300.0, Ts[i])
            T_g_curr = max(300.0, Tg[i])
            inv_rt = 1.0 / (R_gas * T_s_curr + eps)

            # --- KİNETİK HESAPLAMALAR ---
            r0 = 0.0
            if T_s_curr > 850.0 and CaCO3[i] > 1e-5:
                arg = max(-60.0, -Ea_v[0] * inv_rt)
                r0 = min(0.005, k0_v[0] * np.exp(arg) * CaCO3[i])

            r1 = 0.0
            if T_s_curr > 1000.0:
                arg = max(-60.0, -Ea_v[1] * inv_rt)
                r1 = min(0.002, k0_v[1] * np.exp(arg) * CaO[i] * SiO2[i])

            r2 = 0.0
            if T_s_curr > 1300.0:
                arg = max(-60.0, -Ea_v[2] * inv_rt)
                r2 = min(0.001, k0_v[2] * np.exp(arg) * C2S[i] * CaO[i])

            # --- ISI AKILARI ---
            q_conv = h_conv * (T_g_curr - T_s_curr) * exchange_area
            
            temp_term = T_s_curr**4 - 320.0**4
            q_loss = 5.67e-8 * 0.9 * exchange_area * max(0.0, temp_term) * 0.1
            
            q_rxn = (r0 * enthalpies[0] + r1 * enthalpies[1] + r2 * enthalpies[2]) * m_dot_s_in

            # --- TERMAL KÜTLE VE KAYNAKLAR ---
            m_cp_s = max(1000.0, area_solid * dz * rho_s * cp_s)
            m_cp_g = max(500.0, area_gas * dz * 1.2 * cp_g)

            # Alev Kaynağı Isıl Verim Kontrolü
            eff_fuel = max(0.0, fuel_source[i]) * max(0.0, (2500.0 - T_g_curr) / 2500.0)

            # --- SICAKLIK GÜNCELLEME (Stability Limited) ---
            dT_s = ((q_conv - q_rxn - q_loss) / m_cp_s) * dt_sub
            dT_g = ((eff_fuel - q_conv) / m_cp_g) * dt_sub

            lim = 0.5 * dt_sub # Max 0.5K per sub-step
            Ts[i] += max(-lim, min(lim, dT_s))
            Tg[i] += max(-lim * 2, min(lim * 2, dT_g))

            # --- KÜTLE GÜNCELLEME ---
            r0_dt = min(CaCO3[i], r0 * dt_sub)
            CaCO3[i] -= r0_dt
            CaO[i]   += r0_dt * 0.56
            CO2[i]   += r0_dt * 0.44

            if CaO[i] > eps:
                dr1 = min(CaO[i]*0.5, r1 * dt_sub)
                CaO[i]  -= dr1 * 0.5
                SiO2[i] -= dr1 * 0.35
                C2S[i]  += dr1
                
                if C2S[i] > eps:
                    dr2 = min(C2S[i], r2 * dt_sub)
                    C2S[i] -= dr2
                    CaO[i] -= dr2 * 0.5
                    C3S[i] += dr2

    return Ts, Tg, CaCO3, CaO, SiO2, C2S, C3S, CO2

# ============================================================== 
# STATE
# ==============================================================

class KilnState:
    def __init__(self, N):
        self.N = N
        self.Ts = np.full(N, 300.0)
        self.Tg = np.full(N, 300.0)
        self.Tw = np.full(N, 300.0)

        self.CaCO3 = np.zeros(N)
        self.CaO   = np.zeros(N)
        self.SiO2  = np.zeros(N)
        self.Al2O3 = np.zeros(N)
        self.Fe2O3 = np.zeros(N)
        self.C2S   = np.zeros(N)
        self.C3S   = np.zeros(N)
        self.C3A   = np.zeros(N)
        self.C4AF  = np.zeros(N)
        self.CO2   = np.zeros(N)

        self._t_hist = []
        self._Ts_hist = []
        self._Tg_hist = []

    def snapshot(self, t):
        self._t_hist.append(t)
        self._Ts_hist.append(self.Ts.copy())
        self._Tg_hist.append(self.Tg.copy())

    def initialize_profiles(self, config):
        feed = config.get("feed", {})
        self.CaCO3.fill(feed.get("CaCO3", 0.82))
        self.SiO2.fill(feed.get("SiO2", 0.14))
        self.Al2O3.fill(feed.get("Al2O3", 0.02))
        self.Fe2O3.fill(feed.get("Fe2O3", 0.02))

        self.Ts.fill(300.0)
        self.Tg.fill(300.0)

# ============================================================== 
# SOLVER
# ==============================================================

class KilnSolver:
    def __init__(self, config, kinetics_fn, transport, energy):
        self.cfg = config
        self.tra = transport
        self.en = energy

        self.nodes = int(self._extract_val(config["kiln"]["nodes"]))
        self.state = KilnState(self.nodes)
        self.length = self._extract_val(config["kiln"]["length"])
        self.dz = self.length / self.nodes
        self.elapsed_time = 0.0
        
        self.target_temp_inlet = float(config["gas"].get("temp_inlet", 2200.0))

    def _extract_val(self, x):
        if isinstance(x, list):
            return float(x[0])
        return float(x)

    def _calculate_exponential_flame(self, fuel_rate, current_time):
        x = np.linspace(0, self.length, self.nodes)
        L_flame = self.length * 0.15
        dist = self.length - x
        flame = np.exp(-dist / L_flame)
        
        # Üstel Rampa Katsayısı [0.0 - 1.0]
        # alpha(t) = 1 - exp(-t/tau)
        ramp_alpha = 1.0 - np.exp(-current_time / 3e4)
        
        lhv = float(self.cfg["gas"]["lhv_fuel"])
        
        # Yakıt enerji girdisi rampa ile çarpılır
        total_energy = fuel_rate * lhv * ramp_alpha
        
        energy_dist = (flame / np.sum(flame)) * total_energy
        return energy_dist, ramp_alpha

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        s = self.state
        k = self.cfg["kinetics"]
        self.elapsed_time += dt

        k0_v = np.array([k["k0"], k["k0_c2s"], k["k0_c3s"]], dtype=np.float64)
        Ea_v = np.array([k["Ea"], k["Ea_c2s"], k["Ea_c3s"]], dtype=np.float64)
        enthalpies = np.array([1780e3, -590e3, 500e3], dtype=np.float64)

        d = self._extract_val(self.cfg["kiln"]["diameter"])
        A = np.pi * (d/2)**2
        fill = self.tra.get_dynamic_filling_degree(kiln_rpm, feed_rate)
        
        # Alev enerjisi ve rampa katsayısını hesapla
        fuel_profile, alpha = self._calculate_exponential_flame(fuel_rate, self.elapsed_time)

        # Sınır Koşulu Güncelleme: 
        # Gaz giriş sıcaklığı (alev ucu) rampa ile 300K'den hedefe yükselir
        s.Tg[-1] = 300.0 + (self.target_temp_inlet - 300.0) * alpha
        s.Ts[0] = 300.0

        res = _physics_step_core(
            s.Ts, s.Tg, s.CaCO3, s.CaO, s.SiO2, s.C2S, s.C3S,
            s.Al2O3, s.Fe2O3, s.C3A, s.C4AF, s.CO2,
            dt, self.dz,
            self.tra.calculate_solid_velocity(kiln_rpm),
            np.full(self.nodes, 6.5 + fan_rate/200.0),
            (feed_rate*1000.0)/3600.0,
            A*fill, A*(1.0-fill),
            self._extract_val(self.cfg["material"]["rho_s"]),
            self._extract_val(self.cfg["material"]["cp_s"]),
            1050.0,
            np.pi*d*self.dz,
            18.0 + 0.05*fan_rate,
            k0_v, Ea_v, 8.314, enthalpies,
            fuel_profile
        )

        (s.Ts, s.Tg, s.CaCO3, s.CaO, s.SiO2, s.C2S, s.C3S, s.CO2) = res
        s.snapshot(self.elapsed_time)
        return dt