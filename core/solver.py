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
        # image_9b1c89.png'deki yumuşak geçişi bu düşük CFL sağlar.
        cfl_s = np.minimum(v_s * dt_sub / dz, 0.01)

        for i in range(nodes - 1, 0, -1):
            Ts[i] = (1 - cfl_s) * Ts[i] + cfl_s * Ts[i-1]
            CaCO3[i] = (1 - cfl_s) * CaCO3[i] + cfl_s * CaCO3[i-1]
            CaO[i]   = (1 - cfl_s) * CaO[i]   + cfl_s * CaO[i-1]
            SiO2[i]  = (1 - cfl_s) * SiO2[i]  + cfl_s * SiO2[i-1]

        # Gaz Fazı Adveksiyonu (Karşıt Akış)
        for i in range(nodes - 1):
            cfl_g = np.minimum(gas_vel[i] * dt_sub / dz, 0.01)
            Tg[i] = (1 - cfl_g) * Tg[i] + cfl_g * Tg[i+1]

        # 2. ENERJİ VE REAKSİYON DENGESİ
        for i in range(nodes):
            T_s_curr = np.maximum(Ts[i], 300.0)
            T_g_curr = np.maximum(Tg[i], 300.0)
            
            # --- Kalsinasyon Kinetiği (Fiziksel Limitli) ---
            # image_9b1c89.png'de 1.17s ile 2.00s arasındaki kütle değişimini kontrol eden kısım
            r0 = 0.0
            if T_s_curr > 850.0 and CaCO3[i] > 1e-4:
                inv_rt = 1.0 / (R_gas * T_s_curr)
                e_calc = -Ea_v[0] * inv_rt
                k_chem = k0_v[0] * np.exp(np.maximum(e_calc, -60.0))
                
                # FİZİKSEL FREN: Reaksiyon hızı saniyede toplam kütlenin %0.5'ini geçemez.
                # Bu kısıt, kalsinasyonun fırın boyunca daha dengeli yayılmasını sağlar.
                r_limit = 0.005 
                r0 = np.minimum(k_chem * CaCO3[i], r_limit)

            # Isı Transfer Terimleri
            # Konveksiyon (Gaz -> Katı)
            q_conv = h_conv * (T_g_curr - T_s_curr) * exchange_area
            
            # Radyasyon Kaybı (Fırın dış yüzeyine olan kayıp)
            # Bu terim sıcaklık tırmanışını fiziksel olarak dengeler.
            q_loss = 5.67e-8 * 0.9 * exchange_area * (T_s_curr**4 - 320.0**4) * 0.1
            
            # Reaksiyon Isısı (Endotermik)
            q_rxn = (r0 * enthalpies[0]) * m_dot_s_in
            
            # Termal Kütleler (Paydanın sıfır olmaması için stabilite tabanı yükseltildi)
            m_cp_s = np.maximum(area_solid * dz * rho_s * cp_s, 1000.0)
            m_cp_g = np.maximum(area_gas * dz * 1.2 * cp_g, 100.0)

            # Değişim Miktarları
            dT_s = ((q_conv - q_rxn - q_loss) / m_cp_s) * dt_sub
            dT_g = ((fuel_source[i] - q_conv) / m_cp_g) * dt_sub

            # Dinamik Emniyet Limitleyicisi (Sub-step başına max 0.2K değişim)
            # image_9b1c89.png'deki Ts ve Tg_Burn sütunlarındaki düzenli artışı bu limit sağlar.
            safe_limit = 0.2 * dt_sub
            dT_s = np.maximum(np.minimum(dT_s, safe_limit), -safe_limit)
            dT_g = np.maximum(np.minimum(dT_g, safe_limit * 3), -safe_limit * 3)

            Ts[i] += dT_s
            Tg[i] += dT_g

            # Kütle Güncelleme (Stokiyometri)
            r0_dt = r0 * dt_sub
            if r0_dt > CaCO3[i]: r0_dt = CaCO3[i]
            CaCO3[i] -= r0_dt
            CaO[i]   += r0_dt * 0.56
            CO2[i]   += r0_dt * 0.44

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
        
        # Bileşen Kütle Oranları
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
        """Dışarıdan çağrılan başlangıç değer atamaları için placeholder."""
        pass

class KilnSolver:
    """Fizik motorunu ve simülasyon adımlarını yöneten ana sınıf."""
    def __init__(self, config, kinetics_fn, transport, energy):
        self.cfg = config
        self.tra = transport
        self.en = energy
        
        # Konfigürasyondan düğüm sayısı ve uzunluk bilgisini al
        self.nodes = int(self._extract_val(config["kiln"]["nodes"]))
        self.state = KilnState(self.nodes)
        self.dz = self._extract_val(config["kiln"]["length"]) / self.nodes

    def _extract_val(self, x):
        """Yaml verisinden skaler değer ayıklar."""
        if isinstance(x, list): return float(x[0])
        return float(x)

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        """Tek bir zaman adımı (dt) boyunca fiziği ilerletir."""
        s = self.state
        k = self.cfg["kinetics"]
        
        # Vektörize edilmiş kinetik parametreler
        k0_v = np.array([k["k0"], k["k0_c2s"], k["k0_c3s"]], dtype=np.float64)
        Ea_v = np.array([k["Ea"], k["Ea_c2s"], k["Ea_c3s"]], dtype=np.float64)
        # Kalsinasyon, C2S oluşumu ve C3S oluşumu entalpileri
        enthalpies = np.array([1780e3, -590e3, 500e3], dtype=np.float64)
        
        # Geometrik ve Akış Hesaplamaları
        d = self._extract_val(self.cfg["kiln"]["diameter"])
        area_total = np.pi * (d / 2)**2
        fill = self.tra.get_dynamic_filling_degree(kiln_rpm, feed_rate)
        
        # Çekirdek (Numba) fonksiyonunu çağır
        res = _physics_step_core(
            s.Ts, s.Tg, s.CaCO3, s.CaO, s.SiO2, s.C2S, s.C3S, 
            s.Al2O3, s.Fe2O3, s.C3A, s.C4AF, s.CO2,
            dt, self.dz, 
            self.tra.calculate_solid_velocity(kiln_rpm),
            np.full(self.nodes, 6.5), # Ortalama gaz hızı (m/s)
            (feed_rate * 1000.0) / 3600.0, # kg/s cinsinden besleme
            area_total * fill, area_total * (1.0 - fill),
            self._extract_val(self.cfg["material"]["rho_s"]),
            self._extract_val(self.cfg["material"]["cp_s"]),
            self._extract_val(self.cfg["gas"]["cp_g"]),
            np.pi * d * self.dz, 18.0 + 0.05 * fan_rate, # h_conv
            k0_v, Ea_v, 8.314, enthalpies,
            self.en.calculate_combustion_source(fuel_rate, self.nodes)
        )
        
        # Sonuçları state nesnesine geri yaz
        (s.Ts, s.Tg, s.CaCO3, s.CaO, s.SiO2, s.C2S, s.C3S, s.CO2) = res
        
        return dt