import numpy as np
from core.state import KilnState

class KilnSolver:
    def __init__(self, config, kinetics_fn, transport, energy):
        self.cfg = config
        self.kin = kinetics_fn
        self.tra = transport
        self.en = energy

        nodes = int(self._safe_f(config["kiln"]["nodes"]))
        self.state = KilnState(nodes)

        self.dz = self._safe_f(config["kiln"]["length"]) / nodes
        self.total_sim_time = 0.0

        self.state.initialize_profiles(
            T_ambient=300.0,
            T_gas_inlet=2000.0,
            raw_meal_comp=config["raw_meal_composition"]
        )

    def _safe_f(self, v):
        return float(v[0]) if isinstance(v, list) else float(v)

    def _get_kinetics_params(self):
        k = self.cfg["kinetics"]
        k0_vec = np.array([k["k0"], k["k0_c2s"], k["k0_c3s"], k["pre_factor_c3a"], k["pre_factor_c4af"]], dtype=np.float64)
        Ea_vec = np.array([k["Ea"], k["Ea_c2s"], k["Ea_c3s"], 0.0, 0.0], dtype=np.float64)
        T_min_vec = np.array([float(k["T_min_rxn"]), float(k["T_min_c2s"]), float(k["T_min_c3s"]), 1350.0, 1350.0], dtype=np.float64)
        pre_factors = np.array([1.0, 1.0, 1.0, k["pre_factor_c3a"], k["pre_factor_c4af"]], dtype=np.float64)
        return k0_vec, Ea_vec, T_min_vec, pre_factors

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        # CFL kararlılığı için dt kısıtı
        dt = min(float(dt), 0.25)
        cp_s = self._safe_f(self.cfg["material"]["cp_s"])
        cp_g = self._safe_f(self.cfg["gas"]["cp_g"])
        diameter = self._safe_f(self.cfg["kiln"]["diameter"])

        # ------------------------------------------------------
        # FLOW & GEOMETRY
        # ------------------------------------------------------
        v_s = self.tra.calculate_solid_velocity(kiln_rpm)
        fill = self.tra.get_dynamic_filling_degree(kiln_rpm)
        area_total = np.pi * (diameter * 0.5) ** 2
        area_gas = area_total * (1.0 - fill)
        exchange_area = np.pi * diameter * self.dz

        # ------------------------------------------------------
        # MASS FLOW (Reference for Chemistry)
        # ------------------------------------------------------
        m_dot_s = (feed_rate * 1000.0) / 3600.0
        m_dot_g = self._safe_f(self.cfg["gas"]["nominal_flow"]) * (fan_rate / 800.0)

        # ======================================================
        # ADVECTION (SOLID) - Upwind Scheme
        # ======================================================
        cfl = np.clip(v_s * dt / self.dz, 0.0, 0.2)
        fields = ["CaCO3", "CaO", "SiO2", "Al2O3", "Fe2O3", "C2S", "C3S", "C3A", "C4AF"]

        for f in fields:
            arr = getattr(self.state, f)
            arr[1:] = (1 - cfl) * arr[1:] + cfl * arr[:-1]

        self.state.Ts[1:] = (1 - cfl) * self.state.Ts[1:] + cfl * self.state.Ts[:-1]

        # ======================================================
        # GAS FLOW & CO2 TRANSPORT (Counter-current)
        # ======================================================
        gas_vel = m_dot_g / (np.maximum(0.1, self.state.rho_g) * area_gas)
        gas_cfl = np.clip(gas_vel * dt / self.dz, 0.0, 0.2)
        
        # Gaz sıcaklığı ve CO2 gaz fazında olduğu için aynı yönde (çıkışa doğru) taşınır
        self.state.Tg[:-1] = (1 - gas_cfl[:-1]) * self.state.Tg[:-1] + gas_cfl[:-1] * self.state.Tg[1:]
        self.state.CO2[:-1] = (1 - gas_cfl[:-1]) * self.state.CO2[:-1] + gas_cfl[:-1] * self.state.CO2[1:]
        
        # Gaz çıkış sıcaklığı (Basit lineer ısıtma modeli)
        self.state.Tg[-1] = 400.0 + (1 - np.exp(-self.total_sim_time / 28800.0)) * 1600.0

        # ======================================================
        # KINETICS & HEAT TRANSFER
        # ======================================================
        k0_vec, Ea_vec, T_min_vec, pre_factors = self._get_kinetics_params()
        # kinetics r matrisi artık (6, N) boyutunda: r[5] = CO2 üretim hızı
        rates = self.kin(self.state.Ts, self.state.CaCO3, self.state.CaO, self.state.SiO2, 
                         self.state.C2S, self.state.Al2O3, self.state.Fe2O3, self.state.C3A, 
                         self.state.C4AF, k0_vec, Ea_vec, self.cfg["kinetics"]["R"], T_min_vec, pre_factors, dt)

        # Enerji Dengesi (Reaksiyon entalpisi ve Isı transferi)
        dH = np.sum(rates[:5], axis=0) * m_dot_s # İlk 5 reaksiyonun entalpisi
        eff_mass = np.maximum(5.0, (m_dot_s / max(v_s, 1e-4)) * self.dz)
        
        q_conv = self.en.calculate_convective_flux(self.state.Tg, self.state.Ts, exchange_area, self.en.calculate_convection_coeff(fan_rate))
        q_rad = self.en.calculate_radiation_flux(self.state.Tg, self.state.Ts, self.state.Tw, exchange_area)
        
        q_net = q_conv + q_rad - dH
        self.state.Ts += np.clip(q_net / np.maximum(100.0, eff_mass * cp_s), -250, 250) * dt
        
        # ======================================================
        # CHEMISTRY UPDATE (Physically Consistent & Sequential)
        # ======================================================
        r = rates * dt
        
        # 1. KALSİNASYON: CaCO3 -> CaO + CO2 (Kireç Üretimi)
        # MW Oranları: CaO/CaCO3 = 0.5608, CO2/CaCO3 = 0.4392
        r0 = np.minimum(r[0], self.state.CaCO3)
        self.state.CaCO3 -= r0
        self.state.CaO   += r0 * 0.5608
        self.state.CO2   += r0 * 0.4392

        # Kireç tüketen reaksiyonlar için anlık mevcut kireç takibi
        # Not: Reaksiyonlar fırın içindeki önceliğine göre sıralanmıştır.
        
        # 2. C2S (BELİT) OLUŞUMU: 2*CaO + SiO2 -> C2S
        # 1 kg SiO2 tüketmek için 1.8668 kg CaO gerekir.
        r1_potential = np.minimum(r[1], self.state.SiO2)
        r1 = np.minimum(r1_potential, self.state.CaO / 1.8668) 
        
        self.state.CaO   -= r1 * 1.8668
        self.state.SiO2  -= r1
        self.state.C2S   += r1 * 2.8668

        # 3. SIVI FAZLAR (C3A ve C4AF) - Genellikle C3S'ten önce veya eş zamanlı oluşur
        # C3A Oluşumu
        r3_potential = np.minimum(r[3], self.state.Al2O3)
        r3 = np.minimum(r3_potential, self.state.CaO / 1.648)
        self.state.Al2O3 -= r3
        self.state.CaO   -= r3 * 1.648
        self.state.C3A   += r3 * 2.648

        # C4AF Oluşumu
        r4_potential = np.minimum(r[4], self.state.Fe2O3)
        r4 = np.minimum(r4_potential, self.state.CaO / 1.402)
        self.state.Fe2O3 -= r4
        self.state.CaO   -= r4 * 1.402
        self.state.C4AF  += r4 * 2.402

        # 4. C3S (ALİT) OLUŞUMU: C2S + CaO -> C3S
        # Alit oluşumu en son ve en yüksek sıcaklıkta gerçekleşir.
        # 1 kg C2S tüketmek için 0.3256 kg CaO gerekir.
        r2_potential = np.minimum(r[2], self.state.C2S)
        # Alit oluşumu için kireç kalıp kalmadığını kontrol et
        r2 = np.minimum(r2_potential, self.state.CaO / 0.3256)
        
        self.state.C2S -= r2
        self.state.CaO -= r2 * 0.3256
        self.state.C3S += r2 * 1.3256

        # ======================================================
        # BOUNDARY CONDITIONS (Inlet)
        # ======================================================
        raw = self.cfg["raw_meal_composition"]
        
        self.state.CaCO3[0] = raw["CaCO3"]
        self.state.SiO2[0]  = raw["SiO2"]
        self.state.Al2O3[0] = raw["Al2O3"]
        self.state.Fe2O3[0] = raw["Fe2O3"]
        
        # Giriş düğümünde ürünler ve CO2 her zaman temizlenir (Boundary condition)
        self.state.CaO[0]   = 0.0
        self.state.CO2[0]   = 0.0
        self.state.C2S[0]   = 0.0
        self.state.C3S[0]   = 0.0
        self.state.C3A[0]   = 0.0
        self.state.C4AF[0]  = 0.0
        
        self.state.Ts[0] = self._safe_f(self.cfg["material"]["temp_inlet"])

        # Sayısal stabilite: Negatif değerleri engelle
        for arr_name in ["CaCO3", "CaO", "SiO2", "Al2O3", "Fe2O3", "C2S", "C3S", "C3A", "C4AF", "CO2"]:
            arr = getattr(self.state, arr_name)
            np.maximum(arr, 0.0, out=arr)

        self.total_sim_time += dt
        return dt