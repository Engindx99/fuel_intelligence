import numpy as np
from core.state import KilnState
from numba import njit

@njit
def sigmoid_activation(T, T_center, sharpness=0.03):
    """
    Reaksiyonun 'kapısını' açan yumuşak geçiş fonksiyonu.
    """
    z = sharpness * (T - T_center)
    if z > 20: return 1.0
    if z < -20: return 0.0
    return 1.0 / (1.0 + np.exp(-z))

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

        t_gas_target = self._safe_f(config["gas"]["temp_inlet"])

        self.state.initialize_profiles(
            T_ambient=300.0,
            T_gas_inlet=t_gas_target,
            raw_meal_comp=config["raw_meal_composition"]
        )

        self.state.Tg = np.linspace(500.0, t_gas_target, nodes)

    def _safe_f(self, v):
        return float(v[0]) if isinstance(v, list) else float(v)

    def _get_kinetics_params(self):
        k = self.cfg["kinetics"]
        k0_vec = np.array([k["k0"], k["k0_c2s"], k["k0_c3s"], k["pre_factor_c3a"], k["pre_factor_c4af"]], dtype=np.float64)
        Ea_vec = np.array([k["Ea"], k["Ea_c2s"], k["Ea_c3s"], 0.0, 0.0], dtype=np.float64)
        R = float(self.cfg["kinetics"]["R"])
        return k0_vec, Ea_vec, R

    def solve_step(self, dt, fuel_rate, feed_rate, kiln_rpm, fan_rate):
        dt = min(float(dt), 0.25)
        cp_s = self._safe_f(self.cfg["material"]["cp_s"])
        rho_s = self._safe_f(self.cfg["material"]["rho_s"])
        cp_g = self._safe_f(self.cfg["gas"]["cp_g"])
        diameter = self._safe_f(self.cfg["kiln"]["diameter"])

        # --- AKIŞ VE GEOMETRİ ---
        v_s = self.tra.calculate_solid_velocity(kiln_rpm)
        fill = self.tra.get_dynamic_filling_degree(kiln_rpm)
        area_total = np.pi * (diameter * 0.5) ** 2
        area_solid = area_total * fill
        area_gas = area_total * (1.0 - fill)
        exchange_area = np.pi * diameter * self.dz

        m_dot_s = (feed_rate * 1000.0) / 3600.0
        m_dot_g = self._safe_f(self.cfg["gas"]["nominal_flow"]) * (fan_rate / 800.0)
        fuel_kg_s = (fuel_rate * 1000.0) / 3600.0

        # --- ADVEKSİYON (Upwind Scheme) ---
        cfl = np.clip(v_s * dt / self.dz, 0.0, 0.2)
        fields = ["CaCO3", "CaO", "SiO2", "Al2O3", "Fe2O3", "C2S", "C3S", "C3A", "C4AF"]
        for f in fields:
            arr = getattr(self.state, f)
            arr[1:] = (1 - cfl) * arr[1:] + cfl * arr[:-1]
        self.state.Ts[1:] = (1 - cfl) * self.state.Ts[1:] + cfl * self.state.Ts[:-1]

        # --- GAZ TAŞINIMI ---
        gas_vel = m_dot_g / (np.maximum(0.1, self.state.rho_g) * area_gas)
        gas_cfl = np.clip(gas_vel * dt / self.dz, 0.0, 0.2)
        
        self.state.Tg[:-1] = (1 - gas_cfl[:-1]) * self.state.Tg[:-1] + gas_cfl[:-1] * self.state.Tg[1:]
        self.state.CO2[:-1] = (1 - gas_cfl[:-1]) * self.state.CO2[:-1] + gas_cfl[:-1] * self.state.CO2[1:]
        
        # --- EXPONENTIAL RAMP: Gaz Giriş Sıcaklığı ---
        target_tg_in = self._safe_f(self.cfg["gas"]["temp_inlet"])
        start_tg_in = 500.0
        time_constant = 16.0 * 3600.0 
        
        self.state.Tg[-1] = start_tg_in + (target_tg_in - start_tg_in) * (1.0 - np.exp(-self.total_sim_time / time_constant))

        # --- KİNETİK VE SIGMOID AKTİVASYON ---
        k0_vec, Ea_vec, R = self._get_kinetics_params()
        
        act_calc = np.array([sigmoid_activation(t, 1173.0, 0.04) for t in self.state.Ts])
        act_c2s  = np.array([sigmoid_activation(t, 1220.0, 0.03) for t in self.state.Ts])
        act_c3s  = np.array([sigmoid_activation(t, 1450.0, 0.05) for t in self.state.Ts])
        act_liq  = np.array([sigmoid_activation(t, 1350.0, 0.03) for t in self.state.Ts])
        
        activations = np.vstack((act_calc, act_c2s, act_c3s, act_liq, act_liq))

        rates = self.kin(self.state.Ts, self.state.CaCO3, self.state.CaO, self.state.SiO2, 
                         self.state.C2S, self.state.Al2O3, self.state.Fe2O3, self.state.C3A, 
                         self.state.C4AF, k0_vec, Ea_vec, R, activations, dt)

        # --- ISI TRANSFERİ (İyileştirilmiş) ---
        enthalpies = np.array([1780e3, -590e3, 500e3, -120e3, -100e3])
        dH_total = np.sum(rates[:5] * enthalpies[:, np.newaxis], axis=0) * m_dot_s 
        
        # Gaz kütlesi için daha kararlı alt limit (m_dot_g/dt ilişkisi göz önüne alınarak)
        eff_mass_s = np.maximum(50.0, area_solid * self.dz * rho_s)
        eff_mass_g = np.maximum(2.0, area_gas * self.dz * self.state.rho_g)
        
        q_conv = self.en.calculate_convective_flux(self.state.Tg, self.state.Ts, exchange_area, self.en.calculate_convection_coeff(fan_rate))
        q_rad = self.en.calculate_radiation_flux(self.state.Tg, self.state.Ts, self.state.Tw, exchange_area)
        q_combustion = self.en.calculate_combustion_source(fuel_kg_s, self.state.N)
        
        # --- ENERJİ DENKLEMLERİ (Fiziksel Kararlılık ve Limitler) ---
        # Katı Sıcaklık: Isı transferi (+) , Reaksiyonlar (Isı alan +dH ise düşürür)
        q_net_s = q_conv + q_rad - dH_total
        dT_s = (q_net_s / (eff_mass_s * cp_s)) * dt
        self.state.Ts += np.clip(dT_s, -80, 80)
        
        # Gaz Sıcaklık: Yanma (+) , Isı kaybı (-)
        q_net_g = q_combustion - q_conv - q_rad
        dT_g = (q_net_g / (eff_mass_g * cp_g)) * dt
        self.state.Tg += np.clip(dT_g, -120, 120)

        # --- NÜMERİK KORUMA (Yama değil, Fiziksel Sınır) ---
        # Gaz sıcaklığı asla çevre sıcaklığının altına düşmemeli
        self.state.Tg = np.maximum(self.state.Tg, 300.0)
        # Katı sıcaklığının gazı aşırı geçmesini (ekzotermik patlama) makul düzeyde tut
        self.state.Ts = np.minimum(self.state.Ts, self.state.Tg + 100.0)
        
        # --- KİMYASAL GÜNCELLEME (Kütle Korumalı) ---
        r = rates * dt
        
        r0 = np.minimum(r[0], self.state.CaCO3)
        self.state.CaCO3 -= r0
        self.state.CaO   += r0 * 0.5608
        self.state.CO2   += r0 * 0.4392

        r1_pot = np.minimum(r[1], self.state.SiO2)
        r1 = np.minimum(r1_pot, self.state.CaO / 1.8668) 
        self.state.CaO   -= r1 * 1.8668
        self.state.SiO2  -= r1
        self.state.C2S   += r1 * 2.8668

        r3 = np.minimum(np.minimum(r[3], self.state.Al2O3), self.state.CaO / 1.648)
        self.state.Al2O3 -= r3
        self.state.CaO   -= r3 * 1.648
        self.state.C3A   += r3 * 2.648

        r4 = np.minimum(np.minimum(r[4], self.state.Fe2O3), self.state.CaO / 1.402)
        self.state.Fe2O3 -= r4
        self.state.CaO   -= r4 * 1.402
        self.state.C4AF  += r4 * 2.402

        r2 = np.minimum(np.minimum(r[2], self.state.C2S), self.state.CaO / 0.3256)
        self.state.C2S -= r2
        self.state.CaO -= r2 * 0.3256
        self.state.C3S += r2 * 1.3256

        # --- SINIR KOŞULLARI VE TEMİZLİK ---
        raw = self.cfg["raw_meal_composition"]
        self.state.CaCO3[0], self.state.SiO2[0] = raw["CaCO3"], raw["SiO2"]
        self.state.Al2O3[0], self.state.Fe2O3[0] = raw["Al2O3"], raw["Fe2O3"]
        self.state.Ts[0] = self._safe_f(self.cfg["material"]["temp_inlet"])
        
        for p in ["CaO", "CO2", "C2S", "C3S", "C3A", "C4AF"]:
            getattr(self.state, p)[0] = 0.0

        for arr_name in fields + ["CO2"]:
            arr = getattr(self.state, arr_name)
            np.maximum(arr, 0.0, out=arr)

        self.total_sim_time += dt
        return dt