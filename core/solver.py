import numpy as np
from numba import njit
from core.state import KilnState

# --- DERLENMİŞ ÇEKİRDEK (C HIZINDA KOŞAR) ---
@njit
def _numba_step_core(N, Ts, Tg, X, rho_g, dt, dz, v_s, m_dot_s, m_dot_g, 
                     cp_s, cp_g, dH_rxn, area_gas, exchange_area_per_dz, 
                     damping, effective_thermal_mass, h_gs, sigma, eps_eff,
                     k0, Ea, R_const, T_min):
    
    dTs = np.zeros(N)
    dTg = np.zeros(N)
    dX = np.zeros(N)

    for i in range(N):
        # 1. Kinetik: Arrhenius Kalsinasyon Hızı
        r = 0.0
        if Ts[i] >= T_min and X[i] < 1.0:
            k = k0 * np.exp(-Ea / (R_const * Ts[i]))
            r = min(k * (1.0 - X[i]), 0.01) # Sayısal kararlılık sınırı
        dX[i] = r
        
        # 2. Isı Transferi: Konveksiyon + Radyasyon
        q_conv = h_gs * (Tg[i] - Ts[i]) * exchange_area_per_dz
        q_rad = sigma * eps_eff * (Tg[i]**4 - Ts[i]**4) * exchange_area_per_dz
        q_net = q_conv + q_rad
        
        # 3. Katı Faz: Eksenel Taşıma ve Enerji Kaynakları
        if i > 0:
            # v_s (RPM bağımlı) residence time etkisini belirler
            advection_s = v_s * (Ts[i-1] - Ts[i]) / dz
            # Endotermik reaksiyon (- r * dH) ve gazdan gelen ısı
            source_s = (q_net - r * m_dot_s * dH_rxn) / (effective_thermal_mass * cp_s)
            dTs[i] = (advection_s + source_s) * damping
            
        # 4. Gaz Fazı: Karşıt Akış Enerji Dengesi
        if i < N - 1:
            v_g = m_dot_g / (max(0.1, rho_g[i]) * area_gas)
            advection_g = v_g * (Tg[i+1] - Tg[i]) / dz
            dTg[i] = advection_g - q_net / (m_dot_g * cp_g)
            
    return dTs, dTg, dX

# --- SOLVER SINIFI (YÖNETİCİ KATMAN) ---
class KilnSolver:
    def __init__(self, config, kinetics, transport, energy):
        self.cfg = config
        self.kin = kinetics
        self.tra = transport
        self.en  = energy
        
        nodes = int(self._safe_f(config['kiln']['nodes']))
        self.state = KilnState(nodes)
        self.dz = self._safe_f(config['kiln']['length']) / nodes
        self.damping = 0.4 

    def _safe_f(self, value):
        """YAML hiyerarşisinden güvenli float okur."""
        if isinstance(value, list): return float(value[0])
        return float(value)

    def solve_step(self, dt):
        # 1. Dinamik Parametrelerin Güncellenmesi (RPM ve Feed)
        # TransportModel içindeki v_s hesabı güncel RPM'i kullanır
        v_s = self._safe_f(self.tra.calculate_solid_velocity())
        m_dot_s = self._safe_f(self.cfg['material']['feed_rate'])
        m_dot_g = self._safe_f(self.cfg['gas']['flow_rate'])
        
        # Geometrik ve Fiziksel Sabitler
        cp_s = self._safe_f(self.cfg['material']['cp_s'])
        cp_g = self._safe_f(self.cfg['gas']['cp_g'])
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        dH_rxn = self._safe_f(self.kin.dH)
        fill = self._safe_f(self.cfg['kiln'].get('filling_degree', 0.08))
        
        # Isı Transfer Katsayıları
        h_gs = self._safe_f(self.cfg['gas']['h_gs'])
        sigma = 5.67e-8
        eps_eff = self._safe_f(self.cfg['gas']['emissivity_g'])

        # Alan ve Kütle Hesapları
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - fill)
        exchange_area_per_dz = diameter * self.dz 
        
        # RPM arttıkça v_s artar, node_mass azalır (Residence time azalır)
        node_mass = (m_dot_s / max(0.001, v_s)) * self.dz
        effective_thermal_mass = node_mass + 10.0 # Sanal refrakter ataleti

        # Gaz yoğunluğu güncelleme (Statik metot çağrısı)
        self.state.update_gas_density()

        # 2. C Seviyesinde Hesaplama (Numba Call)
        dTs, dTg, dX = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.X, self.state.rho_g,
            float(dt), self.dz, v_s, m_dot_s, m_dot_g, cp_s, cp_g, dH_rxn,
            area_gas, exchange_area_per_dz, self.damping, effective_thermal_mass,
            h_gs, sigma, eps_eff,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min
        )

        # 3. Durum Güncelleme ve Sınır Şartları
        self.state.X  += dX * dt
        self.state.Ts += dTs * dt
        self.state.Tg += dTg * dt
        
        # Giriş Sınır Şartlarını Koru
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        self.state.Tg[-1] = self._safe_f(self.cfg['gas']['temp_inlet'])
        
        # Fiziksel Clipping (Sayısal taşmaları önle)
        self.state.X = np.clip(self.state.X, 0.0, 1.0)
        self.state.Ts = np.clip(self.state.Ts, 300.0, 1900.0)
        self.state.Tg = np.clip(self.state.Tg, 300.0, 2400.0)