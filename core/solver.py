import numpy as np
from numba import njit
import os
import yaml
from core.state import KilnState

@njit
def _numba_step_core(N, Ts, Tg, X, rho_g, dt, dz, v_s, m_dot_s, m_dot_g, 
                     cp_s, cp_g, dH_rxn, area_gas, exchange_area_per_dz, 
                     damping, effective_thermal_mass, h_gs, sigma, eps_eff,
                     k0, Ea, R_const, T_min):
    """C hızında çalışan yüksek performanslı fizik çekirdeği."""
    dTs = np.zeros(N)
    dTg = np.zeros(N)
    dX = np.zeros(N)

    for i in range(N):
        # 1. Kinetik Hesaplama: Arrhenius tipi kalsinasyon
        r = 0.0
        if Ts[i] >= T_min and X[i] < 1.0:
            k = k0 * np.exp(-Ea / (R_const * Ts[i]))
            r = min(k * (1.0 - X[i]), 0.01) # Sayısal kararlılık sınırı
        dX[i] = r
        
        # 2. Isı Transferi: Konveksiyon + Baskın Radyasyon (T^4)
        q_conv = h_gs * (Tg[i] - Ts[i]) * exchange_area_per_dz
        q_rad = sigma * eps_eff * (Tg[i]**4 - Ts[i]**4) * exchange_area_per_dz
        q_net = q_conv + q_rad
        
        # 3. Katı Faz Enerji Dengesi
        if i > 0:
            advection_s = v_s * (Ts[i-1] - Ts[i]) / dz
            # Reaksiyon ısısı (endotermik) enerjiyi tüketir
            source_s = (q_net - r * m_dot_s * dH_rxn) / (effective_thermal_mass * cp_s)
            dTs[i] = (advection_s + source_s) * damping
            
        # 4. Gaz Fazı Enerji Dengesi (Karşıt Akış)
        if i < N - 1:
            v_g = m_dot_g / (max(0.1, rho_g[i]) * area_gas)
            advection_g = v_g * (Tg[i+1] - Tg[i]) / dz
            # Gaz kütlesi (m_dot_g) soğuma hızını belirler
            dTg[i] = advection_g - q_net / (max(1.0, m_dot_g) * cp_g)
            
    return dTs, dTg, dX

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
        """Config değerlerini güvenli bir şekilde float'a çevirir."""
        if isinstance(value, list): return float(value[0])
        return float(value)

    def _calculate_flame_temp(self, fuel_rate):
        """
        Yakıt debisini alev sıcaklığına bağlar. 
        Aşırı hassasiyeti önlemek için sönümlü (sqrt) bir model kullanır.
        """
        # Yakıt 0.65 kg/s iken yaklaşık 2300K - 2400K arası bir sonuç üretir
        base_temp = 20 
        gain_factor = 460
        flame_temp = base_temp + (np.sqrt(max(0, fuel_rate)) * gain_factor)
        
        return min(2400.0, flame_temp)

    def solve_step(self, dt, fuel_rate=None, feed_rate=None, kiln_rpm=None, fan_rate=None):
        """Dinamik kontrol değişkenleri ile tek adım çözümü."""
        
        # 1. Kontrol Girişlerini Sisteme Uygula
        if fuel_rate is not None:
            self.cfg['gas']['temp_inlet'] = self._calculate_flame_temp(fuel_rate)
            self.cfg['gas']['fuel_rate'] = fuel_rate
            
        if feed_rate is not None:
            self.cfg['material']['feed_rate'] = feed_rate
            
        if kiln_rpm is not None:
            self.cfg['kiln']['rpm'] = kiln_rpm
            
        if fan_rate is not None:
            # fan_rate (RPM) -> m_dot_g (kg/s) dönüşümü
            self.cfg['gas']['flow_rate'] = fan_rate / 100.0 

        # 2. Güncel Fiziksel Parametreleri Çek
        # RPM ve eğime bağlı olarak v_s hesaplanır
        v_s = self._safe_f(self.tra.calculate_solid_velocity())
        m_dot_s = self._safe_f(self.cfg['material']['feed_rate'])
        m_dot_g = self._safe_f(self.cfg['gas']['flow_rate'])
        
        cp_s = self._safe_f(self.cfg['material']['cp_s'])
        cp_g = self._safe_f(self.cfg['gas']['cp_g'])
        diameter = self._safe_f(self.cfg['kiln']['diameter'])
        dH_rxn = self._safe_f(self.kin.dH)
        fill = self._safe_f(self.cfg['kiln'].get('filling_degree', 0.10))
        h_gs = self._safe_f(self.cfg['gas']['h_gs'])
        sigma = 5.67e-8
        eps_eff = self._safe_f(self.cfg['gas']['emissivity_g'])

        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - fill)
        exchange_area_per_dz = diameter * self.dz 
        
        # Termal kütle hesabı (v_s düştükçe kütle artar, ısınma yavaşlar ama stabil olur)
        node_mass = (m_dot_s / max(0.001, v_s)) * self.dz
        effective_thermal_mass = node_mass + 15.0 # Fırın refrakter etkisi eklendi

        self.state.update_gas_density()

        # 3. Numba Çekirdeğini Çalıştır
        dTs, dTg, dX = _numba_step_core(
            self.state.N, self.state.Ts, self.state.Tg, self.state.X, self.state.rho_g,
            float(dt), self.dz, v_s, m_dot_s, m_dot_g, cp_s, cp_g, dH_rxn,
            area_gas, exchange_area_per_dz, self.damping, effective_thermal_mass,
            h_gs, sigma, eps_eff,
            self.kin.k0, self.kin.Ea, self.kin.R, self.kin.T_min
        )

        # 4. Entegrasyon ve Sınır Şartları
        self.state.X  += dX * dt
        self.state.Ts += dTs * dt
        self.state.Tg += dTg * dt
        
        # Sınır Şartları Sabitleme
        self.state.Ts[0] = self._safe_f(self.cfg['material']['temp_inlet'])
        self.state.Tg[-1] = self._safe_f(self.cfg['gas']['temp_inlet'])
        
        # 5. Fiziksel Clipping (Sayısal Kararlılık)
        self.state.X = np.clip(self.state.X, 0.0, 1.0)
        self.state.Ts = np.clip(self.state.Ts, 300.0, 1950.0)
        self.state.Tg = np.clip(self.state.Tg, 300.0, 2600.0)