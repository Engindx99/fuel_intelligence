import numpy as np
from core.state import KilnState

def safe_f(value):
    if isinstance(value, list):
        return float(value[0])
    return float(value)

class KilnSolver:
    def __init__(self, config, kinetics, transport, energy):
        self.cfg = config
        self.kin = kinetics
        self.tra = transport
        self.en  = energy
        
        nodes = int(safe_f(config['kiln']['nodes']))
        self.state = KilnState(nodes)
        self.dz = safe_f(config['kiln']['length']) / nodes
        
        # Dinamik RPM kontrolü için damping katsayısı
        self.damping = 0.4 

    def solve_step(self, dt):
        N = self.state.N
        dt = float(dt)
        
        # --- KONTROL DEĞİŞKENLERİ VE PARAMETRELER ---
        # RPM değiştikçe v_s anlık güncellenir (Transport modülünden)
        v_s = safe_f(self.tra.calculate_solid_velocity())
        m_dot_s = safe_f(self.cfg['material']['feed_rate'])
        m_dot_g = safe_f(self.cfg['gas']['flow_rate'])
        cp_s = safe_f(self.cfg['material']['cp_s'])
        cp_g = safe_f(self.cfg['gas']['cp_g'])
        diameter = safe_f(self.cfg['kiln']['diameter'])
        dH_rxn = safe_f(self.kin.dH)
        
        fill = safe_f(self.cfg['kiln'].get('filling_degree', 0.08))
        area_gas = (np.pi * (diameter / 2)**2) * (1.0 - fill)
        exchange_area_per_dz = diameter * self.dz 

        # --- DİNAMİK HOLD-UP VE RESIDENCE TIME ETKİSİ ---
        # RPM artarsa v_s artar -> node_mass azalır -> Isıl atalet düşer.
        # Bu sayede RPM artışı, malzemenin daha az ısınarak fırından hızlı çıkmasını sağlar.
        node_mass = (m_dot_s / max(0.001, v_s)) * self.dz
        
        # Fırın refrakter etkisini de içeren dinamik termal kütle
        effective_thermal_mass = node_mass + 10.0 
        
        Ts = self.state.Ts.copy()
        Tg = self.state.Tg.copy()
        X  = self.state.X.copy()
        
        dTs = np.zeros(N)
        dTg = np.zeros(N)
        dX  = np.zeros(N)

        self.state.update_gas_density()

        for i in range(N):
            # 1. Reaksiyon Hızı (Kinetik)
            r = np.clip(self.kin.compute_rate(Ts[i], X[i]), 0, 0.01)
            dX[i] = r
            
            # 2. Isı Değişimi (Q_gs)
            q_net = self.en.heat_exchange(Tg[i], Ts[i], area=exchange_area_per_dz)
            
            # 3. Katı Faz (RPM Duyarlı Enerji Dengesi)
            if i > 0:
                # Eksenel taşıma (Advection): v_s (RPM) ile doğrudan orantılı
                advection_s = v_s * (Ts[i-1] - Ts[i]) / self.dz
                
                # Kaynak Terimi: q_net ve endotermik reaksiyon kaybı
                # effective_thermal_mass (v_s bağımlı) burada sönümleyici görev yapar
                source_s = (q_net - r * m_dot_s * dH_rxn) / (effective_thermal_mass * cp_s)
                
                dTs[i] = (advection_s + source_s) * self.damping
            
            # 4. Gaz Fazı (Karşıt Akış)
            if i < N - 1:
                v_g = m_dot_g / (max(0.1, float(self.state.rho_g[i])) * area_gas)
                advection_g = v_g * (Tg[i+1] - Tg[i]) / self.dz
                dTg[i] = advection_g - q_net / (m_dot_g * cp_g)

        # Entegrasyon
        self.state.X  += dX * dt
        self.state.Ts += dTs * dt
        self.state.Tg += dTg * dt
        
        # Sınır Şartları
        self.state.Ts[0] = safe_f(self.cfg['material']['temp_inlet'])
        self.state.Tg[-1] = safe_f(self.cfg['gas']['temp_inlet'])
        
        # Fiziksel Sınırlamalar
        self.state.X = np.clip(self.state.X, 0.0, 1.0)
        self.state.Ts = np.clip(self.state.Ts, 300.0, 1600.0)
        self.state.Tg = np.clip(self.state.Tg, 300.0, 2400.0)