import yaml
import time
import numpy as np
import sys
import os
import matplotlib
# Windows için en kararlı backend
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# Proje kök dizinine erişim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.solver import KilnSolver
from core.kinetics import CalcinationKinetics
from core.transport import TransportModel
from core.energy import EnergyModel

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", "model_config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Fizik Motorlarını Başlat
    kinetics = CalcinationKinetics(config)
    transport = TransportModel(config)
    energy = EnergyModel(config)
    solver = KilnSolver(config, kinetics, transport, energy)

    # 2. Başlatma (Initialization) - Kritik Düzeltme
    raw_meal = config.get('raw_meal_composition', 
                          {'CaCO3': 0.82, 'SiO2': 0.13, 'Al2O3': 0.03, 'Fe2O3': 0.02})
    
    # Tüm fırın başlangıçta soğuk (Ambient) olmalı
    ambient_temp = float(config['material'].get('temp_inlet', 300.0))
    gas_inlet_temp = float(config['gas'].get('temp_inlet', 2200.0))
    
    solver.state.initialize_profiles(
        T_ambient=ambient_temp, 
        T_gas_inlet=gas_inlet_temp,
        raw_meal_comp=raw_meal
    )

    # 3. Simülasyon Parametreleri
    t = 0.0
    t_final = 30.0 * 3600.0  
    dt = float(config['solver']['dt'])
    
    fuel_rate = float(config['gas'].get('fuel_rate', 4.16))
    fan_rate  = float(config['gas'].get('fan_rate', 800.0))
    kiln_rpm  = float(config['kiln'].get('rpm', 2.0))
    feed_rate = float(config['material'].get('feed_rate', 10.0))

    # --- TERMİNAL BAŞLIKLARI ---
    print(f"\n[SIMULATION STARTED] Target Duration: {t_final/3600:.1f} hours")
    print(f"Zoned Model: Preheat (35%) | Calc (25%) | Trans (20%) | Burn (20%)")
    print(f"Initial State: All nodes at {ambient_temp}K. Heating will propagate as material moves.")
    
    header = f"{'Time':>6} | {'Ts_Out':>7} | {'X_Calc':>6} | {'CaO':>6} | {'SiO2':>6} | {'C2S':>6} | {'C3S':>6} | {'C3A':>6} | {'C4AF':>6} | {'Mass_Rel':>8}"
    print("-" * 115)
    print(header)
    print("-" * 115)

    # 4. Ana Döngü
    start_wall_time = time.time()
    last_log_t = -1000.0 # İlk adımın basılmasını sağlar
    
    while t < t_final:
        # Solver adımını çalıştır
        actual_dt = solver.solve_step(dt=dt, fuel_rate=fuel_rate, feed_rate=feed_rate, kiln_rpm=kiln_rpm, fan_rate=fan_rate)
        t += actual_dt
        
        # Log basma mantığı: Her 600 saniyede bir (Simülasyon saatiyle)
        if (t - last_log_t) >= 600.0:
            s = solver.state
            print(f"{t/3600:6.2f}h | {s.Ts[-1]:7.1f} | {s.X[-1]:6.3f} | {s.m_CaO[-1]:6.4f} | "
                  f"{s.m_SiO2[-1]:6.4f} | {s.m_C2S[-1]:6.4f} | {s.m_C3S[-1]:6.4f} | "
                  f"{s.m_C3A[-1]:6.4f} | {s.m_C4AF[-1]:6.4f} | "
                  f"{np.mean(s.total_mass):8.4f}", flush=True)
            last_log_t = t

    print("=" * 115)
    print(f"Simulation Completed. Real Processing Time: {time.time() - start_wall_time:.2f} s\n")

    # --- GÖRSELLEŞTİRME ---
    z_axis = np.linspace(0, float(config['kiln']['length']), solver.state.N)
    kiln_len = float(config['kiln']['length'])
    plt.style.use('seaborn-v0_8-muted')
    s = solver.state
    lw = 1.5 

    # Figure 1: Sıcaklık Dinamikleri ve Bölgeler
    fig1 = plt.figure("Temperature Dynamics", figsize=(12, 7))
    plt.axvspan(0, kiln_len*0.35, color='green', alpha=0.05, label='Preheat')
    plt.axvspan(kiln_len*0.35, kiln_len*0.60, color='orange', alpha=0.05, label='Calcination')
    plt.axvspan(kiln_len*0.60, kiln_len*0.80, color='blue', alpha=0.05, label='Transition')
    plt.axvspan(kiln_len*0.80, kiln_len, color='red', alpha=0.05, label='Burning')

    plt.plot(z_axis, s.Ts, color='red', label='Material Temp ($T_s$)', linewidth=lw + 0.5)
    plt.plot(z_axis, s.Tg, color='blue', label='Gas Temp ($T_g$)', linewidth=lw, alpha=0.6)
    plt.plot(z_axis, s.Tw, color='gray', label='Wall Temp ($T_w$)', linewidth=lw, alpha=0.4)
    
    plt.ylabel("Temperature (K)")
    plt.xlabel("Kiln Length (m)")
    plt.title(f"Thermal Profile Evolution ({t/3600:.1f} Hours)")
    plt.legend(loc='upper left', frameon=True, fontsize='small', ncol=2)
    plt.grid(True, alpha=0.15)

    # Figure 2: Kimyasal Envanter
    fig2 = plt.figure("Chemical Inventory", figsize=(14, 8))
    m_caCO3_dyn = raw_meal['CaCO3'] * (1.0 - s.X)
    
    plt.plot(z_axis, m_caCO3_dyn, color='brown', label='$CaCO_3$', linewidth=lw)
    plt.plot(z_axis, s.m_SiO2, color='blue', label='$SiO_2$', linewidth=lw, alpha=0.4)
    plt.plot(z_axis, s.m_CaO, color='orange', label='Free $CaO$', linewidth=lw + 1)
    plt.plot(z_axis, s.m_C2S, color='cyan', label='Belite ($C_2S$)', linewidth=lw)
    plt.plot(z_axis, s.m_C3S, color='#2c3e50', label='Alite ($C_3S$)', linewidth=lw + 1)
    plt.plot(z_axis, s.m_C3A, color='limegreen', label='Aluminate ($C_3A$)', linewidth=lw)
    plt.plot(z_axis, s.m_C4AF, color='purple', label='Ferrite ($C_4AF$)', linewidth=lw)
    plt.plot(z_axis, s.total_mass, color='black', label='Total Solid Mass', linewidth=2)
    
    plt.ylabel("Mass Fraction")
    plt.xlabel("Kiln Length (m)")
    plt.ylim(-0.02, 1.05)
    plt.title(f"Chemical Phase Transformation ({t/3600:.1f} Hours)")
    plt.legend(loc='upper right', fontsize='x-small', frameon=True, ncol=3)
    plt.grid(True, alpha=0.15)

    # Ekranı Kapla
    for f in [fig1, fig2]:
        plt.figure(f.number)
        plt.pause(0.1) 
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')
        except:
            pass

    plt.show()

if __name__ == "__main__":
    main()