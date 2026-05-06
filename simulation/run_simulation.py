import yaml
import time
import numpy as np
import sys
import os
import matplotlib
# Best stable backend for Windows
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# Access to project root directory
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

    # 1. Initialize Physics Engines
    kinetics = CalcinationKinetics(config)
    transport = TransportModel(config)
    energy = EnergyModel(config)
    solver = KilnSolver(config, kinetics, transport, energy)

    # 2. Initialization
    raw_meal = config.get('raw_meal_composition', 
                          {'CaCO3': 0.82, 'SiO2': 0.13, 'Al2O3': 0.03, 'Fe2O3': 0.02})
    
    solver.state.initialize_profiles(
        T_ambient=float(config['material']['temp_inlet']), 
        T_gas_inlet=float(config['gas']['temp_inlet']),
        raw_meal_comp=raw_meal
    )

    # 3. Simulation Parameters
    t = 0.0
    t_final = 8.0 * 3600.0  
    dt = float(config['solver']['dt'])
    
    fuel_rate = float(config['gas'].get('fuel_rate', 16.0))
    fan_rate  = float(config['gas'].get('fan_rate', 800.0))
    kiln_rpm  = float(config['kiln'].get('rpm', 2.0))
    feed_rate = float(config['material'].get('feed_rate', 10.0))

    # --- TERMINAL HEADERS ---
    print(f"\n[SIMULATION STARTED] Target Duration: {t_final/3600:.1f} hours")
    print("=" * 150)
    header = f"{'Time':>6} | {'Ts_Out':>7} | {'X_Calc':>6} | {'CaO':>6} | {'SiO2':>6} | {'C2S':>6} | {'C3S':>6} | {'C3A':>6} | {'C4AF':>6} | {'Mass_Rel':>8}"
    print(header)
    print("-" * 150)

    # 4. Main Loop
    start_wall_time = time.time()
    while t < t_final:
        solver.solve_step(dt=dt, fuel_rate=fuel_rate, feed_rate=feed_rate, kiln_rpm=kiln_rpm, fan_rate=fan_rate)
        t += dt
        
        if int(t) % 600 == 0 and (t - dt) < int(t):
            s = solver.state
            print(f"{t/3600:6.1f}h | {s.Ts[-1]:7.1f} | {s.X[-1]:6.3f} | {s.m_CaO[-1]:6.4f} | "
                  f"{s.m_SiO2[-1]:6.4f} | {s.m_C2S[-1]:6.4f} | {s.m_C3S[-1]:6.4f} | "
                  f"{s.m_C3A[-1]:6.4f} | {s.m_C4AF[-1]:6.4f} | "
                  f"{np.mean(s.total_mass):8.4f}", flush=True)

    print("=" * 150)
    print(f"Simulation Completed. Real Processing Time: {time.time() - start_wall_time:.2f} s\n")

    # --- VISUALIZATION ---
    z_axis = np.linspace(0, float(config['kiln']['length']), solver.state.N)
    plt.style.use('seaborn-v0_8-muted')
    s = solver.state
    lw = 1.2

    # Figure 1: Temperature Dynamics
    fig1 = plt.figure("Temperature Dynamics", figsize=(12, 7))
    plt.plot(z_axis, s.Ts, color='red', label='Material Temperature ($T_s$)', linewidth=lw)
    plt.plot(z_axis, s.Tg, color='blue', label='Gas Temperature ($T_g$)', linewidth=lw, alpha=0.7)
    plt.ylabel("Temperature (K)")
    plt.xlabel("Kiln Length (m)")
    plt.title(f"Thermal Gradient Along the Kiln ({t/3600:.1f} Hours)")
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, alpha=0.2)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1)

    # Figure 2: Integrated Chemical Inventory
    fig2 = plt.figure("Integrated Chemical Inventory", figsize=(14, 8))
    
    # Calcination and Oxide Inputs
    m_caCO3_dyn = raw_meal['CaCO3'] * (1.0 - s.X)
    plt.plot(z_axis, m_caCO3_dyn, color='brown', label='$CaCO_3$', linewidth=lw)
    plt.plot(z_axis, s.m_SiO2, color='blue', label='$SiO_2$', linewidth=lw, alpha=0.5)
    plt.plot(z_axis, s.m_Al2O3, color='gold', label='$Al_2O_3$', linestyle='--', linewidth=lw, alpha=0.7)
    plt.plot(z_axis, s.m_Fe2O3, color='maroon', label='$Fe_2O_3$', linestyle='--', linewidth=lw, alpha=0.7)
    
    # Intermediate Phases and Free Lime
    plt.plot(z_axis, s.m_CaO, color='orange', label='Free $CaO$', linewidth=lw+0.5)
    plt.plot(z_axis, s.m_C2S, color='cyan', label='Belite ($C_2S$)', linewidth=lw)
    
    # Major Phase Outputs
    plt.plot(z_axis, s.m_C3S, color='#2c3e50', label='Alite ($C_3S$)', linewidth=lw+0.5)
    plt.plot(z_axis, s.m_C3A, color='limegreen', label='Aluminate ($C_3A$)', linewidth=lw)
    plt.plot(z_axis, s.m_C4AF, color='purple', label='Ferrite ($C_4AF$)', linewidth=lw)
    
    # Mass Balance
    plt.plot(z_axis, s.total_mass, color='black', label='Total Solid Mass', linewidth=1.5)
    plt.plot(z_axis, 1.0 - s.total_mass, color='gray', label='Released $CO_2$', linewidth=lw, linestyle=':')

    plt.ylabel("Mass Fraction")
    plt.xlabel("Kiln Length (m)")
    plt.ylim(-0.02, 1.05)
    plt.title(f"Integrated Chemical Transformation Profile ({t/3600:.1f} Hours)")
    
    # Legend: 3-column layout
    plt.legend(loc='upper right', fontsize='x-small', frameon=True, ncol=3)
    plt.grid(True, alpha=0.15)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1)

    # --- ENFORCE FULL SCREEN ---
    for f in [fig1, fig2]:
        plt.figure(f.number)
        plt.pause(0.1) 
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')
        except:
            try:
                mng.full_screen_toggle()
            except:
                pass

    plt.show()

if __name__ == "__main__":
    main()