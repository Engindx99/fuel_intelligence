import yaml
import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Proje kök dizinini path'e ekleyerek modül erişimini garantiye alalım
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

    # 2. İlklendirme
    raw_meal = {'CaCO3': 0.78, 'SiO2': 0.14, 'Al2O3': 0.05, 'Fe2O3': 0.03}
    solver.state.initialize_profiles(
        T_ambient=float(config['material']['temp_inlet']), 
        T_gas_inlet=float(config['gas']['temp_inlet']),
        raw_meal_comp=raw_meal
    )

    # 3. Parametreler
    t, t_final = 0.0, float(config['solver']['t_final'])
    dt = float(config['solver']['dt'])
    
    # Girişler
    fuel_rate = float(config['gas'].get('fuel_rate', 16.0))
    fan_rate  = float(config['gas'].get('fan_rate', 800.0))
    kiln_rpm  = float(config['kiln'].get('rpm', 2.0))
    feed_rate = float(config['material'].get('feed_rate', 10.0))

    print(f"\n[SİMÜLASYON BAŞLADI] Hedef: {t_final/3600:.1f} saat")
    print("-" * 110)
    # Header formatı: Zaman | Sıcaklık | Kalsinasyon | Serbest Kireç | Klinker Fazları
    header = f"{'Saat':>6} | {'Ts_out':>8} | {'X':>5} | {'CaO':>6} | {'SiO2':>6} | {'C2S':>6} | {'C3S':>6} | {'C3A':>6} | {'C4AF':>6}"
    print(header)
    print("-" * 110)

    # 4. Simülasyon Döngüsü
    start_wall_time = time.time()
    while t < t_final:
        solver.solve_step(dt=dt, fuel_rate=fuel_rate, feed_rate=feed_rate, kiln_rpm=kiln_rpm, fan_rate=fan_rate)
        t += dt
        
        # Her 10 dakikada bir detaylı log bas (600 saniye)
        if int(t) % 600 == 0 and (t - dt) < int(t):
            s = solver.state
            log_line = (f"{t/3600:6.1f}h | {s.Ts[-1]:8.1f}K | {s.X[-1]:5.3f} | {s.m_CaO[-1]:6.3f} | "
                        f"{s.m_SiO2[-1]:6.3f} | {s.m_C2S[-1]:6.3f} | {s.m_C3S[-1]:6.3f} | "
                        f"{s.m_C3A[-1]:6.3f} | {s.m_C4AF[-1]:6.3f}")
            print(log_line, flush=True)

    print("-" * 110)
    print(f"Tamamlandı. Toplam Simülasyon Süresi: {time.time() - start_wall_time:.2f} s\n")

    # --- GÖRSELLEŞTİRME ---
    z_axis = np.linspace(0, float(config['kiln']['length']), solver.state.N)

    # WINDOW 1: SICAKLIK PROFİLLERİ
    plt.figure("Window 1: Thermal Dynamics", figsize=(10, 6))
    plt.plot(z_axis, solver.state.Ts, 'r-', label='Solid Temperature (Ts)', linewidth=1.0)
    plt.plot(z_axis, solver.state.Tg, 'b--', label='Gas Temperature (Tg)', linewidth=1.0)
    plt.title(f"Spatial Temperature Profile (t = {t/3600:.1f} h)")
    plt.xlabel("Kiln Length (m)")
    plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # WINDOW 2: ANA MALZEME DÖNÜŞÜMÜ
    plt.figure("Window 2: Calcination & Basic Oxides", figsize=(10, 6))
    plt.plot(z_axis, solver.state.m_CaCO3, color='gray', label='CaCO3', linewidth=1.0)
    plt.plot(z_axis, solver.state.m_CaO, color='orange', label='Free CaO', linewidth=1.0)
    plt.plot(z_axis, solver.state.m_SiO2, color='blue', label='SiO2', linewidth=1.0, alpha=0.6)
    
    m_total = (solver.state.m_CaCO3 + solver.state.m_CaO + solver.state.m_SiO2 + 
               solver.state.m_Al2O3 + solver.state.m_Fe2O3 + solver.state.m_C2S + 
               solver.state.m_C3S + solver.state.m_C3A + solver.state.m_C4AF)
    
    plt.plot(z_axis, m_total, 'k:', label='Total Solid Mass (CO2 Loss)', linewidth=1.0)
    plt.title("Primary Reactants and CO2 Loss Profile")
    plt.xlabel("Kiln Length (m)")
    plt.ylabel("Mass Fraction")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.2)

    # WINDOW 3: KLİNKER FAZLARI
    plt.figure("Window 3: Clinker Phases (Bogue)", figsize=(10, 6))
    plt.plot(z_axis, solver.state.m_C2S, 'c-', label='Belite (C2S)', linewidth=1.0)
    plt.plot(z_axis, solver.state.m_C3S, 'm-', label='Alite (C3S)', linewidth=1.0)
    plt.plot(z_axis, solver.state.m_C3A, 'y-', label='C3A', linewidth=1.0)
    plt.plot(z_axis, solver.state.m_C4AF, 'g-', label='C4AF', linewidth=1.0)
    plt.title("Clinker Phase Formation Along the Kiln")
    plt.xlabel("Kiln Length (m)")
    plt.ylabel("Mass Fraction")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.2)

    print("Görselleştirme pencereleri açılıyor...")
    plt.show()

if __name__ == "__main__":
    main()