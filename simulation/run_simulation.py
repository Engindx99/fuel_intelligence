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
    fuel_rate = float(config['gas'].get('fuel_rate', 0.65))
    fan_rate  = float(config['gas'].get('fan_rate', 1200.0))
    kiln_rpm  = float(config['kiln'].get('rpm', 1.2))
    feed_rate = float(config['material'].get('feed_rate', 10.0))

    print(f" Simülasyon Başladı: {t_final/3600:.1f} saatlik operasyon...")

    # 4. Simülasyon Döngüsü
    start_wall_time = time.time()
    while t < t_final:
        solver.solve_step(dt=dt, fuel_rate=fuel_rate, feed_rate=feed_rate, kiln_rpm=kiln_rpm, fan_rate=fan_rate)
        t += dt
        if int(t) % 600 == 0 and (t - dt) < int(t):
            print(f" {t/3600:5.1f}h | Ts_out: {solver.state.Ts[-1]:6.1f}K | CaO_out: {solver.state.m_CaO[-1]:.3f}", flush=True)

    print(f"\n Tamamlandı. Süre: {time.time() - start_wall_time:.2f} s")

    # --- GÖRSELLEŞTİRME (İKİ AYRI PENCERE) ---
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

    # WINDOW 2: GİRDİ VE ÜRÜNLER (KÜTLE DENGESİ)
    plt.figure("Window 2: Material Transformation", figsize=(10, 6))
    
    # Toplam kütleyi hesaplayalım (CO2 kaybını görmek için)
    m_total = (solver.state.m_CaCO3 + solver.state.m_CaO + 
               solver.state.m_SiO2 + solver.state.m_Al2O3 + solver.state.m_Fe2O3)

    plt.stackplot(z_axis, 
                  solver.state.m_CaO, 
                  solver.state.m_CaCO3, 
                  solver.state.m_SiO2 + solver.state.m_Al2O3 + solver.state.m_Fe2O3,
                  labels=['Product (CaO)', 'Raw (CaCO3)', 'Inerts (SiO2+Al2O3+Fe2O3)'],
                  colors=['#2ecc71', '#95a5a6', '#34495e'],
                  alpha=0.8)
    
    # Toplam kütle çizgisini ekle (CO2 kaybını vurgular)
    plt.plot(z_axis, m_total, 'k:', label='Total Solid Mass (shows CO2 loss)', linewidth=1.0)
    
    plt.title("Mass Composition Along the Kiln")
    plt.xlabel("Kiln Length (m)")
    plt.ylabel("Mass Fraction")
    plt.ylim(0, 1.05)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.2)

    print(" Pencereler açılıyor...")
    plt.show()

if __name__ == "__main__":
    main()