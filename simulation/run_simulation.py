import yaml
import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Proje kök dizini erişimi
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
    raw_meal = config.get('raw_meal_composition', 
                          {'CaCO3': 0.82, 'SiO2': 0.13, 'Al2O3': 0.03, 'Fe2O3': 0.02})
    
    solver.state.initialize_profiles(
        T_ambient=float(config['material']['temp_inlet']), 
        T_gas_inlet=float(config['gas']['temp_inlet']),
        raw_meal_comp=raw_meal
    )

    # 3. Simülasyon Parametreleri (26 SAAT GÜNCELLEMESİ)
    t = 0.0
    t_final = 26.0 * 3600.0  # 26 saat saniye cinsinden
    dt = float(config['solver']['dt'])
    
    fuel_rate = float(config['gas'].get('fuel_rate', 16.0))
    fan_rate  = float(config['gas'].get('fan_rate', 800.0))
    kiln_rpm  = float(config['kiln'].get('rpm', 2.0))
    feed_rate = float(config['material'].get('feed_rate', 10.0))

    print(f"\n[SİMÜLASYON BAŞLADI] Hedef Süre: {t_final/3600:.1f} saat")
    print("=" * 125)
    header = f"{'Saat':>6} | {'Ts_Out(K)':>9} | {'X_Calc':>7} | {'CaO':>7} | {'SiO2':>7} | {'C2S':>7} | {'C3S':>7} | {'Mass_Rel':>10}"
    print(header)
    print("-" * 125)

    # 4. Ana Döngü
    start_wall_time = time.time()
    while t < t_final:
        solver.solve_step(dt=dt, fuel_rate=fuel_rate, feed_rate=feed_rate, kiln_rpm=kiln_rpm, fan_rate=fan_rate)
        t += dt
        
        # Periyodik Loglama (Her 1 saatte bir özet geçelim, çok uzun sürmesin)
        if int(t) % 600 == 0 and (t - dt) < int(t):
            s = solver.state
            print(f"{t/3600:6.1f}h | {s.Ts[-1]:9.1f} | {s.X[-1]:7.3f} | {s.m_CaO[-1]:7.5f} | "
                  f"{s.m_SiO2[-1]:7.5f} | {s.m_C2S[-1]:7.5f} | {s.m_C3S[-1]:7.5f} | "
                  f"{np.mean(s.total_mass):10.5f}", flush=True)

    print("=" * 125)
    print(f"Simülasyon Tamamlandı. Gerçek İşlem Süresi: {time.time() - start_wall_time:.2f} s\n")

    # --- BİRLEŞTİRİLMİŞ GÖRSELLEŞTİRME ---
    z_axis = np.linspace(0, float(config['kiln']['length']), solver.state.N)
    lw_val = 1.0 
    plt.style.use('seaborn-v0_8-muted')
    s = solver.state

    # Window 1: Termal Dinamikler
    plt.figure("Sıcaklık Profili (26. Saat)", figsize=(12, 5))
    plt.plot(z_axis, s.Ts, 'r-', label='Malzeme Sıcaklığı ($T_s$)', linewidth=lw_val)
    plt.plot(z_axis, s.Tg, 'b--', label='Gaz Sıcaklığı ($T_g$)', linewidth=lw_val, alpha=0.6)
    plt.ylabel("Sıcaklık (K)")
    plt.xlabel("Fırın Boyu (m)")
    plt.legend(loc='center right')
    plt.grid(True, alpha=0.2)

    # Window 2: Bütünleşik Kimyasal Dönüşüm
    plt.figure("Bütünleşik Kimyasal Dönüşüm Profili (26. Saat)", figsize=(12, 7))
    
    # 1. Hammadde ve Toplam Kütle
    m_caCO3_dynamic = raw_meal['CaCO3'] * (1.0 - s.X)
    plt.plot(z_axis, m_caCO3_dynamic, color='brown', label='$CaCO_3$ (Hammadde)', linewidth=lw_val, linestyle='--')
    plt.plot(z_axis, s.total_mass, 'k-', label='Toplam Katı Kütlesi', linewidth=1.5, alpha=0.8)
    
    # 2. Ara Ürün ve Fazlar
    plt.plot(z_axis, s.m_CaO, color='orange', label='Serbest $CaO$', linewidth=lw_val)
    plt.plot(z_axis, s.m_C2S, 'c-', label='Belit ($C_2S$)', linewidth=lw_val)
    plt.plot(z_axis, s.m_C3S, color='#2c3e50', label='Alit ($C_3S$)', linewidth=lw_val) # Alit rengi güncel
    plt.plot(z_axis, s.m_C3A, color='#8e44ad', label='$C_3A$', linewidth=lw_val, alpha=0.5)
    plt.plot(z_axis, s.m_C4AF, color='#27ae60', label='$C_4AF$', linewidth=lw_val, alpha=0.5)

    plt.axhline(y=0.40, color='red', linestyle=':', linewidth=0.8, alpha=0.6, label='%40 Hedef')
    plt.title(f"Hammadde Ayrışması ve Klinker Faz Oluşumu (Simülasyon Süresi: 26 Saat)")
    plt.ylabel("Kütle Oranı")
    plt.xlabel("Fırın Boyu (m)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.grid(True, alpha=0.2)
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()