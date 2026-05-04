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
    """
    Fuel Intelligence: 18-State Rotary Kiln Digital Twin Simulation Entry Point
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", "model_config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Hata: Yapılandırma dosyası bulunamadı -> {config_path}")
        return

    # 2. Fizik Motorlarını ve Çözücüyü Başlat
    try:
        kinetics = CalcinationKinetics(config)
        transport = TransportModel(config)
        energy = EnergyModel(config)
        solver = KilnSolver(config, kinetics, transport, energy)
    except KeyError as e:
        print(f"❌ Config dosyasında eksik parametre: {e}")
        return

    # 3. Başlangıç Şartlarını Ayarla
    solver.state.initialize_profiles(
        T_ambient=float(config['material']['temp_inlet']), 
        T_gas_inlet=float(config['gas']['temp_inlet'])
    )
    solver.state.update_gas_density()

    # 4. Simülasyon Kontrol Parametreleri
    t = 0.0
    t_final = float(config['solver']['t_final'])
    dt = float(config['solver']['dt'])
    
    fuel_rate = float(config['gas'].get('fuel_rate', 0.65))
    fan_rate  = float(config['gas'].get('fan_rate', 900.0))
    kiln_rpm  = float(config['kiln'].get('rpm', 1.2))
    feed_rate = float(config['material'].get('feed_rate', 10.0))
    
    save_interval = max(10, int(t_final / 20)) 
    
    # Görselleştirme için veri listeleri
    time_history = []
    exit_ts_history = []
    exit_tg_history = []

    print("\n" + "="*60)
    print(f"🔥 FUEL INTELLIGENCE - Simülasyon ve Görselleştirme Başlatıldı")
    print("="*60 + "\n")

    start_wall_time = time.time()

    try:
        while t < t_final:
            solver.solve_step(
                dt=dt, 
                fuel_rate=fuel_rate, 
                feed_rate=feed_rate, 
                kiln_rpm=kiln_rpm, 
                fan_rate=fan_rate
            )
            t += dt

            # Belirli aralıklarla veri kaydet ve ekrana bas
            if int(t) % save_interval == 0 and (t - dt) < int(t):
                exit_Ts = solver.state.Ts[-1]
                exit_Tg = solver.state.Tg[0] # Gaz fırın girişinden (bacadan) çıkar
                
                time_history.append(t / 3600.0) # Saat cinsinden
                exit_ts_history.append(exit_Ts)
                exit_tg_history.append(exit_Tg)
                
                progress = (t / t_final) * 100
                print(f"⏱️ {int(t):6d}s | %{progress:4.1f} | Exit Ts: {exit_Ts:6.1f} K", flush=True)

        end_wall_time = time.time()
        print(f"\n✅ Simülasyon Tamamlandı! Hesaplama Süresi: {end_wall_time - start_wall_time:.2f} s")

# --- VISUALIZATION SECTION ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.3)

        # Plot 1: Temperature Time Series (Kiln Exit)
        ax1.plot(time_history, exit_ts_history, 'r', label='Solid Discharge Temp (Ts Exit)', linewidth=1.0)
        ax1.plot(time_history, exit_tg_history, 'b', label='Gas Exhaust Temp (Tg Exit)', linewidth=1.0)
        ax1.set_title("Evolution of Kiln Exit Temperatures Over Time")
        ax1.set_xlabel("Time (Hours)")
        ax1.set_ylabel("Temperature (K)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Spatial Temperature Profile (Steady-State / Final Snapshot)
        z_axis = np.linspace(0, float(config['kiln']['length']), solver.state.N)
        ax2.plot(z_axis, solver.state.Ts, 'r-', linewidth=1.0, label='Solid Profile (Ts)')
        ax2.plot(z_axis, solver.state.Tg, 'b-', linewidth=1.0, label='Gas Profile (Tg)')
        ax2.set_title(f"Spatial Temperature Profile along Kiln (t = {t/3600:.1f} h)")
        ax2.set_xlabel("Kiln Length (m)")
        ax2.set_ylabel("Temperature (K)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        print("📊 Generating plots...")
        plt.show()

    except Exception:
        print("\n❌ Simülasyon sırasında kritik bir hata oluştu:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()