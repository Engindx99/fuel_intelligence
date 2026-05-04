import yaml
import time
import numpy as np
import sys
import os

# Proje kök dizinini path'e ekleyerek modül erişimini garantiye alalım
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.solver import KilnSolver
from core.kinetics import CalcinationKinetics
from core.transport import TransportModel
from core.energy import EnergyModel
from analysis.plots import plot_kiln_results 

def main():
    """
    Fuel Intelligence: 18-State Rotary Kiln Digital Twin Simulation Entry Point
    """
    # 1. Config Dosyasını UTF-8 ve Dinamik Yol ile Yükle
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
    
    # YAML'dan kontrol değişkenlerini çek
    fuel_rate = float(config['gas'].get('fuel_rate', 0.65))
    fan_rate  = float(config['gas'].get('fan_rate', 900.0))
    kiln_rpm  = float(config['kiln'].get('rpm', 1.2))
    feed_rate = float(config['material'].get('feed_rate', 10.0))
    
    save_interval = max(10, int(t_final / 20)) 
    start_wall_time = time.time()
    
    print("\n" + "="*60)
    print(f"🔥 FUEL INTELLIGENCE - Dinamik Kontrol Başlatıldı")
    print(f"🎮 Kontrol Değişkenleri: Fuel: {fuel_rate} | Fan: {fan_rate} | RPM: {kiln_rpm}")
    print(f"⏱️ Toplam Simülasyon Süresi: {t_final} s")
    print("="*60 + "\n")

    try:
        while t < t_final:
            # GÜNCELLEME: Kontrol değişkenlerini her adımda solver'a gönderiyoruz
            solver.solve_step(
                dt=dt, 
                fuel_rate=fuel_rate, 
                feed_rate=feed_rate, 
                kiln_rpm=kiln_rpm, 
                fan_rate=fan_rate
            )
            t += dt

            # Terminale Durum Bilgisi Yazdır
            if int(t) % save_interval == 0 and (t - dt) < int(t):
                max_X = np.max(solver.state.X)
                avg_Ts = np.mean(solver.state.Ts)
                # Fırın sonundaki (discharge) sıcaklık bizim asıl hedefimiz
                exit_Ts = solver.state.Ts[-1]
                progress = (t / t_final) * 100
                
                print(f"⏱️ {int(t):6d}s | %{progress:4.1f} | 转化率: {max_X:.4f} | Exit Ts: {exit_Ts:6.1f} K", flush=True)

        end_wall_time = time.time()
        print("\n" + "="*60)
        print(f"✅ Simülasyon Tamamlandı! (Süre: {end_wall_time - start_wall_time:.2f} s)")
        print(f"🌡️ Nihai Çıkış Sıcaklığı: {solver.state.Ts[-1]:.2f} K")
        print("="*60 + "\n")

        # 5. Görselleştirme
        z_axis = np.linspace(0, float(config['kiln']['length']), solver.state.N)
        plot_kiln_results(
            z=z_axis, 
            Ts=solver.state.Ts, 
            Tg=solver.state.Tg, 
            X=solver.state.X, 
            title=f"Fuel Intelligence: Exit Ts = {solver.state.Ts[-1]:.1f}K"
        )

    except Exception:
        print("\n❌ Simülasyon sırasında kritik bir hata oluştu:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()