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
    # 1. Konfigürasyonu Yükle
    config_path = 'configs/model_config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Hata: {config_path} bulunamadı!")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Fizik Motorlarını ve Çözücüyü Başlat
    try:
        kinetics = CalcinationKinetics(config)
        transport = TransportModel(config)
        energy = EnergyModel(config)
        solver = KilnSolver(config, kinetics, transport, energy)
    except KeyError as e:
        print(f"❌ Config dosyasında eksik parametre: {e}")
        return

    # 3. Başlangıç Şartlarını ve Gaz Yoğunluğunu Ayarla
    # T_ambient ve T_gas_inlet değerlerini float'a zorlayarak tip güvenliği sağlıyoruz
    solver.state.initialize_profiles(
        T_ambient=float(config['material']['temp_inlet']), 
        T_gas_inlet=float(config['gas']['temp_inlet'])
    )
    solver.state.update_gas_density()

    # 4. Simülasyon Kontrol Parametreleri
    t = 0.0
    t_final = float(config['solver']['t_final'])
    dt = float(config['solver']['dt'])
    
    # Loglama frekansını simülasyon süresine göre dinamik ayarlayalım
    save_interval = max(10, int(t_final / 20)) 
    
    start_wall_time = time.time()
    
    print("\n" + "="*60)
    print(f"🔥 FUEL INTELLIGENCE - Döner Fırın Dijital İkiz Simülasyonu")
    print(f"📍 Fırın Uzunluğu: {config['kiln']['length']} m")
    print(f"⏱️ Toplam Simülasyon Süresi: {t_final} s (dt: {dt}s)")
    print("="*60 + "\n")

    try:
        while t < t_final:
            solver.solve_step(dt)
            t += dt

            # Belirli aralıklarla terminale durum bilgisi yazdır
            if int(t) % save_interval == 0 and (t - dt) < int(t):
                max_X = np.max(solver.state.X)
                avg_Ts = np.mean(solver.state.Ts)
                progress = (t / t_final) * 100
                
                print(f"⏱️ {int(t):5d}s | 📊 İlerleme: %{progress:5.1f} | 转化率 (Max X): {max_X:.4f} | 🌡️ Ort. Ts: {avg_Ts:6.1f} K", flush=True)

        end_wall_time = time.time()
        print("\n" + "="*60)
        print(f"✅ Simülasyon Başarıyla Tamamlandı!")
        print(f"🚀 Gerçek Zamanlı Hesaplama Süresi: {end_wall_time - start_wall_time:.2f} s")
        print("="*60 + "\n")

        # 5. Profesyonel Görselleştirme
        z_axis = np.linspace(0, float(config['kiln']['length']), solver.state.N)
        
        # image_b829f2.png dosyasındaki test başarısından sonra gerçek verileri basıyoruz
        plot_kiln_results(
            z=z_axis, 
            Ts=solver.state.Ts, 
            Tg=solver.state.Tg, 
            X=solver.state.X, 
            title=f"Fuel Intelligence: {config['kiln']['length']}m Rotary Kiln Digital Twin Result"
        )

    except Exception:
        print("\n❌ Simülasyon sırasında kritik bir hata oluştu:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()