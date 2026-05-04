"""
Fuel Intelligence - Rotary Kiln Visualization Tool (18-State Optimized)
"""

import hashlib
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from simulation.engine import KilnSimulation
from core.state import *

def run_and_visualize():
    # --- 1. SİMÜLASYON AYARLARI ---
    L            = 60.0
    N_CELLS      = 100
    DT           = 0.05  # Vektörize motor için kararlı adım
    SIM_HOURS    = 2.0
    TOTAL_STEPS  = int(SIM_HOURS * 3600 / DT)
    LOG_INTERVAL = 1000  # Terminal çıktısı sıklığı
    
    sim = KilnSimulation(L, N_CELLS, DT)
    
    # Başlangıç Koşulları
    initial_x = create_zero_state(N_CELLS)
    zs = np.linspace(0.0, L, N_CELLS)
    initial_x[:, IDX_T_S] = sim._T_s_in
    
    tg_cold = float(max(350.0, sim._T_amb + 80.0))
    tg_hot = float(min(1050.0, sim._T_s_in - 80.0))
    initial_x[:, IDX_T_G] = tg_cold + (tg_hot - tg_cold) * (zs / L)
    initial_x[:, IDX_CaCO3] = 0.8
    initial_x[:, IDX_SiO2] = 0.14
    initial_x[:, IDX_Al2O3] = 0.04
    initial_x[:, IDX_Fe2O3] = 0.02
    initial_x[:, IDX_EPSILON] = 0.4
    sim.set_initial_condition(initial_x)
    
    # Kontrol Girdileri (U)
    u = create_zero_control()
    u[IDX_FUEL]    = 16.0   
    u[IDX_FAN]     = 1200.0 
    u[IDX_FEED]    = 10.0  
    u[IDX_REACTOR] = 2.0   
    
    # Veri Toplama Listeleri
    history_t = []
    history_tg_hotend = []
    history_ts_exit = []
    
    print(f"Simülasyon başlatılıyor: {SIM_HOURS} saatlik operasyon...", flush=True)
    start_time = time.time()
    
    try:
        for step_idx in range(TOTAL_STEPS):
            current_X = sim.step(u)
            
            # Her 10 dakikada bir veri kaydet
            if step_idx % int(600/DT) == 0:
                history_t.append((step_idx * DT) / 3600.0)
                history_ts_exit.append(current_X[-1, IDX_T_S])
                # Sadece Brülör Ucu (z=L) gaz sıcaklığı kaydediliyor
                history_tg_hotend.append(current_X[-1, IDX_T_G])
                
            if step_idx % LOG_INTERVAL == 0:
                prog = (step_idx / TOTAL_STEPS) * 100
                print(f"İlerleme: %{prog:.1f} | t: {step_idx*DT/3600:.1f} sa", end='\r', flush=True)

    except Exception as e:
        print(f"\n[HATA] Simülasyon sırasında hata oluştu: {e}")
        return

    print(f"\nSimülasyon bitti. Gerçek süre: {time.time() - start_time:.1f}s", flush=True)

    _root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(_root, "configs", "model_config.yaml")
    with open(cfg_path, "rb") as cf:
        cfg_bytes = cf.read()
    cfg_yaml = yaml.safe_load(cfg_bytes.decode("utf-8")) or {}
    cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()[:16]
    stamp = (
        f"Fuel Intelligence | SIM_HOURS={SIM_HOURS} | N_CELLS={N_CELLS} | "
        f"DT={DT} | model_config.sha256[{cfg_sha}] | "
        f"fuel_mode={cfg_yaml.get('thermal', {}).get('fuel_heat_mode', '?')} | "
        f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} local"
    )
    manifest = stamp + os.linesep + (
        "Outputs: kiln_temp_profile, kiln_temp_trend, kiln_chem_*, kiln_clinker_quality.png"
        + os.linesep
    )
    with open(os.path.join(_root, "plots_run_manifest.txt"), "w", encoding="utf-8") as mf:
        mf.write(manifest)

    # --- 2. GÖRSELLEŞTİRME ---
    sns.set_theme(style="darkgrid")
    z = np.linspace(0, L, N_CELLS)
    
    # Plot 1: Sıcaklık Profili
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_T_G], 'r-', label='Gaz Sıcaklığı (Tg)', linewidth=1)
    plt.plot(z, current_X[:, IDX_T_S], 'b-', label='Katı Sıcaklığı (Ts)', linewidth=1)
    plt.title("Fırın Boyunca Sıcaklık Profili (Steady-State)", fontsize=12)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Sıcaklık [K]")
    plt.legend()
    plt.savefig("kiln_temp_profile.png", dpi=200)
    plt.close()
    
    # Plot 2: Ana Fazlar
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_CaCO3], label='CaCO3', linewidth=1)
    plt.plot(z, current_X[:, IDX_CaO], label='CaO', linewidth=1)
    plt.plot(z, current_X[:, IDX_C2S], label='C2S (Belit)', linewidth=1)
    plt.plot(z, current_X[:, IDX_C3S], label='C3S (Alit)', linewidth=1, color='darkgreen')
    plt.title("Ana Kimyasal Faz Dönüşümü", fontsize=12)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Kütle Oranı")
    plt.legend()
    plt.savefig("kiln_chem_main.png", dpi=200)
    plt.close()

    # Plot 3: Oksitler ve Azınlık Fazlar
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_SiO2], '--', label='SiO2')
    plt.plot(z, current_X[:, IDX_Al2O3], ':', label='Al2O3')
    plt.plot(z, current_X[:, IDX_Fe2O3], '-.', label='Fe2O3')
    plt.plot(z, current_X[:, IDX_C3A], label='C3A', color='orange')
    plt.plot(z, current_X[:, IDX_C4AF], label='C4AF', color='brown')
    plt.title("Oksitler ve Yan Fazlar", fontsize=12)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Kütle Oranı")
    plt.legend()
    plt.savefig("kiln_chem_minor.png", dpi=200)
    plt.close()
    
    # Plot 4: Çıkış Trendleri (GÜNCELLENDİ)
    plt.figure(figsize=(10, 6))
    plt.plot(history_t, history_tg_hotend, 'r', alpha=0.8, label='Brülör Ucu (Tg, z=L)', linewidth=1)
    plt.plot(history_t, history_ts_exit, 'b', label='Klinker (Exit Ts)', linewidth=1)
    plt.title("Zamanla Çıkış Sıcaklık Değişimi", fontsize=12)
    plt.xlabel("Zaman [saat]")
    plt.ylabel("Sıcaklık [K]")
    plt.legend()
    plt.savefig("kiln_temp_trend.png", dpi=200)
    plt.close()
    
    # Plot 5: Klinkerleşme Oranı
    plt.figure(figsize=(10, 6))
    total_solid = np.sum(current_X[:, SOLID_SPECIES], axis=1)
    clinker_quality = np.divide(current_X[:, IDX_C3S], total_solid, 
                                out=np.zeros_like(total_solid), 
                                where=total_solid!=0)
    
    plt.fill_between(z, clinker_quality, color='green', alpha=0.3, label='C3S / Toplam Katı')
    plt.title("Klinkerleşme İlerlemesi (C3S Formasyonu)", fontsize=12)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Bağıl Oran")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig("kiln_clinker_quality.png", dpi=200)
    plt.close()

    print("Analiz grafikleri (PNG) başarıyla güncellendi.", flush=True)

if __name__ == "__main__":
    run_and_visualize()