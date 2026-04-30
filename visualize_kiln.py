# visualize_kiln.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation.engine import KilnSimulation
from core.state import *
import time

def run_and_visualize():
    # --- 1. SİMÜLASYON AYARLARI ---
    L            = 60.0
    N_CELLS      = 100
    DT           = 0.01
    SIM_HOURS    = 7
    TOTAL_STEPS  = int(SIM_HOURS * 3600 / DT)
    LOG_INTERVAL = 500
    
    sim = KilnSimulation(L, N_CELLS, DT)
    
    # Başlangıç Koşulları
    initial_x = create_zero_state()
    initial_x[IDX_T_S]   = 300.0
    initial_x[IDX_T_G]   = 300.0
    initial_x[IDX_CaCO3] = 0.8
    initial_x[IDX_SiO2]  = 0.14
    initial_x[IDX_Al2O3] = 0.04
    initial_x[IDX_Fe2O3] = 0.02
    initial_x[IDX_EPSILON] = 0.4
    sim.set_initial_condition(initial_x)
    
    # Kontrol Girdileri
    u = create_zero_control()
    u[IDX_FUEL]    = 8.0
    u[IDX_FAN]     = 5000.0
    u[IDX_FEED]    = 10.0
    u[IDX_REACTOR] = 3.5
    
    # Veri Toplama
    history_t = []
    history_tg_exit = []
    history_ts_exit = []
    
    print("Simülasyon başlatılıyor...", flush=True)
    start_time = time.time()
    
    for step_idx in range(TOTAL_STEPS):
        current_X = sim.step(u)
        
        if step_idx % 600 == 0:  # Her 10 dakikada bir kaydet
            history_t.append((step_idx * DT) / 3600.0)
            history_ts_exit.append(current_X[-1, IDX_T_S])
            history_tg_exit.append(current_X[0, IDX_T_G]) # Smoke Temp (Exit)
            
        if step_idx % LOG_INTERVAL == 0:
            print(f"Saat: {step_idx*DT/3600:.1f} / {SIM_HOURS}", flush=True)

    print(f"Simülasyon bitti. Süre: {time.time() - start_time:.1f}s", flush=True)

    # --- 2. GÖRSELLEŞTİRME (FERAH AYRI DOSYALAR) ---
    sns.set_theme(style="darkgrid")
    z = np.linspace(0, L, N_CELLS)
    
    # Plot 1: Sıcaklık Profili
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_T_G], 'r-', label='Gaz Sıcaklığı (Tg)', linewidth=2.5)
    plt.plot(z, current_X[:, IDX_T_S], 'b-', label='Katı Sıcaklığı (Ts)', linewidth=2.5)
    plt.title("Fırın Boyunca Sıcaklık Dağılımı", fontsize=14)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Sıcaklık [K]")
    plt.legend()
    plt.savefig("kiln_temp_profile.png", dpi=200)
    plt.close()
    
    # Plot 2: Ana Kimyasal Türler (Major Phases)
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_CaCO3], label='CaCO3', linewidth=2.5)
    plt.plot(z, current_X[:, IDX_CaO], label='CaO', linewidth=2.5)
    plt.plot(z, current_X[:, IDX_C2S], label='C2S (Belit)', linewidth=2.5)
    plt.plot(z, current_X[:, IDX_C3S], label='C3S (Alit)', linewidth=2.5, color='darkgreen')
    plt.title("Ana Kimyasal Dönüşüm (Major Phases)", fontsize=14)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Konsantrasyon")
    plt.legend()
    plt.savefig("kiln_chem_main.png", dpi=200)
    plt.close()

    # Plot 3: Oksitler ve Azınlık Fazlar (Minor/Trace)
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_SiO2], '--', label='SiO2', linewidth=2)
    plt.plot(z, current_X[:, IDX_Al2O3], ':', label='Al2O3', linewidth=2)
    plt.plot(z, current_X[:, IDX_Fe2O3], '-.', label='Fe2O3', linewidth=2)
    plt.plot(z, current_X[:, IDX_C3A], label='C3A', linewidth=2, color='orange')
    plt.plot(z, current_X[:, IDX_C4AF], label='C4AF', linewidth=2, color='brown')
    plt.title("Oksitler ve Azınlık Fazlar (Minor/Trace Components)", fontsize=14)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Konsantrasyon")
    plt.legend()
    plt.savefig("kiln_chem_minor.png", dpi=200)
    plt.close()
    
    # Plot 4: Çıkış Sıcaklık Trendleri
    plt.figure(figsize=(10, 6))
    plt.plot(history_t, history_tg_exit, 'r--', label='Baca Gazı Sıcaklığı (Smoke Tg)')
    plt.plot(history_t, history_ts_exit, 'b--', label='Klinker Çıkış Sıcaklığı (Clinker Ts)')
    plt.title("Zamanla Fırın Çıkış Sıcaklık Değişimleri", fontsize=14)
    plt.xlabel("Zaman [saat]")
    plt.ylabel("Sıcaklık [K]")
    plt.legend()
    plt.savefig("kiln_temp_trend.png", dpi=200)
    plt.close()
    
    # Plot 5: Klinker Kalitesi
    plt.figure(figsize=(10, 6))
    solid_sum = np.sum(current_X[:, SOLID_SPECIES], axis=1) + 1e-9
    quality = current_X[:, IDX_C3S] / solid_sum
    plt.fill_between(z, quality, color='green', alpha=0.3, label='Klinkerleşme Oranı (% C3S)')
    plt.title("Klinkerleşme İlerlemesi", fontsize=14)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Oran")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig("kiln_clinker_quality.png", dpi=200)
    plt.close()

    print("5 Analiz dosyası başarıyla oluşturuldu.")
    # plt.show() # Not showing in headless environment

if __name__ == "__main__":
    run_and_visualize()
