"""
Fuel Intelligence - Rotary Kiln Visualization Tool (18-State Optimized)
"""

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
    DT           = 0.04  # Vektörize motor için kararlı adım
    SIM_HOURS    = 6.0
    TOTAL_STEPS  = int(SIM_HOURS * 3600 / DT)
    LOG_INTERVAL = 1000  # Terminal çıktısı sıklığı
    
    sim = KilnSimulation(L, N_CELLS, DT)
    
    # Başlangıç Koşulları (Sınır Şartları)
    initial_x = create_zero_state()
    initial_x[IDX_T_S]   = 300.0
    initial_x[IDX_T_G]   = 300.0
    initial_x[IDX_CaCO3] = 0.8
    initial_x[IDX_SiO2]  = 0.14
    initial_x[IDX_Al2O3] = 0.04
    initial_x[IDX_Fe2O3] = 0.02
    initial_x[IDX_EPSILON] = 0.4
    sim.set_initial_condition(initial_x)
    
    # Kontrol Girdileri (U)
    u = create_zero_control()
    u[IDX_FUEL]    = 17.0   # kg/s
    u[IDX_FAN]     = 2500.0 # m3/h
    u[IDX_FEED]    = 10.0  # kg/s
    u[IDX_REACTOR] = 2.0   # RPM
    
    # Veri Toplama Listeleri
    history_t = []
    history_tg_exit = []
    history_ts_exit = []
    
    print(f"Simülasyon başlatılıyor: {SIM_HOURS} saatlik operasyon...", flush=True)
    start_time = time.time()
    
    try:
        for step_idx in range(TOTAL_STEPS):
            # Vektörize motor adımı
            current_X = sim.step(u)
            
            # Her 10 dakikada bir (600 saniye / DT) veri kaydet
            if step_idx % int(600/DT) == 0:
                history_t.append((step_idx * DT) / 3600.0)
                history_ts_exit.append(current_X[-1, IDX_T_S])
                history_tg_exit.append(current_X[0, IDX_T_G]) # Gaz çıkışı (0. metre)
                
            if step_idx % LOG_INTERVAL == 0:
                prog = (step_idx / TOTAL_STEPS) * 100
                print(f"İlerleme: %{prog:.1f} | t: {step_idx*DT/3600:.1f} sa", end='\r', flush=True)

    except Exception as e:
        print(f"\n[HATA] Simülasyon sırasında hata oluştu: {e}")
        return

    print(f"\nSimülasyon bitti. Gerçek süre: {time.time() - start_time:.1f}s", flush=True)

    # --- 2. GÖRSELLEŞTİRME ---
    sns.set_theme(style="darkgrid")
    z = np.linspace(0, L, N_CELLS)
    
    # Plot 1: Sıcaklık Profili
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_T_G], 'r-', label='Gaz Sıcaklığı (Tg)', linewidth=2)
    plt.plot(z, current_X[:, IDX_T_S], 'b-', label='Katı Sıcaklığı (Ts)', linewidth=2)
    plt.title("Fırın Boyunca Sıcaklık Profili (Steady-State)", fontsize=12)
    plt.xlabel("Fırın Boyu [m]")
    plt.ylabel("Sıcaklık [K]")
    plt.legend()
    plt.savefig("kiln_temp_profile.png", dpi=200)
    plt.close()
    
    # Plot 2: Ana Fazlar
    plt.figure(figsize=(10, 6))
    plt.plot(z, current_X[:, IDX_CaCO3], label='CaCO3', linewidth=2)
    plt.plot(z, current_X[:, IDX_CaO], label='CaO', linewidth=2)
    plt.plot(z, current_X[:, IDX_C2S], label='C2S (Belit)', linewidth=2)
    plt.plot(z, current_X[:, IDX_C3S], label='C3S (Alit)', linewidth=2, color='darkgreen')
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
    
    # Plot 4: Çıkış Trendleri
    plt.figure(figsize=(10, 6))
    plt.plot(history_t, history_tg_exit, 'r--', label='Baca Gazı (Exit Tg)')
    plt.plot(history_t, history_ts_exit, 'b--', label='Klinker (Exit Ts)')
    plt.title("Zamanla Çıkış Sıcaklık Değişimi", fontsize=12)
    plt.xlabel("Zaman [saat]")
    plt.ylabel("Sıcaklık [K]")
    plt.legend()
    plt.savefig("kiln_temp_trend.png", dpi=200)
    plt.close()
    
    # Plot 5: Klinkerleşme Oranı (Litre Ağırlığı/Kalite Göstergesi)
    plt.figure(figsize=(10, 6))
    # SOLID_SPECIES içindeki tüm bileşenlerin toplamına bölerek normalize et
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

    print("Analiz grafikleri (PNG) başarıyla güncellendi.")

if __name__ == "__main__":
    run_and_visualize()