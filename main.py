"""
Rotary Kiln Digital Twin - Optimized Industrial Driver (18-State Version)
"""

import time
import numpy as np
from simulation.engine import KilnSimulation
from core.state import *
from core.validation import validate_state

def run_digital_twin():
    # -----------------------------
    # 1. SIM PARAMS (OPTIMIZED)
    # -----------------------------
    L = 60.0
    N_CELLS = 100
    DT = 0.04 # Fiziksel kararlılık için ideal adım
    SIM_HOURS = 10.0
    LOG_EVERY_MINUTES = 10  

    TOTAL_STEPS = int(SIM_HOURS * 3600 / DT)
    LOG_INTERVAL = max(1, int((LOG_EVERY_MINUTES * 60) / DT))

    # Simülasyon motorunu başlat
    sim = KilnSimulation(L, N_CELLS, DT)

    # -----------------------------
    # 2. INITIAL CONDITION (Sınır Şartları)
    # -----------------------------
    # Tüm fırın boyunca başlangıç değerlerini ata
    x0 = create_zero_state()
    x0[IDX_T_S] = 300.0
    x0[IDX_T_G] = 300.0
    x0[IDX_CaCO3] = 0.8
    x0[IDX_SiO2] = 0.14
    x0[IDX_Al2O3] = 0.04
    x0[IDX_Fe2O3] = 0.02
    x0[IDX_EPSILON] = 0.35

    sim.set_initial_condition(x0)

    # -----------------------------
    # 3. CONTROL INPUT (İşletme Parametreleri)
    # -----------------------------
    u = create_zero_control()
    u[IDX_FUEL] = 8.0      # kg/s yakıt
    u[IDX_FAN] = 5000.0    # m3/h hava
    u[IDX_FEED] = 10.0     # kg/s hammadde
    u[IDX_REACTOR] = 3.5   # rpm fırın hızı

    print("\nRotary Kiln Digital Twin (18-STATE OPTIMIZED MODE)")
    print("=" * 110)
    print(f"Config: {N_CELLS} cells | DT: {DT}s | Total Sim: {SIM_HOURS} hours")
    print("-" * 110)

    # Başlıklar
    print(f"{'t(min)':>8} | {'Ts(K)':>8} | {'Tg(K)':>8} | {'CaCO3':>8} | {'CaO':>8} | {'C2S':>8} | {'C3S':>8} | {'TotalSol':>8} | {'Speed(x)':>10}")
    print("-" * 110)

    wall_start = time.time()

    # -----------------------------
    # 4. SIM LOOP
    # -----------------------------
    try:
        for step in range(TOTAL_STEPS):
            # Ana simülasyon adımı (Vektörize edilmiş)
            X = sim.step(u)

            # Loglama ve Doğrulama
            if step % LOG_INTERVAL == 0:
                # Sayısal kararlılık kontrolü
                if not np.all(np.isfinite(X)):
                    raise RuntimeError(f"Numerical instability (NaN) at t = {step*DT/60:.2f} min")

                # Fiziksel doğrulama (Kütle ve sıcaklık sınırları)
                validate_state(X, step)

                t_min = (step * DT) / 60
                elapsed_wall = time.time() - wall_start
                real_time_ratio = (t_min * 60) / max(0.001, elapsed_wall)
                
                avg = lambda i: np.mean(X[:, i])

                # 18 bileşenli yapıda toplam katı kütlesi takibi
                # CaCO3 (2) ile Fe2O3 (10) arasındaki tüm katıları toplar
                current_solids = np.sum([avg(i) for i in range(StateIdx.CaCO3, StateIdx.Fe2O3 + 1)])

                print(f"{t_min:8.1f} | "
                      f"{avg(IDX_T_S):8.1f} | "
                      f"{avg(IDX_T_G):8.1f} | "
                      f"{avg(IDX_CaCO3):8.3f} | "
                      f"{avg(IDX_CaO):8.3f} | "
                      f"{avg(IDX_C2S):8.3f} | "
                      f"{avg(IDX_C3S):8.3f} | "
                      f"{current_solids:8.3f} | "
                      f"{real_time_ratio:10.1f}x")

    except Exception as e:
        print(f"\n[ERROR] Simülasyon durduruldu: {e}")
        return

    # -----------------------------
    # 5. FINAL REPORT
    # -----------------------------
    wall_time = time.time() - wall_start
    print("\n" + "=" * 110)
    print(f"SIMULATION COMPLETE | Total Wall Time: {wall_time:.2f}s | Final Speedup: {(SIM_HOURS*3600)/wall_time:.1f}x")
    print("=" * 110)

    def report_cell(name, row):
        print(f"\n[{name}]")
        print(f"Temps -> Solid: {row[IDX_T_S]:.1f} K, Gas: {row[IDX_T_G]:.1f} K")
        print(f"Clinker -> C2S: {row[IDX_C2S]:.4f}, C3S: {row[IDX_C3S]:.4f}, CaO: {row[IDX_CaO]:.4f}")
        print(f"Rest -> PHI: {row[IDX_PHI]:.4f}, EPS: {row[IDX_EPSILON]:.4f}")

    report_cell("Fırın Çıkışı (Klinker)", X[-1])

if __name__ == "__main__":
    run_digital_twin()