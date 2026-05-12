import yaml
import time
import numpy as np
import sys
import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ==============================================================
# PROJECT ROOT
# ==============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# ==============================================================
# IMPORTS
# ==============================================================

from core.solver import KilnSolver
from core.transport import TransportModel
from core.energy import EnergyModel
from core.kinetics import compute_clinker_kinetics_numba as kinetics


# ==============================================================
# MAIN
# ==============================================================

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(
        script_dir,
        "..",
        "configs",
        "model_config.yaml"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ==========================================================
    # PHYSICS ENGINES
    # ==========================================================

    transport = TransportModel(config)
    energy = EnergyModel(config)

    solver = KilnSolver(config, kinetics, transport, energy)

    # ==========================================================
    # INITIAL STATE
    # ==========================================================

    raw_meal = config.get(
        "raw_meal_composition",
        {"CaCO3": 0.73, "SiO2": 0.21, "Al2O3": 0.04, "Fe2O3": 0.02}
    )

    # initialize_profiles artık config'i doğrudan alarak daha stabil bir başlangıç yapar
    solver.state.initialize_profiles(config)
    
    # t=0'da fırın hammadde dolu varsayımı (Sayısal şokları önlemek için)
    # Katı akışı z=0 (Giriş) -> z=L (Çıkış) yönündedir.
    solver.state.CaCO3[:] = raw_meal["CaCO3"]
    solver.state.SiO2[:]  = raw_meal["SiO2"]
    solver.state.Al2O3[:] = raw_meal["Al2O3"]
    solver.state.Fe2O3[:] = raw_meal["Fe2O3"]
    
    # Gaz sıcaklığı başlangıç tahmini: Brülör tarafı (z=L, -1) sıcak, baca (z=0, 0) tarafı soğuk.
    # Solver içinde gaz denklemi sağdan sola (N-1 -> 0) çözüldüğü için bu gradyan kritiktir.
    gas_inlet_temp = float(config["gas"].get("temp_inlet", 2200.0))
    solver.state.Tg = np.linspace(800.0, gas_inlet_temp, solver.state.N)

    # ==========================================================
    # SIMULATION PARAMETERS (Hour Based Integration)
    # ==========================================================

    t = 0.0
    sim_hours = float(config["solver"].get("simulation_hours", 30.0))
    t_final = sim_hours * 3600.0
    dt = float(config["solver"]["dt"])

    fuel_rate = float(config["gas"].get("fuel_rate", 11.72))
    fan_rate = float(config["gas"].get("fan_rate", 800.0))
    kiln_rpm = float(config["kiln"].get("rpm", 2.0))
    feed_rate = float(config["material"].get("feed_rate", 125.0))

    print("\n[ROTARY KILN DIGITAL TWIN]")
    print(f"Simulation Duration : {sim_hours} h ({t_final} s)")
    print(f"Nodes               : {solver.state.N}")
    print(f"Time Step           : {dt:.3f} s")
    


    print(
        f"{'Time':>7} | {'Ts':>8} | {'Tg_Burn':>8} | "
        f"{'CaCO3':>8} | {'CaO':>8} | {'SiO2':>8} | "
        f"{'C2S':>8} | {'C3S':>8} | {'CO2':>8} | {'Mass':>8}"
    )

    start_wall_time = time.time()
    last_log = -1e9

    # ==========================================================
    # MAIN LOOP
    # ==============================================================

    try:
        while t < t_final:

            actual_dt = solver.solve_step(
                dt=dt,
                fuel_rate=fuel_rate,
                feed_rate=feed_rate,
                kiln_rpm=kiln_rpm,
                fan_rate=fan_rate
            )

            t += actual_dt

            # Her 10 dakikada bir (600s) log bas
            if (t - last_log) >= 600.0:

                s = solver.state
                
                # Katı Faz Kütle Dengesi
                solid_mass_profile = (s.CaCO3 + s.CaO + s.SiO2 + s.Al2O3 + s.Fe2O3 + 
                                     s.C2S + s.C3S + s.C3A + s.C4AF)

                # Loglama Güncellemesi: 
                # Ts[-1] ve Tg[-1] brülör tarafındaki (çıkış) değerleri gösterir.
                print(
                    f"{t/3600:7.2f} | "
                    f"{s.Ts[1]:8.1f} | "
                    f"{s.Tg[-1]:8.1f} | " 
                    f"{s.CaCO3[-1]:8.4f} | "
                    f"{s.CaO[-1]:8.4f} | "
                    f"{s.SiO2[-1]:8.4f} | "
                    f"{s.C2S[-1]:8.4f} | "
                    f"{s.C3S[-1]:8.4f} | "
                    f"{s.CO2[-1]:8.4f} | " 
                    f"{solid_mass_profile[-1]:8.4f}"
                )

                last_log = t

    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user.")

    print(f"\nCompleted in {time.time() - start_wall_time:.2f} s\n")

    # ==========================================================
    # VISUALIZATION
    # ==============================================================

    s = solver.state
    kiln_length = float(config["kiln"]["length"])
    z = np.linspace(0, kiln_length, s.N)

    plt.style.use("seaborn-v0_8-muted")

    # 1. Termal Profil: Ts ve Tg Birlikte
    plt.figure("Thermal Profile", figsize=(13, 7))
    plt.plot(z, s.Ts, label="Solid Temperature (Ts)", color='blue', linestyle='-', linewidth=2.0)
    plt.plot(z, s.Tg, label="Gas Temperature (Tg)", color='red', linestyle='-', linewidth=1.5)
    plt.plot(z, s.Tw, label="Wall Temperature (Tw)", color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.title(f"Thermal Profiles Along Kiln (Time: {t/3600:.2f} h)")
    plt.xlabel("Kiln Length (m) [0: Inlet, L: Outlet/Burner]")
    plt.ylabel("Temperature (K)")
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    # 2. Kimya Profili: Tüm bileşenler
    plt.figure("Chemistry Profile", figsize=(14, 8))
    
    plt.plot(z, s.CaCO3, label="CaCO3", linestyle='-', linewidth=1.2)
    plt.plot(z, s.CaO, label="CaO", linestyle='--', linewidth=1.0)
    plt.plot(z, s.C2S, label="C2S (Belite)", linestyle='-', linewidth=2.0)
    plt.plot(z, s.C3S, label="C3S (Alite)", linestyle='-', linewidth=2.5, color='darkgreen')
    
    # Diğer bileşenleri daha ince çizgilerle göstererek karmaşayı azaltıyoruz
    plt.plot(z, s.SiO2, label="SiO2", alpha=0.5, linewidth=0.8)
    plt.plot(z, s.Al2O3, label="Al2O3", alpha=0.5, linewidth=0.8)
    plt.plot(z, s.Fe2O3, label="Fe2O3", alpha=0.5, linewidth=0.8)
    plt.plot(z, s.C3A, label="C3A", alpha=0.5, linewidth=0.8)
    plt.plot(z, s.C4AF, label="C4AF", alpha=0.5, linewidth=0.8)
    
    plt.plot(z, s.CO2, label="CO2 (Solid-Tracked)", color='gray', linestyle=':', linewidth=1.0, alpha=0.6)

    plt.title(f"Chemical Composition Along Kiln (Time: {t/3600:.2f} h)")
    plt.xlabel("Kiln Length (m)")
    plt.ylabel("Mass Fraction")
    plt.legend(ncol=3, loc='upper right', fontsize='small')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()