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
        {"CaCO3": 0.82, "SiO2": 0.13, "Al2O3": 0.03, "Fe2O3": 0.02}
    )

    ambient_temp = float(config["material"].get("temp_inlet", 300.0))
    gas_inlet_temp = float(config["gas"].get("temp_inlet", 2200.0))

    solver.state.initialize_profiles(
        T_ambient=ambient_temp,
        T_gas_inlet=gas_inlet_temp,
        raw_meal_comp=raw_meal
    )

    # ==========================================================
    # SIMULATION PARAMETERS
    # ==========================================================

    t = 0.0
    t_final = 20.0 * 3600.0
    dt = float(config["solver"]["dt"])

    fuel_rate = float(config["gas"].get("fuel_rate", 4.0))
    fan_rate = float(config["gas"].get("fan_rate", 800.0))
    kiln_rpm = float(config["kiln"].get("rpm", 2.0))
    feed_rate = float(config["material"].get("feed_rate", 125.0))

    print("\n[ROTARY KILN DIGITAL TWIN]")
    print(f"Simulation Duration : {t_final/3600:.1f} h")
    print(f"Nodes               : {solver.state.N}")
    print(f"Time Step           : {dt:.3f} s")
    print("-" * 120)

    print(
        f"{'Time':>7} | {'Ts_out':>8} | {'Tg_out':>8} | "
        f"{'CaCO3':>8} | {'CaO':>8} | {'C2S':>8} | {'C3S':>8} | {'Mass':>8}"
    )

    print("-" * 120)

    start_wall_time = time.time()
    last_log = -1e9

    # ==========================================================
    # MAIN LOOP
    # ==============================================================

    while t < t_final:

        actual_dt = solver.solve_step(
            dt=dt,
            fuel_rate=fuel_rate,
            feed_rate=feed_rate,
            kiln_rpm=kiln_rpm,
            fan_rate=fan_rate
        )

        t += actual_dt

        if (t - last_log) >= 600.0:

            s = solver.state

            print(
                f"{t/3600:7.2f} | "
                f"{s.Ts[-1]:8.1f} | "
                f"{s.Tg[-1]:8.1f} | "
                f"{s.CaCO3[-1]:8.4f} | "
                f"{s.CaO[-1]:8.4f} | "
                f"{s.C2S[-1]:8.4f} | "
                f"{s.C3S[-1]:8.4f} | "
                f"{np.mean(s.CaCO3 + s.CaO + s.C2S + s.C3S):8.4f}"
            )

            last_log = t

    print("-" * 120)
    print(f"Completed in {time.time() - start_wall_time:.2f} s\n")

    # ==========================================================
    # VISUALIZATION
    # ==============================================================

    s = solver.state

    kiln_length = float(config["kiln"]["length"])
    z = np.linspace(0, kiln_length, s.N)

    plt.style.use("seaborn-v0_8-muted")

    plt.figure("Thermal Profile", figsize=(13, 7))
    plt.plot(z, s.Ts, label="Solid")
    plt.plot(z, s.Tg, label="Gas")
    plt.plot(z, s.Tw, label="Wall")
    plt.xlabel("Kiln Length (m)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.figure("Chemistry", figsize=(14, 8))
    plt.plot(z, s.CaCO3, label="CaCO3")
    plt.plot(z, s.CaO, label="CaO")
    plt.plot(z, s.C2S, label="C2S")
    plt.plot(z, s.C3S, label="C3S")
    plt.legend(ncol=2)
    plt.grid(alpha=0.2)

    plt.show()


if __name__ == "__main__":
    main()