import yaml
import time
import numpy as np
import sys
import os
import logging

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

# ==============================================================
# LOGGER
# ==============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

logger = logging.getLogger("Kiln")

# ==============================================================
# CLOSURE
# ==============================================================

class ClosureTracker:

    def __init__(self):
        self.initial_mass = None
        self.initial_energy = None

    def total_mass(self, s):

        return np.sum(
            s.CaCO3 + s.CaO + s.SiO2 +
            s.Al2O3 + s.Fe2O3 +
            s.C2S + s.C3S +
            s.C3A + s.C4AF +
            s.CO2
        )

    def total_energy(self, s):

        return np.sum(s.Tg) + np.sum(s.Ts)

    def initialize(self, s):

        self.initial_mass = self.total_mass(s)
        self.initial_energy = self.total_energy(s)

    def check(self, s):

        mass_error = (
            self.total_mass(s) - self.initial_mass
        ) / max(self.initial_mass, 1e-12)

        energy_error = (
            self.total_energy(s) - self.initial_energy
        ) / max(self.initial_energy, 1e-12)

        return mass_error, energy_error


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

    solver = KilnSolver(config, transport, energy)

    closure = ClosureTracker()

    # ==========================================================
    # INITIAL STATE
    # ==========================================================

    raw_meal = config.get(
        "raw_meal_composition",
        {"CaCO3": 0.73, "SiO2": 0.21, "Al2O3": 0.04, "Fe2O3": 0.02}
    )

    solver.state.initialize_profiles(config)

    # Kimyasal kompozisyonu hammadde değerleri ile doldur
    solver.state.CaCO3[:] = raw_meal["CaCO3"]
    solver.state.SiO2[:]  = raw_meal["SiO2"]
    solver.state.Al2O3[:] = raw_meal["Al2O3"]
    solver.state.Fe2O3[:] = raw_meal["Fe2O3"]

    # Termal Gradyan Kurulumu:
    gas_inlet_temp = float(config["gas"].get("temp_inlet", 2200.0))

    # Katı sıcaklığı fırın genelinde 300K, gaz sıcaklığı ise sistemin
    # ısınmasını hızlandırmak için 800K başlangıç değerine atanır.
    solver.state.Ts.fill(300.0)
    solver.state.Tg.fill(800.0)

    closure.initialize(solver.state)

    # ==========================================================
    # SIMULATION PARAMETERS
    # ==============================================================

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
        f"{'Time':>7} | {'Ts_Out':>8} | {'Tg_Burn':>8} | "
        f"{'CaCO3_O':>8} | {'CaO_O':>8} | {'SiO2_O':>8} | "
        f"{'C2S_O':>8} | {'C3S_O':>8} | {'CO2_O':>8} | "
        f"{'Mass_O':>8} | {'M_ERR':>10} | {'E_ERR':>10}"
    )

    start_wall_time = time.time()
    last_log = -1e9

    # ==========================================================
    # HISTORY BUFFER
    # ==============================================================

    history_time = []
    history_Ts = []
    history_Tg = []
    history_CaCO3 = []
    history_CaO = []
    history_SiO2 = []
    history_C2S = []
    history_C3S = []
    history_CO2 = []

    history_mass_error = []
    history_energy_error = []

    # ==========================================================
    # MAIN LOOP
    # ==============================================================

    try:
        while t < t_final:

            # Sınır koşullarını her adımda zorla
            solver.state.Tg[-1] = gas_inlet_temp
            solver.state.Ts[0] = 300.0

            actual_dt = solver.solve_step(
                dt=dt,
                fuel_rate=fuel_rate,
                feed_rate=feed_rate,
                kiln_rpm=kiln_rpm,
                fan_rate=fan_rate
            )

            t += actual_dt
            s = solver.state

            mass_error, energy_error = closure.check(s)

            # Loglama ve History Kaydı
            if (t - last_log) >= 600.0:

                history_time.append(t)
                history_Ts.append(s.Ts.copy())
                history_Tg.append(s.Tg.copy())
                history_CaCO3.append(s.CaCO3.copy())
                history_CaO.append(s.CaO.copy())
                history_SiO2.append(s.SiO2.copy())
                history_C2S.append(s.C2S.copy())
                history_C3S.append(s.C3S.copy())
                history_CO2.append(s.CO2.copy())

                history_mass_error.append(mass_error)
                history_energy_error.append(energy_error)

                solid_mass_profile = (
                    s.CaCO3 + s.CaO + s.SiO2 +
                    s.Al2O3 + s.Fe2O3 +
                    s.C2S + s.C3S +
                    s.C3A + s.C4AF
                )

                print(
                    f"{t/3600:7.2f} | "
                    f"{s.Ts[-1]:8.1f} | "
                    f"{s.Tg[-1]:8.1f} | "
                    f"{s.CaCO3[-1]:8.4f} | "
                    f"{s.CaO[-1]:8.4f} | "
                    f"{s.SiO2[-1]:8.4f} | "
                    f"{s.C2S[-1]:8.4f} | "
                    f"{s.C3S[-1]:8.4f} | "
                    f"{s.CO2[-1]:8.4f} | "
                    f"{solid_mass_profile[-1]:8.4f}"
                    f" | {mass_error:10.3e}"
                    f" | {energy_error:10.3e}"
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

    lw = 0.8

    plt.style.use("seaborn-v0_8-muted")

    # ==========================================================
    # THERMAL PROFILE
    # ==============================================================

    plt.figure("Thermal Profile", figsize=(13, 7))

    plt.plot(z, history_Ts[-1], label="Solid Temperature (Ts)", color='blue', lw=lw)
    plt.plot(z, history_Tg[-1], label="Gas Temperature (Tg)", color='red', lw=lw)
    plt.plot(z, s.Tw, label="Wall Temperature (Tw)", linestyle='--', alpha=0.5, color='black', lw=lw)

    plt.title(f"Thermal Profiles Along Kiln (Time: {t/3600:.2f} h)")
    plt.xlabel("Kiln Length (m)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(alpha=0.3)

    # ==========================================================
    # CHEMISTRY PROFILE
    # ==============================================================

    plt.figure("Chemistry Profile", figsize=(14, 8))

    plt.plot(z, s.CaCO3, label="CaCO3", lw=lw)
    plt.plot(z, s.CaO, label="CaO", lw=lw)
    plt.plot(z, s.C2S, label="C2S", lw=lw)
    plt.plot(z, s.C3S, label="C3S", color='darkgreen', lw=lw)

    plt.plot(z, s.SiO2, label="SiO2", alpha=0.5, lw=0.6)
    plt.plot(z, s.Al2O3, label="Al2O3", alpha=0.5, lw=0.6)
    plt.plot(z, s.Fe2O3, label="Fe2O3", alpha=0.5, lw=0.6)
    plt.plot(z, s.C3A, label="C3A", alpha=0.5, lw=0.6)
    plt.plot(z, s.C4AF, label="C4AF", alpha=0.5, lw=0.6)

    plt.plot(z, s.CO2, label="CO2", linestyle=':', alpha=0.6)

    plt.title(f"Chemical Composition (Time: {t/3600:.2f} h)")
    plt.xlabel("Kiln Length (m)")
    plt.ylabel("Mass Fraction")
    plt.legend(ncol=3)
    plt.grid(alpha=0.3)

    # ==========================================================
    # TIME EVOLUTION
    # ==============================================================

    plt.figure("Outlet Temperature Evolution", figsize=(12, 6))

    outlet_Ts = [x[-1] for x in history_Ts]
    outlet_Tg = [x[-1] for x in history_Tg]

    plt.plot(np.array(history_time)/3600, outlet_Ts, label="Ts outlet", lw=lw)
    plt.plot(np.array(history_time)/3600, outlet_Tg, label="Tg outlet", lw=lw)

    plt.title("Temperature Evolution at Burner Side over Time")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()