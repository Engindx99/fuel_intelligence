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

        cp_s = 1000.0
        cp_g = 1050.0

        return np.sum(s.Ts) * cp_s + np.sum(s.Tg) * cp_g

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

    solver.state.CaCO3[:] = raw_meal["CaCO3"]
    solver.state.SiO2[:]  = raw_meal["SiO2"]
    solver.state.Al2O3[:] = raw_meal["Al2O3"]
    solver.state.Fe2O3[:] = raw_meal["Fe2O3"]

    gas_inlet_temp = float(config["gas"].get("temp_inlet", 2200.0))

    solver.state.Ts.fill(300.0)
    solver.state.Tg.fill(800.0)

    # 🔧 FIX: wall temperature was unused → make consistent
    solver.state.Tw.fill(900.0)

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
    last_log = 0.0

    history_time = []
    history_Ts = []
    history_Tg = []

    history_mass_error = []
    history_energy_error = []

    try:
        while t < t_final:

            # ==================================================
            # 🔧 FIX BOUNDARIES (CRITICAL)
            # ==================================================

            # hot gas enters from inlet
            solver.state.Tg[0] = gas_inlet_temp

            # solid leaves at outlet (NOT forced cold)
            solver.state.Ts[-1] = solver.state.Ts[-2]

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

            if (t - last_log) >= 600.0:

                history_time.append(t)
                history_Ts.append(s.Ts.copy())
                history_Tg.append(s.Tg.copy())

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
        print("\n[INFO] Stopped")

    print(f"\nDone in {time.time() - start_wall_time:.2f}s")

    # ==========================================================
    # PLOTS
    # ==============================================================

    s = solver.state
    z = np.linspace(0, float(config["kiln"]["length"]), s.N)

    plt.style.use("seaborn-v0_8-muted")

    plt.figure("Thermal Profile")
    plt.plot(z, s.Ts)
    plt.plot(z, s.Tg)
    plt.plot(z, s.Tw)

    plt.figure("Chemistry Profile")
    plt.plot(z, s.CaCO3)
    plt.plot(z, s.CaO)
    plt.plot(z, s.C2S)
    plt.plot(z, s.C3S)

    plt.show()


if __name__ == "__main__":
    main()