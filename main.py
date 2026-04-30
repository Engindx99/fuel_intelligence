"""
Rotary Kiln Digital Twin - Industrial Driver

Enhancements:
- Runtime physical validation
- Stability monitoring (CFL + NaN checks)
- Mass/energy sanity hooks
- Safe logging
"""

import time
import numpy as np
from simulation.engine import KilnSimulation
from core.state import *
from core.validation import validate_state


def run_digital_twin():

    # -----------------------------
    # 1. SIM PARAMS
    # -----------------------------
    L = 60.0
    N_CELLS = 100
    DT = 0.04
    SIM_HOURS = 7.0

    TOTAL_STEPS = int(SIM_HOURS * 3600 / DT)
    LOG_INTERVAL = max(1, int(3600 / DT))

    sim = KilnSimulation(L, N_CELLS, DT)

    # -----------------------------
    # 2. INITIAL CONDITION
    # -----------------------------
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
    # 3. CONTROL INPUT
    # -----------------------------
    u = create_zero_control()

    u[IDX_FUEL] = 8.0
    u[IDX_FAN] = 5000.0
    u[IDX_FEED] = 10.0
    u[IDX_REACTOR] = 3.5

    print("\nRotary Kiln Digital Twin (INDUSTRIAL MODE)")
    print("=" * 70)
    print(f"Cells     : {N_CELLS}")
    print(f"dt        : {DT} s")
    print(f"Sim time  : {SIM_HOURS} h")
    print(f"Steps     : {TOTAL_STEPS:,}")
    print("=" * 70)

    print(f"{'t(h)':>8} | {'Ts':>8} | {'Tg':>8} | {'CaCO3':>8} | {'CaO':>8} | {'C2S':>8} | {'C3S':>8} | {'total':>8} | {'wall(s)':>10}")
    print("-" * 100)

    wall_start = time.time()

    # -----------------------------
    # 4. SIM LOOP
    # -----------------------------
    for step in range(TOTAL_STEPS):

        X = sim.step(u)

        # -----------------------------
        # SAFETY CHECKS (CRITICAL)
        # -----------------------------

        # NaN / Inf check
        if np.any(~np.isfinite(X)):
            raise RuntimeError(f"Numerical instability detected at step {step}")

        # Physical validation
        validate_state(X, step)

        # -----------------------------
        # LOGGING
        # -----------------------------
        if step % LOG_INTERVAL == 0:

            t_h = step * DT / 3600
            elapsed = time.time() - wall_start

            avg = lambda i: np.mean(X[:, i])

            total = (
                avg(IDX_CaCO3) +
                avg(IDX_CaO) +
                avg(IDX_C2S) +
                avg(IDX_C3S)
            )

            print(f"{t_h:8.2f} | "
                  f"{avg(IDX_T_S):8.1f} | "
                  f"{avg(IDX_T_G):8.1f} | "
                  f"{avg(IDX_CaCO3):8.3f} | "
                  f"{avg(IDX_CaO):8.3f} | "
                  f"{avg(IDX_C2S):8.3f} | "
                  f"{avg(IDX_C3S):8.3f} | "
                  f"{total:8.3f} | "
                  f"{elapsed:10.1f}")

    # -----------------------------
    # 5. FINAL REPORT
    # -----------------------------
    wall_time = time.time() - wall_start

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Wall time : {wall_time:.2f} s")
    print(f"Speedup   : {(SIM_HOURS*3600)/wall_time:.1f}x real time")

    def report(name, row):
        print(f"\n[{name}]")
        print(f"T_s   : {row[IDX_T_S]:.2f}")
        print(f"T_g   : {row[IDX_T_G]:.2f}")
        print(f"CaCO3 : {row[IDX_CaCO3]:.4f}")
        print(f"CaO   : {row[IDX_CaO]:.4f}")
        print(f"C2S   : {row[IDX_C2S]:.4f}")
        print(f"C3S   : {row[IDX_C3S]:.4f}")

    report("EXIT", X[-1])
    report("MID", X[len(X)//2])


if __name__ == "__main__":
    run_digital_twin()