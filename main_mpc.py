# main_mpc.py

import time
import numpy as np
from simulation.engine import KilnSimulation
from mpc.controller import MPCController
from core.state import *

def run_mpc_digital_twin():
    L = 60.0
    N_CELLS = 100
    DT = 0.04
    SIM_HOURS = 7.0
    MPC_INTERVAL_SEC = 60.0  # MPC runs every 60s of sim time
    
    TOTAL_STEPS = int(SIM_HOURS * 3600 / DT)
    LOG_INTERVAL = int(3600 / DT)
    MPC_STEPS = int(MPC_INTERVAL_SEC / DT)

    sim = KilnSimulation(L, N_CELLS, DT)
    controller = MPCController(N_horizon=10, dt_mpc=MPC_INTERVAL_SEC)

    # Initial Condition
    initial_x = create_zero_state()
    initial_x[IDX_T_S] = 300.0
    initial_x[IDX_T_G] = 300.0
    initial_x[IDX_CaCO3] = 0.8
    initial_x[IDX_SiO2] = 0.14
    initial_x[IDX_Al2O3] = 0.04
    initial_x[IDX_Fe2O3] = 0.02
    initial_x[IDX_EPSILON] = 0.35
    sim.set_initial_condition(initial_x)

    # Initial Control
    u = create_zero_control()
    u[IDX_FUEL] = 6.0
    u[IDX_FAN] = 4000.0
    u[IDX_FEED] = 10.0
    u[IDX_REACTOR] = 3.5

    # Target State (Burning Zone Goal)
    x_target = create_zero_state()
    x_target[IDX_T_S] = 1450.0  # Target 1450 K for solid
    x_target[IDX_T_G] = 1800.0

    print("MPC-Controlled Rotary Kiln Digital Twin")
    print("=" * 60)
    print(f"{'Time(h)':>8} | {'T_s_exit':>8} | {'Fuel':>8} | {'Fan':>8} | {'CaCO3':>8} | {'C3S':>8}")
    print("-" * 70)

    wall_start = time.time()

    for step_idx in range(TOTAL_STEPS):
        
        # MPC Update
        if step_idx % MPC_STEPS == 0:
            # Measure exit cell (or average of burning zone)
            x_measured = sim.X[-1].copy() 
            
            # Compute action
            u_new = controller.compute_action(x_measured, x_target, u)
            
            # Update only controllable variables
            u[IDX_FUEL] = u_new[IDX_FUEL]
            u[IDX_FAN] = u_new[IDX_FAN]
            # (Feed and Reactor RPM kept constant or also controlled)

        X = sim.step(u)

        if step_idx % LOG_INTERVAL == 0:
            sim_hours = step_idx * DT / 3600.0
            print(f"{sim_hours:8.2f} | {X[-1, IDX_T_S]:8.1f} | {u[IDX_FUEL]:8.2f} | "
                  f"{u[IDX_FAN]:8.1f} | {np.mean(X[:, IDX_CaCO3]):8.3f} | {np.mean(X[:, IDX_C3S]):8.3f}")

    print("=" * 60)
    print(f"MPC Simulation completed in {time.time() - wall_start:.2f} s")

if __name__ == "__main__":
    run_mpc_digital_twin()
