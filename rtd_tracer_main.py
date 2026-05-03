import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from simulation.engine import KilnSimulation
from core.state import *
from utils.rtd_tracer import compute_rtd_lagrangian, rtd_histogram


def _steady_snapshot(L=60.0, N_CELLS=100, DT=0.04, SIM_HOURS=1.25, log_every_s=120.0):
    # Aynı global thermal YAML (calibration_reference) — RTD süresi bağımsız.
    sim = KilnSimulation(L, N_CELLS, DT)

    u = create_zero_control()
    u[IDX_FUEL] = 16.0
    u[IDX_FAN] = 800.0
    u[IDX_FEED] = 10.0
    u[IDX_REACTOR] = 2.0

    initial_x = create_zero_state(N_CELLS)
    zs = np.linspace(0.0, L, N_CELLS)
    initial_x[:, IDX_T_S] = sim._T_s_in
    initial_x[:, IDX_T_G] = sim.gas_ic_axial_k(u, zs / L)
    initial_x[:, IDX_CaCO3] = 0.8
    initial_x[:, IDX_SiO2] = 0.14
    initial_x[:, IDX_Al2O3] = 0.04
    initial_x[:, IDX_Fe2O3] = 0.02
    initial_x[:, IDX_EPSILON] = 0.4
    sim.set_initial_condition(initial_x)

    TOTAL_STEPS = int(SIM_HOURS * 3600 / DT)
    log_every = max(1, int(log_every_s / DT))
    for k in range(TOTAL_STEPS):
        sim.step(u)
        if k % log_every == 0:
            print(f"steady snapshot: t={k*DT/3600.0:.2f} h", flush=True)

    z = np.linspace(0, L, N_CELLS)
    return sim, u, z, sim.X.copy()


def main():
    sim, u, z, X = _steady_snapshot()
    eps_field = X[:, IDX_EPSILON]

    D_solid = 0.015
    D_gas = 0.002

    print("tracer RTD (solid) ...", flush=True)
    # Long tail for solids: increase t_max until exited/released ~= 95%+ before trusting mean τ.
    res_s = compute_rtd_lagrangian(
        z_grid_m=z,
        epsilon_field=eps_field,
        fan=float(u[IDX_FAN]),
        reactor_rpm=float(u[IDX_REACTOR]),
        feed_rate=float(u[IDX_FEED]),
        phase="solid",
        n_particles=16000,
        t_release_s=400.0,
        dt_s=2.25,
        t_max_s=84 * 3600.0,
        D_ax_m2_s=D_solid,
        seed=11,
    )
    print("tracer RTD (gas) ...", flush=True)
    res_g = compute_rtd_lagrangian(
        z_grid_m=z,
        epsilon_field=eps_field,
        fan=float(u[IDX_FAN]),
        reactor_rpm=float(u[IDX_REACTOR]),
        feed_rate=float(u[IDX_FEED]),
        phase="gas",
        n_particles=25000,
        t_release_s=25.0,
        dt_s=0.035,
        t_max_s=180.0,
        D_ax_m2_s=D_gas,
        seed=13,
    )

    sns.set_theme(style="darkgrid")

    def plot_one(res, fname):
        bins = 100 if res.phase == "solid" else 72
        centers, pdf = rtd_histogram(res.t_exit_s, bins=bins)

        plt.figure(figsize=(10, 6))
        if res.phase == "gas":
            plt.plot(centers, pdf, linewidth=1)
            plt.xlabel("Residence time [s]")
            plt.ylabel("E(t) [1/s]")
            mean_lab = res.mean_s
            mean_txt = f"{mean_lab:.2f} s"
        else:
            plt.plot(centers / 60.0, pdf * 60.0, linewidth=1)
            plt.xlabel("Residence time [min]")
            plt.ylabel("E(t) [1/min]")
            mean_txt = f"{res.mean_s/60.0:.1f} min"

        pct = (
            (100.0 * res.n_exited / res.n_released) if res.n_released > 0 else float("nan")
        )
        plt.title(
            f"RTD (Lagrangian tracer) - {res.phase} | "
            f"mean={mean_txt} | exited={res.n_exited}/{res.n_released} ({pct:.0f}%)",
            fontsize=11,
        )
        plt.savefig(fname, dpi=200)
        plt.close()

    plot_one(res_s, "kiln_rtd_solid.png")
    plot_one(res_g, "kiln_rtd_gas.png")

    print(
        "RTD plots generated: kiln_rtd_solid.png, kiln_rtd_gas.png",
        flush=True,
    )


if __name__ == "__main__":
    main()
