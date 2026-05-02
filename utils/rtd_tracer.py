from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.flow import v_gas_model, v_solid_model


Phase = Literal["solid", "gas"]


@dataclass(frozen=True)
class RTDResult:
    phase: Phase
    t_exit_s: np.ndarray  # shape (n_exited,)
    n_released: int
    n_exited: int
    mean_s: float
    std_s: float


def _interp1(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, x, y)


def compute_rtd_lagrangian(
    *,
    z_grid_m: np.ndarray,
    epsilon_field: np.ndarray,
    fan: float,
    reactor_rpm: float,
    feed_rate: float,
    phase: Phase = "solid",
    n_particles: int = 20000,
    t_release_s: float = 600.0,
    dt_s: float = 0.2,
    t_max_s: float = 24 * 3600.0,
    D_ax_m2_s: float = 0.0,
    seed: int = 7,
) -> RTDResult:
    """
    Lagrangian tracer RTD (Residence Time Distribution) in 1D kiln axis.

    We release particles uniformly over a release window and propagate them with:
      dz = v(z) dt + sqrt(2 D_ax dt) dW

    Phase conventions (consistent with simulation):
    - solid: z=0 -> L (v_s >= 0), exit at z>=L
    - gas  : z=L -> 0 (v_g <= 0), exit at z<=0
    """
    z_grid_m = np.asarray(z_grid_m, dtype=np.float64)
    eps = np.asarray(epsilon_field, dtype=np.float64)
    if z_grid_m.ndim != 1 or eps.ndim != 1 or z_grid_m.shape[0] != eps.shape[0]:
        raise ValueError("z_grid_m and epsilon_field must be 1D arrays of equal length")
    if not np.all(np.diff(z_grid_m) > 0):
        raise ValueError("z_grid_m must be strictly increasing")

    L = float(z_grid_m[-1])
    rng = np.random.default_rng(seed)

    # release times (uniform over window)
    t0 = rng.uniform(0.0, float(t_release_s), size=int(n_particles)).astype(np.float64)

    if phase == "solid":
        z0 = np.zeros_like(t0)
        exit_cond = lambda z: z >= L
        v_fn = lambda z, e: v_solid_model(reactor_rpm, feed_rate, e)
        z_clip = (0.0, L)
    elif phase == "gas":
        z0 = np.full_like(t0, L)
        exit_cond = lambda z: z <= 0.0
        v_fn = lambda z, e: v_gas_model(fan, e)
        z_clip = (0.0, L)
    else:
        raise ValueError("phase must be 'solid' or 'gas'")

    z = z0.copy()
    t = np.zeros_like(z)
    active = np.zeros_like(z, dtype=bool)
    exited = np.zeros_like(z, dtype=bool)
    t_exit = np.full_like(z, np.nan)

    # Brownian term
    D = float(max(0.0, D_ax_m2_s))
    sqrt_2Ddt = np.sqrt(2.0 * D * float(dt_s)) if D > 0 else 0.0

    n_steps = int(np.ceil(float(t_max_s) / float(dt_s)))
    for _ in range(n_steps):
        # activate newly released particles
        active |= (t >= t0) & (~exited)
        if not np.any(active):
            # advance global time
            t += float(dt_s)
            continue

        # local epsilon from frozen field
        eps_loc = _interp1(z_grid_m, eps, z[active])
        v_loc = v_fn(z[active], eps_loc)

        dz = v_loc * float(dt_s)
        if D > 0:
            dz = dz + sqrt_2Ddt * rng.standard_normal(size=dz.shape[0])

        z[active] = z[active] + dz
        # keep inside for numerical stability; exit detection happens immediately after
        np.clip(z[active], z_clip[0] - 1e-6, z_clip[1] + 1e-6, out=z[active])

        # update time for all particles (including inactive) to keep release logic simple
        t += float(dt_s)

        # check exits for active particles
        act_idx = np.flatnonzero(active)
        z_act = z[act_idx]
        is_exit = exit_cond(z_act)
        if np.any(is_exit):
            idx_exit = act_idx[is_exit]
            exited[idx_exit] = True
            active[idx_exit] = False
            t_exit[idx_exit] = t[idx_exit] - t0[idx_exit]

        if np.all(exited):
            break

    t_exit_valid = t_exit[np.isfinite(t_exit)]
    mean_s = float(np.mean(t_exit_valid)) if t_exit_valid.size else float("nan")
    std_s = float(np.std(t_exit_valid)) if t_exit_valid.size else float("nan")

    return RTDResult(
        phase=phase,
        t_exit_s=t_exit_valid,
        n_released=int(n_particles),
        n_exited=int(t_exit_valid.size),
        mean_s=mean_s,
        std_s=std_s,
    )


def rtd_histogram(
    t_exit_s: np.ndarray, *, bins: int = 60, t_max_s: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_centers_s, pdf) where integral(pdf dt) ≈ 1."""
    t = np.asarray(t_exit_s, dtype=np.float64)
    if t.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if t_max_s is None:
        t_max_s = float(np.nanmax(t))
    edges = np.linspace(0.0, float(t_max_s), int(bins) + 1)
    hist, edges = np.histogram(t, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

