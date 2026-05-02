"""Scalar PDE assembly (core/model.py) wired to vector physics."""

from core.model import kiln_pde_system
from core.state import (
    IDX_CaCO3,
    IDX_EPSILON,
    IDX_T_S,
    N_CONTROLS,
    N_STATES,
    StateIdx,
    create_zero_control,
)


def test_kiln_pde_system_runs():
    x = [0.0] * N_STATES
    x[IDX_T_S] = 1500.0
    x[IDX_CaCO3] = 0.2
    u = create_zero_control()
    dx_dz = [0.0] * N_STATES
    out = kiln_pde_system(x, u, dx_dz)
    assert len(out["dx_dt"]) == N_STATES
    assert "solid_energy" in out["terms"]["energy"]
    eps_name = StateIdx(IDX_EPSILON).name
    assert eps_name in out["terms"]["mass"]
    assert out["terms"]["reaction"].shape == (5,)

