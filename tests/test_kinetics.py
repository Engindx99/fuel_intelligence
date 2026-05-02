"""Kinetics: stoichiometric matrix S and dX = S @ r (conservation refactor directive)."""

import os

import numpy as np
import pytest
import yaml

from core.kinetics import (
    STOICHIOMETRY_MATRIX_S,
    compute_reaction_rates,
    compute_reaction_rates_vec,
    dX_kin,
    dXdt_kinetic_subspace,
)
from core.physics import mass_terms_vec
from core.state import IDX_Al2O3, IDX_CaCO3, IDX_CaO, IDX_C2S, IDX_C3S, IDX_CO2
from core.state import IDX_Fe2O3
from core.state import N_CONTROLS, N_STATES, IDX_SiO2, IDX_T_S

_S_EXPECTED = np.array(
    [
        [-1, 0, 0],
        [1, -1, -1],
        [0, -1, 0],
        [0, 1, -1],
        [0, 0, 1],
        [1, 0, 0],
    ],
    dtype=float,
)


def test_stoichiometry_matrix_matches_directive():
    np.testing.assert_array_equal(STOICHIOMETRY_MATRIX_S, _S_EXPECTED)


def _ones_X(n_cells: int = 5) -> np.ndarray:
    X = np.zeros((n_cells, N_STATES), dtype=float)
    X[:, IDX_T_S] = 1500.0
    X[:, IDX_CaCO3] = 0.3
    X[:, IDX_CaO] = 0.2
    X[:, IDX_C2S] = 0.15
    X[:, IDX_C3S] = 0.05
    X[:, IDX_SiO2] = 0.1
    return X


def test_dXdt_equals_r_dot_ST():
    r = np.array([[1.0, 2.0, 0.5], [0.0, 0.0, 0.0]], dtype=float)
    d = dXdt_kinetic_subspace(r)
    np.testing.assert_allclose(d, r @ STOICHIOMETRY_MATRIX_S.T, rtol=0, atol=0)


@pytest.mark.parametrize("rho_s", [2000.0])
def test_mass_terms_vectorized_kinetic_scatter(rho_s):
    X = _ones_X(3)
    u = np.zeros(N_CONTROLS, dtype=float)
    dC = mass_terms_vec(X, u)
    R = compute_reaction_rates_vec(X)
    r_mat = np.column_stack(
        [R["r_calcination"], R["r_C2S"], R["r_C3S"], R["r_C3A"], R["r_C4AF"]]
    )
    d_kin = dX_kin(r_mat) / rho_s
    order = [
        IDX_CaCO3,
        IDX_CaO,
        IDX_SiO2,
        IDX_C2S,
        IDX_C3S,
        IDX_CO2,
    ]
    for idx in order:
        np.testing.assert_allclose(dC[:, idx], d_kin[:, idx], rtol=1e-9, atol=1e-12)


def test_r_C3S_zero_below_1200K():
    X = _ones_X()
    X[:, IDX_T_S] = 1200.0 - 1.0
    R = compute_reaction_rates_vec(X)
    assert np.all(R["r_C3S"] < 5e-6)


def test_compute_reaction_rates_api():
    X = _ones_X(2)
    r = compute_reaction_rates(X)
    assert r.shape == (2, 5)
    assert np.all(np.isfinite(r))


def test_c3a_c4af_rates_near_zero_when_cold():
    X = _ones_X()
    X[:, IDX_T_S] = 850.0
    X[:, IDX_Al2O3] = 0.04
    X[:, IDX_Fe2O3] = 0.02
    R = compute_reaction_rates_vec(X)
    assert np.amax(np.abs(R["r_C3A"])) < 5e-12
    assert np.amax(np.abs(R["r_C4AF"])) < 5e-12


def test_c3a_positive_with_alumina_when_hot():
    X = _ones_X(2)
    X[:, IDX_Al2O3] = 0.04
    X[:, IDX_Fe2O3] = 0.02
    X[:, IDX_CaO] = 0.25
    R = compute_reaction_rates_vec(X)
    assert np.all(R["r_C3A"] > 1e-20)


@pytest.mark.parametrize(
    ("field",),
    [(IDX_CaCO3,), (IDX_CaO,), (IDX_C2S,), (IDX_C3S,)],
)
def test_non_negative_clamp_rates(field):
    X = np.zeros((2, N_STATES))
    X[:, IDX_T_S] = 1500.0
    X[:, field] = -1.0
    R = compute_reaction_rates_vec(X)
    assert np.all(np.isfinite(R["r_calcination"]))
    assert np.all(np.isfinite(R["r_C2S"]))
    assert np.all(np.isfinite(R["r_C3S"]))


def test_yaml_activation_energy_ranking_directive_7_2():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "configs", "model_config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    phys = cfg["physics"]
    assert phys["c3s_formation"]["Ea"] > phys["c2s_formation"]["Ea"]
