import numpy as np

from esf_sim.experimental_boundary import build_harmonic_neumann_matrix


def test_neumann_operator_annihilates_constant_except_for_gauge_row():
    matrix, scale = build_harmonic_neumann_matrix(np.ones((7, 8)), 1.0, 1.0)
    action = matrix @ np.ones(56)
    assert scale == 1.0
    assert action[0] == 1.0
    assert np.max(np.abs(action[1:])) < 1.0e-12
