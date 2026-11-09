import numpy as np

from esf_sim.config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
    SimulationConfig,
)
from esf_sim.mimetic_potential import build_harmonic_equal_extrapolated_matrix


def test_mimetic_operator_is_configurable_only_with_equal_extrapolation():
    config = SimulationConfig(
        potential_operator=MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
    )
    assert config.potential_operator == MIMETIC_HARMONIC_LOG_RHS_OPERATOR


def test_harmonic_equal_extrapolated_matrix_has_gauge_and_positive_diagonal():
    coefficient = np.linspace(1.0, 3.0, 30).reshape(5, 6)
    matrix, scale = build_harmonic_equal_extrapolated_matrix(
        coefficient, 2.0, 3.0
    )
    assert matrix.shape == (30, 30)
    assert scale == 3.0
    assert matrix[0, 0] == 1.0
    assert matrix.getrow(0).nnz >= 1
    assert np.all(matrix.diagonal() > 0.0)
