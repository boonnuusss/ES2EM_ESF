import numpy as np

from esf_sim import _model_impl
from esf_sim.config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    BEST_BASELINE_POTENTIAL_OPERATOR,
    SimulationConfig,
)
from esf_sim.point_centered_potential import (
    EqualExtrapolatedLogPointCenteredPotentialSolver,
    build_equal_extrapolated_log_point_centered_matrix,
)


def test_equal_extrapolated_operator_annihilates_a_constant_without_gauge():
    y, x = np.mgrid[:8, :9]
    coefficient = np.exp(0.03 * x + 0.05 * y)
    matrix = build_equal_extrapolated_log_point_centered_matrix(
        coefficient, 1200.0, 900.0, impose_gauge=False
    )
    residual = matrix @ np.ones(matrix.shape[0])
    assert np.max(np.abs(residual)) < 1.0e-18


def test_equal_extrapolated_operator_has_one_identity_gauge_row():
    matrix = build_equal_extrapolated_log_point_centered_matrix(
        np.ones((7, 8)), 2.0, 3.0
    )
    row = matrix.getrow(0).toarray().ravel()
    assert row[0] == 1.0
    assert np.count_nonzero(row) == 1


def test_best_baseline_dispatches_extrapolated_solver_and_edge_derivative():
    config = SimulationConfig(
        nx=9,
        ny=8,
        x_max_km=10.0,
        y_min_km=250.0,
        y_max_km=257.0,
        dx_km=1.25,
        dy_km=1.0,
        potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
    )
    model = _model_impl.ESFModel(config)
    assert type(model.potential_solver) is EqualExtrapolatedLogPointCenteredPotentialSolver
    potential = np.arange(config.ny, dtype=float)[:, None] * np.ones((1, config.nx))
    velocity_x, _ = model.velocity_from_potential(potential)
    assert np.allclose(velocity_x[0], -0.5 / config.dy_m)
    assert np.allclose(velocity_x[-1], -0.5 / config.dy_m)