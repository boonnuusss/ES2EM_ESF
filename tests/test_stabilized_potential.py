import numpy as np

from esf_sim.config import SimulationConfig
from esf_sim.stabilized_potential import (
    ExponentialConservativePotentialSolver,
    build_exponential_equal_extrapolated_matrix,
    exponential_face_mean,
    point_centered_peclet_diagnostics,
)


def test_exponential_face_mean_is_positive_symmetric_and_consistent():
    left = np.array([1.0, 2.0, 1.0e-5, 7.0])
    right = np.array([1.0, 8.0, 1.0e3, 7.0 * (1.0 + 1.0e-10)])
    forward = exponential_face_mean(left, right)
    reverse = exponential_face_mean(right, left)
    assert np.all(forward > 0.0)
    np.testing.assert_allclose(forward, reverse, rtol=2.0e-12)
    np.testing.assert_allclose(forward[[0]], left[[0]])


def test_exponential_matrix_is_symmetric_monotone_and_conservative():
    coefficient = np.exp(np.linspace(-8.0, 8.0, 63)).reshape(7, 9)
    matrix, scale = build_exponential_equal_extrapolated_matrix(
        coefficient, 1250.0, 1000.0
    )
    assert scale == float(np.max(coefficient))
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T)
    np.testing.assert_allclose(matrix @ np.ones(matrix.shape[0]), 0.0, atol=1.0e-20)
    off_diagonal = matrix - np.diag(matrix.diagonal())
    assert np.max(off_diagonal) <= 0.0


def test_exponential_solver_handles_a_strong_positive_density_jump():
    config = SimulationConfig(
        nx=9,
        ny=9,
        x_max_km=10.0,
        y_min_km=250.0,
        y_max_km=258.0,
        dx_km=1.25,
        dy_km=1.0,
        potential_operator="point_centered_equation_2_21_log_coefficients",
        potential_boundary_y="equal_extrapolation",
    )
    density = np.full((9, 9), 1.0e8)
    density[:, 4:] = 1.0e12
    solver = ExponentialConservativePotentialSolver(config)
    potential, diagnostics = solver.solve(density)
    assert np.all(np.isfinite(potential))
    assert diagnostics.relative_residual < config.potential_rtol
    health = point_centered_peclet_diagnostics(
        density * solver.collision_frequency[:, None], config.dx_m, config.dy_m
    )
    assert health["cells_peclet_x_above_one"] > 0
