import numpy as np

from esf_sim.config import SimulationConfig
from esf_sim.experimental_staggered import (
    StaggeredESFModel,
    staggered_face_velocity,
)
from esf_sim.model import ESFModel, initial_density


def test_staggered_faces_equal_center_then_average_on_uniform_grid():
    config = SimulationConfig(nx=9, ny=9, x_max_km=10.0, y_max_km=258.0)
    rng = np.random.default_rng(5)
    potential = rng.standard_normal((9, 9))
    model = ESFModel(config)
    velocity_x, velocity_y = model.velocity_from_potential(potential)
    expected_x = 0.5 * (velocity_x + np.roll(velocity_x, -1, axis=1))
    expected_y = np.zeros((10, 9))
    expected_y[1:-1] = 0.5 * (velocity_y[:-1] + velocity_y[1:])
    actual_x, actual_y = staggered_face_velocity(
        potential, config.dx_m, config.dy_m
    )
    assert np.allclose(actual_x, expected_x, rtol=0.0, atol=1.0e-15)
    assert np.allclose(actual_y, expected_y, rtol=0.0, atol=1.0e-15)


def test_staggered_rhs_matches_baseline_rhs():
    config = SimulationConfig(
        nx=9,
        ny=9,
        x_max_km=10.0,
        y_max_km=258.0,
        potential_rtol=1.0e-10,
    )
    density = initial_density(config)
    baseline, _ = ESFModel(config).evaluate_rhs(density)
    staggered, _ = StaggeredESFModel(config).evaluate_rhs(density)
    assert np.allclose(staggered, baseline, rtol=1.0e-11, atol=1.0e-8)
