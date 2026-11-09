import numpy as np

from esf_sim._numerics_impl import reconstruct_interfaces as reference_reconstruct
from esf_sim.config import SimulationConfig
from esf_sim.numerics import advection_rhs as reference_advection
from esf_sim.optimized_numerics_v2 import (
    advection_rhs as optimized_advection,
    reconstruct_interfaces as optimized_reconstruct,
)
from esf_sim.optimized_potential import AdaptiveMonotonePotentialSolver
from esf_sim.production import variable_step_predictor


def test_optimized_reconstruction_matches_production_wrapper():
    rng = np.random.default_rng(42)
    quantity = rng.normal(size=(17, 13))
    for axis, periodic in ((1, True), (0, False)):
        for scheme in ("first_order", "van_leer", "pdm7"):
            expected = reference_reconstruct(
                quantity, axis=axis, periodic=periodic,
                scheme=scheme, pdm_alpha=0.4,
            )
            actual = optimized_reconstruct(
                quantity, axis=axis, periodic=periodic,
                scheme=scheme, pdm_alpha=0.4,
            )
            for left, right in zip(actual, expected):
                np.testing.assert_allclose(left, right, rtol=0.0, atol=2e-15)


def test_optimized_advection_is_roundoff_equivalent():
    rng = np.random.default_rng(7)
    n = np.exp(rng.normal(size=(19, 16)))
    u = rng.normal(size=n.shape)
    v = rng.normal(size=n.shape)
    kwargs = dict(dx=1250.0, dy=1000.0, scheme="pdm7", pdm_alpha=0.4,
                  periodic_y=False)
    expected = reference_advection(n, u, v, **kwargs)
    actual = optimized_advection(n, u, v, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-17)


def test_variable_step_predictor_is_exact_for_quadratic_state():
    times = [2.0, 1.25, -0.5]
    states = [np.array([t*t-3*t+2]) for t in times]
    target = 2.37
    actual = variable_step_predictor(states, times, target)
    np.testing.assert_allclose(actual, [target*target-3*target+2], atol=2e-15)


def test_constant_step_predictor_recovers_paper_coefficients():
    rng = np.random.default_rng(3)
    states = [rng.normal(size=8) for _ in range(3)]
    actual = variable_step_predictor(states, [2.0, 1.0, 0.0], 2.5)
    expected = 15/8*states[0]-5/4*states[1]+3/8*states[2]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-16)


def test_adaptive_solver_switch_is_one_way(monkeypatch):
    config = SimulationConfig(
        nx=9, ny=9, x_max_km=10.0, y_min_km=0.0, y_max_km=8.0,
        dx_km=1.25, dy_km=1.0, potential_operator=
        "point_centered_equation_2_21_log_coefficients",
        potential_boundary_y="equal_extrapolation",
    )
    solver = AdaptiveMonotonePotentialSolver(config, switch_peclet=0.75)
    monkeypatch.setattr(solver.centered, "peclet", lambda density: 0.8)
    solver.solve(np.ones((9, 9))*1e11)
    assert solver.mode == "exponential_conservative"
    monkeypatch.setattr(solver.centered, "peclet", lambda density: 0.0)
    solver.solve(np.ones((9, 9))*1e11)
    assert solver.mode == "exponential_conservative"

