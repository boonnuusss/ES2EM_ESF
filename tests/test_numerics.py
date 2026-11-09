import numpy as np

from esf_sim.numerics import (
    PDM7_LEFT_COEFFICIENTS,
    advection_rhs,
    reconstruct_interfaces,
    van_leer_slope,
)


def test_seventh_order_coefficients_are_conservative():
    assert np.isclose(np.sum(PDM7_LEFT_COEFFICIENTS), 1.0)


def test_reconstructions_preserve_constant_state():
    state = np.full((12, 16), 3.25)
    for scheme in ("first_order", "van_leer", "pdm7"):
        left, right = reconstruct_interfaces(
            state, axis=1, periodic=True, scheme=scheme, pdm_alpha=0.2
        )
        assert np.allclose(left, state)
        assert np.allclose(right, state)


def test_van_leer_slope_is_zero_at_extremum():
    minus = np.asarray([1.0, -1.0, 2.0])
    plus = np.asarray([-1.0, -2.0, 4.0])
    slope = van_leer_slope(minus, plus)
    assert slope[0] == 0.0
    assert slope[1] < 0.0
    assert slope[2] > 0.0


def test_pdm_step_is_positive_and_conservative():
    cells = 128
    dx = 1.0 / cells
    x = (np.arange(cells) + 0.5) * dx
    state = np.where((x > 0.3) & (x < 0.6), 1.0, 1.0e-10)[None, :]
    velocity_x = np.ones_like(state)
    velocity_y = np.zeros_like(state)
    initial_mass = np.sum(state)
    dt = 0.2 * dx
    for _ in range(50):
        state += dt * advection_rhs(
            state,
            velocity_x,
            velocity_y,
            dx=dx,
            dy=1.0,
            scheme="pdm7",
            pdm_alpha=0.2,
            periodic_y=True,
        )
    assert state.min() >= -1.0e-12
    assert state.max() <= 1.0 + 1.0e-12
    assert np.isclose(np.sum(state), initial_mass, rtol=1.0e-13)
