from __future__ import annotations

import numpy as np
import pytest

from esf_sim.config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    BEST_BASELINE_POTENTIAL_OPERATOR,
    SimulationConfig,
)
from esf_sim.fac_closure import (
    DEFAULT_OXYGEN_ION_MASS_KG,
    DimensionalFACMimeticClosure,
    DimensionalFACPointCenteredClosure,
    ParallelCurrentGradient,
    ParallelSurfaceCurrent,
    normalized_fac_rhs_s_inv,
)
from esf_sim.point_centered_potential import (
    build_equal_extrapolated_log_point_centered_matrix,
)
from esf_sim.potential import PotentialDiagnostics
from esf_sim.mimetic_potential import build_harmonic_equal_extrapolated_matrix


B0_T = 3.0e-5


def _config(*, gravity_m_s2: float = 0.0) -> SimulationConfig:
    return SimulationConfig(
        nx=9,
        ny=8,
        x_min_km=0.0,
        x_max_km=8.0,
        y_min_km=250.0,
        y_max_km=257.0,
        dx_km=1.0,
        dy_km=1.0,
        perturbation_wavelength_km=9.0,
        gravity_m_s2=gravity_m_s2,
        collision_scale_km=1.0e30,
        potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        potential_rtol=1.0e-10,
    )


def test_normalized_fac_rhs_has_derived_sign_and_si_scaling():
    density = np.full((2, 3), 4.0e11)
    collision = np.array([2.0, 5.0])
    gradient = np.array([[1.0e-12, -1.0e-12, 0.0], [2.0e-12, 0.0, -2.0e-12]])
    forcing = ParallelCurrentGradient(gradient)
    actual = normalized_fac_rhs_s_inv(
        density,
        collision,
        forcing,
        background_field_t=B0_T,
        ion_mass_kg=DEFAULT_OXYGEN_ION_MASS_KG,
    )
    expected = -B0_T * gradient / (
        DEFAULT_OXYGEN_ION_MASS_KG * density * collision[:, None]
    )
    assert np.array_equal(actual, expected)
    assert actual[0, 0] < 0.0
    assert actual[0, 1] > 0.0


class _UnforcedSpy:
    def __init__(self, shape: tuple[int, int]):
        self.calls = 0
        self.result = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)

    def solve(self, density: np.ndarray):
        self.calls += 1
        return self.result, PotentialDiagnostics(7, 3.0e-13, True, False)


def test_none_forcing_is_literal_historical_solver_branch():
    config = _config()
    spy = _UnforcedSpy((config.ny, config.nx))
    closure = DimensionalFACPointCenteredClosure(
        config, background_field_t=B0_T, unforced_solver=spy
    )
    density = np.full((config.ny, config.nx), 1.0e11)
    potential, diagnostics = closure.solve(density, forcing=None)
    assert spy.calls == 1
    assert potential is spy.result
    assert diagnostics.iterations == 7
    assert closure.last_fac_diagnostics is None


def test_manufactured_fac_solution_fixes_operator_sign_and_normalization():
    config = _config()
    density = np.full((config.ny, config.nx), 7.0e11)
    collision = config.collision_nu0_s * np.exp(
        -config.y_coordinates_m / config.collision_frequency_scale_height_m
    )
    coefficient = density * collision[:, None]
    ungauged = build_equal_extrapolated_log_point_centered_matrix(
        coefficient,
        config.dx_m,
        config.dy_m,
        impose_gauge=False,
    )
    phase = 2.0 * np.pi * np.arange(config.nx) / config.nx
    target = np.broadcast_to(2.5e7 * np.sin(phase), density.shape).copy()
    target -= np.mean(target)
    normalized_target_rhs = (ungauged @ target.ravel()).reshape(target.shape)
    gradient = (
        -DEFAULT_OXYGEN_ION_MASS_KG
        * coefficient
        / B0_T
        * normalized_target_rhs
    )
    assert abs(np.sum(gradient)) < 1.0e-13 * np.sum(np.abs(gradient))

    closure = DimensionalFACPointCenteredClosure(
        config, background_field_t=B0_T
    )
    actual, diagnostics = closure.solve(
        density, ParallelCurrentGradient(gradient)
    )
    assert np.allclose(actual, target, rtol=2.0e-12, atol=2.0e-6)
    assert diagnostics.relative_residual < 1.0e-10
    assert closure.last_fac_diagnostics is not None
    assert closure.last_fac_diagnostics.relative_net_current_gradient < 1.0e-13
    assert np.isinf(closure.last_fac_diagnostics.fac_to_gravity_ratio)


def test_forced_solution_is_linear_in_current_gradient():
    config = _config()
    density = np.full((config.ny, config.nx), 9.0e11)
    x = np.arange(config.nx, dtype=np.float64)
    gradient = np.broadcast_to(
        1.0e-13 * np.sin(2.0 * np.pi * x / config.nx), density.shape
    ).copy()
    closure = DimensionalFACPointCenteredClosure(
        config, background_field_t=B0_T
    )
    first, _ = closure.solve(density, ParallelCurrentGradient(gradient))
    second, _ = closure.solve(density, ParallelCurrentGradient(2.0 * gradient))
    assert np.allclose(second, 2.0 * first, rtol=2.0e-12, atol=1.0e-8)


def test_mimetic_manufactured_fac_solution_and_zero_branch():
    config = _config()
    density = np.full((config.ny, config.nx), 5.0e11)
    collision = config.collision_nu0_s * np.exp(
        -config.y_coordinates_m / config.collision_frequency_scale_height_m
    )
    coefficient = density * collision[:, None]
    matrix, scale = build_harmonic_equal_extrapolated_matrix(
        coefficient, config.dx_m, config.dy_m
    )
    phase = 2.0 * np.pi * np.arange(config.nx) / config.nx
    target = np.broadcast_to(1.2e7 * np.sin(phase), density.shape).copy()
    target -= np.mean(target)
    scaled_rhs = (matrix @ target.ravel()).reshape(target.shape)
    gradient = -DEFAULT_OXYGEN_ION_MASS_KG * scale / B0_T * scaled_rhs
    assert abs(np.sum(gradient)) < 1.0e-12 * np.sum(np.abs(gradient))

    closure = DimensionalFACMimeticClosure(config, background_field_t=B0_T)
    actual, diagnostics = closure.solve(
        density, ParallelCurrentGradient(gradient)
    )
    assert np.allclose(actual, target, rtol=2.0e-12, atol=2.0e-6)
    assert diagnostics.relative_residual < 1.0e-10

    spy = _UnforcedSpy(density.shape)
    with_spy = DimensionalFACMimeticClosure(
        config, background_field_t=B0_T, unforced_solver=spy
    )
    returned, _ = with_spy.solve(density, None)
    assert returned is spy.result
    assert spy.calls == 1


def test_unbalanced_parallel_current_gradient_is_rejected_not_hidden():
    config = _config()
    density = np.full((config.ny, config.nx), 1.0e11)
    closure = DimensionalFACPointCenteredClosure(
        config, background_field_t=B0_T
    )
    with pytest.raises(ValueError, match="no transverse return path"):
        closure.solve(
            density,
            ParallelCurrentGradient(np.ones_like(density) * 1.0e-12),
        )


def _continuum_manufactured_error(nx: int) -> float:
    length_m = 96_000.0
    dx_m = length_m / nx
    config = SimulationConfig(
        nx=nx,
        ny=8,
        x_min_km=0.0,
        x_max_km=(nx - 1) * dx_m / 1000.0,
        y_min_km=250.0,
        y_max_km=257.0,
        dx_km=dx_m / 1000.0,
        dy_km=1.0,
        perturbation_wavelength_km=length_m / 1000.0,
        gravity_m_s2=0.0,
        collision_scale_km=1.0e30,
        potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        potential_rtol=1.0e-10,
    )
    density = np.full((config.ny, config.nx), 6.0e11)
    collision = config.collision_nu0_s * np.exp(
        -config.y_coordinates_m / config.collision_frequency_scale_height_m
    )
    x = np.arange(nx) * dx_m
    wavenumber = 2.0 * np.pi / length_m
    target = np.broadcast_to(np.sin(wavenumber * x), density.shape).copy()
    gradient = (
        -DEFAULT_OXYGEN_ION_MASS_KG
        * density
        * collision[:, None]
        * wavenumber**2
        * target
        / B0_T
    )
    closure = DimensionalFACPointCenteredClosure(
        config, background_field_t=B0_T
    )
    actual, _ = closure.solve(density, ParallelCurrentGradient(gradient))
    return float(np.sqrt(np.mean((actual - target) ** 2)))


def test_continuum_dimensional_fac_manufactured_solution_is_second_order():
    errors = np.array([_continuum_manufactured_error(n) for n in (17, 33, 65)])
    orders = np.log(errors[:-1] / errors[1:]) / np.log(
        np.array([33.0 / 17.0, 65.0 / 33.0])
    )
    assert np.all(np.diff(errors) < 0.0)
    assert np.min(orders) > 1.95


def test_parallel_current_gradient_owns_immutable_copy():
    source = np.zeros((3, 4))
    forcing = ParallelCurrentGradient(source)
    source[0, 0] = 1.0
    assert forcing.d_j_parallel_ds_a_m3[0, 0] == 0.0
    with pytest.raises(ValueError):
        forcing.d_j_parallel_ds_a_m3[0, 0] = 2.0


def test_column_closure_requires_two_independent_positive_weights():
    config = _config()
    gravity = np.linspace(2.0e5, 8.0e5, config.ny)
    collision = np.linspace(3.0e5, 1.6e6, config.ny)
    with pytest.raises(ValueError, match="supplied together"):
        DimensionalFACPointCenteredClosure(
            config,
            background_field_t=B0_T,
            gravity_column_weight_m=gravity,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        DimensionalFACPointCenteredClosure(
            config,
            background_field_t=B0_T,
            gravity_column_weight_m=gravity,
            collision_column_weight_m=-collision,
        )


def test_point_column_components_use_separate_gravity_and_collision_moments():
    config = _config(gravity_m_s2=9.8)
    y = np.arange(config.ny, dtype=np.float64)[:, None]
    x = np.arange(config.nx, dtype=np.float64)[None, :]
    density = 5.0e11 * (
        1.0 + 0.08 * np.sin(2.0 * np.pi * x / config.nx) + 0.01 * y
    )
    gravity_weight = np.linspace(3.0e5, 9.0e5, config.ny)
    collision_weight = np.geomspace(4.0e5, 2.0e6, config.ny)
    closure = DimensionalFACPointCenteredClosure(
        config,
        background_field_t=B0_T,
        gravity_column_weight_m=gravity_weight,
        collision_column_weight_m=collision_weight,
    )
    components = closure.linear_components(density)
    collision = closure.collision_frequency_s[:, None]
    log_density = np.log(density)
    derivative = (
        np.roll(log_density, -1, axis=1)
        - np.roll(log_density, 1, axis=1)
    ) / (2.0 * config.dx_m)
    expected_gravity = (
        config.gravity_m_s2
        * derivative
        / collision
        * gravity_weight[:, None]
        / collision_weight[:, None]
    )
    expected_multiplier = -B0_T / (
        DEFAULT_OXYGEN_ION_MASS_KG
        * density
        * collision
        * collision_weight[:, None]
    )
    expected_gravity.flat[0] = 0.0
    expected_multiplier.flat[0] = 0.0
    np.testing.assert_allclose(components.gravity_rhs_s_inv, expected_gravity)
    np.testing.assert_allclose(components.fac_multiplier, expected_multiplier)
    assert components.forcing_representation == "surface_current"


def test_mimetic_column_surface_current_manufactured_solution():
    config = _config()
    density = np.full((config.ny, config.nx), 5.0e11)
    gravity_weight = np.linspace(3.0e5, 8.0e5, config.ny)
    collision_weight = np.geomspace(4.0e5, 2.0e6, config.ny)
    closure = DimensionalFACMimeticClosure(
        config,
        background_field_t=B0_T,
        gravity_column_weight_m=gravity_weight,
        collision_column_weight_m=collision_weight,
    )
    coefficient = (
        density
        * closure.collision_frequency_s[:, None]
        * collision_weight[:, None]
    )
    matrix, scale = build_harmonic_equal_extrapolated_matrix(
        coefficient, config.dx_m, config.dy_m
    )
    phase = 2.0 * np.pi * np.arange(config.nx) / config.nx
    target = np.broadcast_to(1.3e7 * np.sin(phase), density.shape).copy()
    rhs = (matrix @ target.ravel()).reshape(target.shape)
    surface_current = -DEFAULT_OXYGEN_ION_MASS_KG * scale / B0_T * rhs
    assert abs(np.sum(surface_current)) < 2.0e-13 * np.sum(
        np.abs(surface_current)
    )
    actual, diagnostics = closure.solve(
        density, ParallelSurfaceCurrent(surface_current)
    )
    np.testing.assert_allclose(actual, target, rtol=3.0e-12, atol=3.0e-6)
    assert diagnostics.relative_residual < 1.0e-10
    assert closure.last_fac_diagnostics.forcing_units == "A m^-2"
    with pytest.raises(TypeError, match="ParallelSurfaceCurrent"):
        closure.solve(density, ParallelCurrentGradient(surface_current))


def test_parallel_surface_current_owns_immutable_copy():
    source = np.zeros((3, 4))
    forcing = ParallelSurfaceCurrent(source)
    source[0, 0] = 1.0
    assert forcing.surface_current_a_m2[0, 0] == 0.0
    with pytest.raises(ValueError):
        forcing.surface_current_a_m2[0, 0] = 2.0


@pytest.mark.parametrize(
    "closure_type",
    [DimensionalFACPointCenteredClosure, DimensionalFACMimeticClosure],
)
def test_constant_two_column_limit_recovers_legacy_local_equation(closure_type):
    config = _config(gravity_m_s2=9.8)
    y = np.arange(config.ny, dtype=np.float64)[:, None]
    x = np.arange(config.nx, dtype=np.float64)[None, :]
    density = 6.0e11 * (
        1.0 + 0.05 * np.sin(2.0 * np.pi * x / config.nx) + 0.02 * y
    )
    q_volume = np.broadcast_to(
        2.0e-13 * np.sin(2.0 * np.pi * x / config.nx), density.shape
    ).copy()
    length_m = 1.75e6
    local = closure_type(config, background_field_t=B0_T)
    column = closure_type(
        config,
        background_field_t=B0_T,
        gravity_column_weight_m=np.full(config.ny, length_m),
        collision_column_weight_m=np.full(config.ny, length_m),
    )
    local_potential, _ = local.solve(
        density, ParallelCurrentGradient(q_volume)
    )
    column_potential, _ = column.solve(
        density, ParallelSurfaceCurrent(length_m * q_volume)
    )
    np.testing.assert_allclose(
        column_potential, local_potential, rtol=2.0e-12, atol=4.0e-6
    )
