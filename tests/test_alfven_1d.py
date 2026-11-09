from __future__ import annotations

import numpy as np
import pytest

from esf_sim.alfven_1d import (
    MU0,
    AlfvenState,
    CharacteristicBoundary,
    ConductanceLoad,
    FieldLineGrid,
    PeriodicMidpointAlfvenSolver,
    UpwindAlfvenSolver,
    alfven_energy,
    characteristic_amplitudes,
    poynting_flux_w_m2,
    right_going_state,
)


B0 = 5.0e-5
VA = 1.0e6
RHO = B0**2 / (MU0 * VA**2)
LENGTH = 1.0e6


def make_grid(cells: int, density=RHO) -> FieldLineGrid:
    if np.isscalar(density):
        return FieldLineGrid.uniform(
            length_m=LENGTH,
            cells=cells,
            background_field_t=B0,
            mass_density_kg_m3=float(density),
        )
    spacing = LENGTH / cells
    s = (np.arange(cells) + 0.5) * spacing
    return FieldLineGrid(s, np.full(cells, B0), np.asarray(density))


def sine_error(cells: int) -> tuple[float, float]:
    grid = make_grid(cells)
    solver = PeriodicMidpointAlfvenSolver(grid)
    mode = 2
    wavenumber = 2.0 * np.pi * mode / LENGTH
    initial_velocity = np.sin(wavenumber * grid.s_m)
    initial = right_going_state(initial_velocity, grid)
    duration = 0.37 * LENGTH / VA
    dt = 0.35 * grid.spacing_m / VA
    result = solver.advance(initial, duration_s=duration, dt_s=dt)
    exact_velocity = np.sin(wavenumber * (grid.s_m - VA * duration))
    error = float(np.sqrt(np.mean((result.velocity_m_s - exact_velocity) ** 2)))
    right, left = characteristic_amplitudes(result, grid)
    walen_defect = float(np.linalg.norm(left) / np.linalg.norm(right))
    return error, walen_defect


def test_uniform_wave_has_second_order_phase_accuracy_and_walen_relation():
    errors = [sine_error(cells)[0] for cells in (64, 128, 256)]
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(orders > 1.85), (errors, orders)
    assert errors[-1] < 1.5e-3
    assert sine_error(96)[1] < 2.0e-13


def test_lossless_midpoint_step_preserves_total_energy_and_equipartition():
    grid = make_grid(96)
    rng = np.random.default_rng(42)
    initial = AlfvenState(
        rng.standard_normal(grid.cells),
        2.0e-9 * rng.standard_normal(grid.cells),
    )
    solver = PeriodicMidpointAlfvenSolver(grid)
    energy0 = alfven_energy(initial, grid)
    result = solver.advance(
        initial,
        duration_s=0.8 * LENGTH / VA,
        dt_s=0.4 * grid.spacing_m / VA,
    )
    energy1 = alfven_energy(result, grid)
    assert energy1.total_j_m2 == pytest.approx(energy0.total_j_m2, rel=2.0e-12)

    travelling = right_going_state(np.cos(2.0 * np.pi * grid.s_m / LENGTH), grid)
    travelling_energy = alfven_energy(travelling, grid)
    assert travelling_energy.kinetic_j_m2 == pytest.approx(
        travelling_energy.magnetic_j_m2, rel=2.0e-15
    )
    assert np.all(poynting_flux_w_m2(travelling, grid) >= 0.0)


def test_nonuniform_alfven_speed_preserves_lossless_periodic_energy():
    cells = 128
    base = make_grid(cells)
    density = RHO * (1.0 + 0.30 * np.sin(2.0 * np.pi * base.s_m / LENGTH)) ** -2
    grid = make_grid(cells, density)
    rng = np.random.default_rng(8)
    initial = AlfvenState(
        0.2 * rng.standard_normal(cells),
        1.0e-9 * rng.standard_normal(cells),
    )
    solver = PeriodicMidpointAlfvenSolver(grid)
    before = alfven_energy(initial, grid).total_j_m2
    result = solver.advance(
        initial,
        duration_s=0.6 * LENGTH / VA,
        dt_s=0.25 * grid.spacing_m / np.max(grid.alfven_speed_m_s),
    )
    after = alfven_energy(result, grid).total_j_m2
    assert after == pytest.approx(before, rel=4.0e-12)


def test_conductance_load_matches_analytic_reflection_coefficient():
    grid = make_grid(32)
    sigma_a = float(grid.alfven_conductance_s[0])
    matched = ConductanceLoad(sigma_a)
    assert matched.reflection_coefficient(sigma_a) == pytest.approx(0.0)
    assert ConductanceLoad(0.0).reflection_coefficient(sigma_a) == pytest.approx(1.0)
    assert ConductanceLoad(3.0 * sigma_a).reflection_coefficient(
        sigma_a
    ) == pytest.approx(-0.5)

    # Incoming Poynting power equals reflected power plus load Joule heating.
    incident_e = 0.08
    for sigma_p in (0.0, 0.25 * sigma_a, sigma_a, 4.0 * sigma_a):
        reflection = ConductanceLoad(sigma_p).reflection_coefficient(sigma_a)
        incident = sigma_a * incident_e**2
        reflected = sigma_a * (reflection * incident_e) ** 2
        load_e = (1.0 + reflection) * incident_e
        joule = sigma_p * load_e**2
        assert incident == pytest.approx(reflected + joule, rel=3.0e-15)


def test_wave_packet_reflects_from_conducting_load_with_expected_sign_and_size():
    grid = make_grid(800)
    sigma_a = float(grid.alfven_conductance_s[-1])
    expected = -0.5
    boundary = CharacteristicBoundary.from_loads(
        grid,
        left=ConductanceLoad(sigma_a),
        right=ConductanceLoad(3.0 * sigma_a),
    )
    solver = UpwindAlfvenSolver(grid, boundary)
    velocity = np.exp(-0.5 * ((grid.s_m - 0.30 * LENGTH) / (0.035 * LENGTH)) ** 2)
    initial = right_going_state(velocity, grid)
    initial_right, _ = characteristic_amplitudes(initial, grid)
    result = solver.advance(initial, duration_s=0.95 * LENGTH / VA, cfl=0.8)
    _, reflected_left = characteristic_amplitudes(result, grid)
    reflected_peak = reflected_left[np.argmax(np.abs(reflected_left))]
    measured = float(reflected_peak / np.max(initial_right))
    assert measured == pytest.approx(expected, abs=0.07)


def test_impedance_jump_produces_analytic_internal_reflection():
    cells = 1000
    base_grid = make_grid(cells)
    density = np.full(cells, RHO)
    density[base_grid.s_m >= 0.60 * LENGTH] = 4.0 * RHO
    grid = make_grid(cells, density)
    solver = UpwindAlfvenSolver(grid, periodic=False)

    velocity = np.exp(-0.5 * ((grid.s_m - 0.30 * LENGTH) / (0.03 * LENGTH)) ** 2)
    initial = right_going_state(velocity, grid)
    initial_right, _ = characteristic_amplitudes(initial, grid)
    result = solver.advance(initial, duration_s=0.72 * LENGTH / VA, cfl=0.8)
    _, reflected_left = characteristic_amplitudes(result, grid)

    z1 = np.sqrt(MU0 * RHO)
    z2 = np.sqrt(MU0 * 4.0 * RHO)
    expected = (z1 - z2) / (z1 + z2)
    reflected_peak = reflected_left[np.argmax(np.abs(reflected_left))]
    measured = float(reflected_peak / np.max(initial_right))
    assert measured == pytest.approx(expected, abs=0.06)


def test_invalid_field_line_parameters_are_rejected():
    with pytest.raises(ValueError, match="positive magnitude"):
        FieldLineGrid.uniform(
            length_m=1.0,
            cells=8,
            background_field_t=-1.0,
            mass_density_kg_m3=1.0,
        )
    with pytest.raises(ValueError, match="Pedersen"):
        ConductanceLoad(-1.0)
