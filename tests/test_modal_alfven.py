from __future__ import annotations

import numpy as np
import pytest

from esf_sim.alfven_1d import MU0
from esf_sim.modal_alfven import (
    ModalAlfvenLine,
    ModalAlfvenState,
    ModalBasis,
    TransverseMode,
)


def make_basis() -> ModalBasis:
    return ModalBasis(
        nx=17,
        ny=13,
        dx_m=1_250.0,
        dy_m=1_000.0,
        modes=(
            TransverseMode(1, 0, "cos"),
            TransverseMode(1, 0, "sin"),
            TransverseMode(0, 2, "cos"),
            TransverseMode(3, 1, "cos"),
            TransverseMode(2, 3, "sin"),
        ),
    )


def test_modes_are_exact_eigenvectors_of_periodic_x_equal_ghost_y_laplacian():
    basis = make_basis()
    for index, eigenvalue in enumerate(basis.eigenvalues_m2):
        mode = basis.mode_field(index)
        applied = basis.negative_laplacian(mode)
        np.testing.assert_allclose(
            applied,
            eigenvalue * mode,
            rtol=3.0e-13,
            atol=2.0e-19,
        )

    # Check the equal-extrapolated y rows directly, where an accidental
    # periodic or fixed-value implementation would differ.
    y_mode = basis.mode_field(2)
    expected_bottom = (
        2.0 * y_mode[0]
        - y_mode[0]
        - y_mode[1]
        + 2.0 * y_mode[0]
        - np.roll(y_mode[0], 1)
        - np.roll(y_mode[0], -1)
    )
    expected_bottom /= basis.dy_m**2  # dx term is zero for this y-only mode.
    np.testing.assert_allclose(
        basis.negative_laplacian(y_mode)[0], expected_bottom, atol=2.0e-19
    )


def test_modal_projection_lifting_and_area_weighted_parseval_identity():
    basis = make_basis()
    rng = np.random.default_rng(728)
    first = rng.normal(size=basis.mode_count)
    second = rng.normal(size=basis.mode_count)
    first_field = basis.lift(first)
    second_field = basis.lift(second)

    np.testing.assert_allclose(basis.project(first_field), first, atol=7.0e-15)
    np.testing.assert_allclose(basis.project(second_field), second, atol=7.0e-15)
    assert basis.field_inner_product(first_field, second_field) == pytest.approx(
        basis.modal_inner_product(first, second), rel=3.0e-15, abs=1.0e-8
    )


def test_constant_k_perp_zero_mode_is_excluded_and_all_fac_has_zero_mean():
    with pytest.raises(ValueError, match="k_perp=0"):
        ModalBasis(
            nx=8,
            ny=7,
            dx_m=1.0,
            dy_m=1.0,
            modes=(TransverseMode(0, 0, "cos"),),
        )

    basis = ModalBasis.low_wavenumber(
        nx=16,
        ny=12,
        dx_m=1_250.0,
        dy_m=1_000.0,
        max_x_wavenumber=3,
        max_y_wavenumber=2,
    )
    assert np.all(basis.eigenvalues_m2 > 0.0)
    assert TransverseMode(0, 0, "cos") not in basis.modes
    rng = np.random.default_rng(19)
    apar = 2.0e-4 * rng.standard_normal(basis.mode_count)
    current = basis.parallel_current_density_a_m2(apar)
    assert np.mean(current) == pytest.approx(0.0, abs=2.0e-13)


def test_apar_to_parallel_current_has_correct_shape_units_factor_and_sign():
    basis = make_basis()
    apar = np.zeros(basis.mode_count)
    apar[0] = 3.5e-4  # T m
    modal_current = basis.parallel_current_coefficients_a_m2(apar)
    assert modal_current.shape == (basis.mode_count,)
    assert modal_current[0] == pytest.approx(
        basis.eigenvalues_m2[0] * apar[0] / MU0
    )
    assert modal_current[0] > 0.0

    grid_current = basis.parallel_current_density_a_m2(apar)
    assert grid_current.shape == (basis.ny, basis.nx)
    np.testing.assert_allclose(
        grid_current,
        basis.negative_laplacian(basis.lift(apar)) / MU0,
        rtol=4.0e-13,
        atol=3.0e-16,
    )


def test_modal_state_conversion_preserves_voltage_and_vector_potential():
    line = ModalAlfvenLine(
        length_m=2.0e7,
        cells=20,
        alfven_speed_m_s=8.0e5,
        mode_count=3,
    )
    rng = np.random.default_rng(3)
    potential = 100.0 * rng.standard_normal((line.cells, line.mode_count))
    apar = 2.0e-4 * rng.standard_normal((line.cells, line.mode_count))
    state = ModalAlfvenState.from_potential_and_apar(
        potential, apar, alfven_speed_m_s=line.alfven_speed_m_s
    )
    np.testing.assert_allclose(line.potential_coefficients_v(state), potential)
    np.testing.assert_allclose(line.apar_coefficients_t_m(state), apar)


def test_uniform_characteristics_propagate_in_the_documented_directions():
    cells = 320
    speed = 9.0e5
    length = 4.0e7
    line = ModalAlfvenLine(
        length_m=length,
        cells=cells,
        alfven_speed_m_s=speed,
        mode_count=2,
        spatial_order=2,
        periodic=True,
    )
    s = line.s_m
    w_plus = np.zeros((cells, 2))
    w_minus = np.zeros_like(w_plus)
    w_plus[:, 0] = np.sin(2.0 * np.pi * s / length)
    w_minus[:, 1] = np.cos(2.0 * np.pi * s / length)
    initial = ModalAlfvenState(w_plus, w_minus)
    duration = 0.23 * length / speed
    result = line.advance(initial, duration_s=duration, cfl=0.45)

    exact_plus = np.sin(2.0 * np.pi * (s - speed * duration) / length)
    exact_minus = np.cos(2.0 * np.pi * (s + speed * duration) / length)
    plus_error = np.sqrt(np.mean((result.w_plus_v[:, 0] - exact_plus) ** 2))
    minus_error = np.sqrt(np.mean((result.w_minus_v[:, 1] - exact_minus) ** 2))
    assert plus_error < 1.2e-3
    assert minus_error < 1.2e-3
    assert np.max(np.abs(result.w_plus_v[:, 1])) == 0.0
    assert np.max(np.abs(result.w_minus_v[:, 0])) == 0.0


def test_left_voltage_response_returns_incoming_characteristic_and_positive_ds_fac():
    basis = ModalBasis(
        nx=15,
        ny=11,
        dx_m=2_000.0,
        dy_m=1_500.0,
        modes=(TransverseMode(1, 0, "cos"),),
    )
    line = ModalAlfvenLine(
        length_m=8.0e6,
        cells=8,
        alfven_speed_m_s=1.0e6,
        mode_count=1,
        spatial_order=1,
    )
    target_cell_apar = 2.0e-4
    state = ModalAlfvenState(
        np.full((line.cells, 1), 2.0 * line.alfven_speed_m_s * target_cell_apar),
        np.zeros((line.cells, 1)),
    )
    boundary_voltage = np.zeros(1)
    response = line.left_boundary_response(
        state,
        basis,
        boundary_voltage_coefficients_v=boundary_voltage,
    )

    np.testing.assert_array_equal(response.outgoing_w_minus_v, np.zeros(1))
    np.testing.assert_array_equal(response.incoming_w_plus_v, np.zeros(1))
    np.testing.assert_array_equal(response.boundary_apar_coefficients_t_m, np.zeros(1))
    expected_d_apar = target_cell_apar / (0.5 * line.spacing_m)
    assert response.current_gradient.d_apar_ds_t[0] == pytest.approx(expected_d_apar)
    assert response.current_gradient.modal_d_j_parallel_ds_a_m3[0] == pytest.approx(
        basis.eigenvalues_m2[0] * expected_d_apar / MU0
    )
    assert response.current_gradient.modal_d_j_parallel_ds_a_m3[0] > 0.0
    assert response.current_gradient.d_j_parallel_ds_a_m3.shape == (
        basis.ny,
        basis.nx,
    )
    assert np.mean(response.current_gradient.d_j_parallel_ds_a_m3) == pytest.approx(
        0.0, abs=1.0e-20
    )


def test_line_rhs_exposes_cfl_and_rejects_incompatible_shapes():
    line = ModalAlfvenLine(
        length_m=1.0e7,
        cells=10,
        alfven_speed_m_s=5.0e5,
        mode_count=2,
    )
    state = ModalAlfvenState(np.zeros((10, 2)), np.zeros((10, 2)))
    derivative = line.rhs(
        state, left_voltage_v=np.zeros(2), right_voltage_v=np.zeros(2)
    )
    assert derivative.w_plus_v.shape == (10, 2)
    assert derivative.w_minus_v.shape == (10, 2)
    assert line.maximum_stable_dt_s == pytest.approx(line.spacing_m / 5.0e5)
    assert line.cfl_number(0.4 * line.maximum_stable_dt_s) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="shape"):
        line.rhs(
            ModalAlfvenState(np.zeros((10, 1)), np.zeros((10, 1))),
            left_voltage_v=np.zeros(2),
            right_voltage_v=np.zeros(2),
        )


def test_remote_conductance_reflection_is_passive_and_has_matched_limit():
    line = ModalAlfvenLine(
        length_m=1.0e7,
        cells=10,
        alfven_speed_m_s=1.0e6,
        mode_count=2,
    )
    sigma_a = line.alfven_conductance_s
    assert line.conductance_reflection_coefficient(0.0) == pytest.approx(1.0)
    assert line.conductance_reflection_coefficient(sigma_a) == pytest.approx(0.0)
    assert line.conductance_reflection_coefficient(3.0 * sigma_a) == pytest.approx(
        -0.5
    )

    state = ModalAlfvenState(
        np.tile(np.array([2.0, -3.0]), (line.cells, 1)),
        np.zeros((line.cells, 2)),
    )
    outgoing = line.right_outgoing_characteristic_v(state)
    for conductance in (0.0, 0.2 * sigma_a, sigma_a, 5.0 * sigma_a):
        reflection = line.conductance_reflection_coefficient(conductance)
        incoming = line.right_incoming_characteristic_v(state, reflection)
        np.testing.assert_allclose(incoming, reflection * outgoing)
        # Characteristic energy flux is proportional to w**2; a passive
        # terminal never returns more energy than arrives.
        assert np.sum(incoming * incoming) <= np.sum(outgoing * outgoing) + 1e-14

    with pytest.raises(ValueError, match="non-negative"):
        line.conductance_reflection_coefficient(-1.0)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        line.right_incoming_characteristic_v(state, 1.01)


def test_matched_remote_boundary_transmits_pulse_without_reflection():
    cells = 160
    line = ModalAlfvenLine(
        length_m=1.6e7,
        cells=cells,
        alfven_speed_m_s=8.0e5,
        mode_count=1,
        spatial_order=2,
    )
    s = line.s_m
    pulse = np.exp(-((s - 0.78 * line.length_m) / (0.04 * line.length_m)) ** 2)
    initial = ModalAlfvenState(pulse[:, None], np.zeros((cells, 1)))
    result = line.advance(
        initial,
        duration_s=0.35 * line.length_m / line.alfven_speed_m_s,
        cfl=0.45,
        left_voltage_v=0.0,
        right_reflection_coefficient=0.0,
    )
    assert np.max(np.abs(result.w_minus_v)) == 0.0
    assert np.linalg.norm(result.w_plus_v) < 2.0e-3 * np.linalg.norm(
        initial.w_plus_v
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        line.rhs(
            initial,
            left_voltage_v=0.0,
            right_voltage_v=0.0,
            right_reflection_coefficient=0.0,
        )
