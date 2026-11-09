from __future__ import annotations

import numpy as np
import pytest

from esf_sim import _model_impl
from esf_sim.alfven_1d import MU0
from esf_sim.bidirectional_em import (
    DimensionalBidirectionalModel,
    DimensionalEMState,
    PhysicalCouplingConvergenceError,
    PhysicalFeedbackConfig,
    density_mode_amplitude,
)
from esf_sim.config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
    SimulationConfig,
)
from esf_sim.fac_closure import DimensionalFACMimeticClosure
from esf_sim.modal_alfven import (
    ModalAlfvenLine,
    ModalAlfvenState,
    ModalBasis,
    TransverseMode,
)


B0_T = 3.0e-5


def _config() -> SimulationConfig:
    return SimulationConfig(
        nx=9,
        ny=9,
        x_min_km=0.0,
        x_max_km=8.0,
        y_min_km=250.0,
        y_max_km=258.0,
        dx_km=1.0,
        dy_km=1.0,
        perturbation_wavelength_km=9.0,
        scheme="pdm7",
        pdm_alpha=0.4,
        dt_s=0.05,
        cfl_limit=0.25,
        potential_operator=MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        potential_rtol=1.0e-10,
    )


def _components(
    feedback: PhysicalFeedbackConfig, *, line_spatial_order: int = 1
):
    config = _config()
    closure = DimensionalFACMimeticClosure(
        config, background_field_t=B0_T
    )
    legacy = _model_impl.ESFModel.__new__(_model_impl.ESFModel)
    legacy.config = config
    legacy.potential_solver = closure
    basis = ModalBasis(
        nx=config.nx,
        ny=config.ny,
        dx_m=config.dx_m,
        dy_m=config.dy_m,
        modes=(
            TransverseMode(1, 0, "cos"),
            TransverseMode(1, 0, "sin"),
            TransverseMode(0, 1, "cos"),
        ),
    )
    line = ModalAlfvenLine(
        length_m=8.0e6,
        cells=8,
        alfven_speed_m_s=8.0e5,
        mode_count=basis.mode_count,
        spatial_order=line_spatial_order,
    )
    model = DimensionalBidirectionalModel(
        legacy, closure, basis, line, feedback
    )
    y = np.arange(config.ny, dtype=np.float64)[:, None]
    x = np.arange(config.nx, dtype=np.float64)[None, :]
    density = 5.0e11 * (
        1.0
        + 0.06 * np.cos(2.0 * np.pi * x / config.nx)
        + 0.015 * (y - np.mean(y)) / config.ny
    )
    line_state = model.initial_alfven_state_from_li_potential(density)
    return model, density, line_state


def _seed_boundary_gradient(state: ModalAlfvenState) -> ModalAlfvenState:
    plus = state.w_plus_v.copy()
    minus = state.w_minus_v.copy()
    # A small, real transverse electromagnetic perturbation.  It has an
    # associated k_perp and therefore a genuine FAC, unlike a bare 1-D b(s).
    plus[0, 0] += 2.0e-3
    plus[0, 1] -= 0.8e-3
    return ModalAlfvenState(plus, minus)


def _passive_single_mode_components(
    *, line_cells: int, solver: str = "modal_woodbury"
):
    config = _config().with_updates(
        gravity_m_s2=0.0,
        collision_scale_km=1.0e30,
    )
    closure = DimensionalFACMimeticClosure(
        config, background_field_t=B0_T
    )
    legacy = _model_impl.ESFModel.__new__(_model_impl.ESFModel)
    legacy.config = config
    legacy.potential_solver = closure
    basis = ModalBasis(
        nx=config.nx,
        ny=config.ny,
        dx_m=config.dx_m,
        dy_m=config.dy_m,
        modes=(TransverseMode(1, 0, "cos"),),
    )
    line = ModalAlfvenLine(
        length_m=8.0e6,
        cells=line_cells,
        alfven_speed_m_s=8.0e5,
        mode_count=1,
        spatial_order=2,
    )
    feedback = PhysicalFeedbackConfig(
        gain=1.0,
        solver=solver,
        fac_source_model="symmetric_flux_tube",
        parallel_averaging_length_m=2.0 * line.length_m,
        hemisphere_count=2,
        relative_tolerance=2.0e-10,
    )
    model = DimensionalBidirectionalModel(
        legacy, closure, basis, line, feedback
    )
    density = np.full((config.ny, config.nx), 5.0e11)
    return model, density


def _passive_multimode_components(
    *,
    line_cells: int,
    solver: str = "modal_woodbury",
    parallel_profile_m: np.ndarray | None = None,
):
    """Return a heterogeneous, frozen, gravity-free passive load problem."""

    config = _config().with_updates(gravity_m_s2=0.0)
    collision_profile_m = (
        None
        if parallel_profile_m is None
        else parallel_profile_m
        * np.geomspace(1.05, 2.1, config.ny)
    )
    closure = DimensionalFACMimeticClosure(
        config,
        background_field_t=B0_T,
        gravity_column_weight_m=parallel_profile_m,
        collision_column_weight_m=collision_profile_m,
    )
    legacy = _model_impl.ESFModel.__new__(_model_impl.ESFModel)
    legacy.config = config
    legacy.potential_solver = closure
    basis = ModalBasis.low_wavenumber(
        nx=config.nx,
        ny=config.ny,
        dx_m=config.dx_m,
        dy_m=config.dy_m,
        max_x_wavenumber=2,
        max_y_wavenumber=2,
        include_sine=True,
    )
    line = ModalAlfvenLine(
        length_m=8.0e6,
        cells=line_cells,
        alfven_speed_m_s=8.0e5,
        mode_count=basis.mode_count,
        spatial_order=2,
    )
    feedback = PhysicalFeedbackConfig(
        gain=1.0,
        solver=solver,
        fac_source_model="symmetric_flux_tube",
        parallel_averaging_length_m=(
            2.0 * line.length_m if parallel_profile_m is None else None
        ),
        gravity_column_weight_m=parallel_profile_m,
        collision_column_weight_m=collision_profile_m,
        hemisphere_count=2,
        relative_tolerance=2.0e-10,
    )
    model = DimensionalBidirectionalModel(
        legacy, closure, basis, line, feedback
    )

    y = np.arange(config.ny, dtype=np.float64)[:, None]
    x = np.arange(config.nx, dtype=np.float64)[None, :]
    x_wave = np.cos(2.0 * np.pi * x / config.nx)
    y_wave = np.cos(np.pi * (y + 0.5) / config.ny)
    density = 5.0e11 * (
        1.0 + 0.13 * x_wave + 0.08 * y_wave + 0.04 * x_wave * y_wave
    )
    assert np.min(density) > 0.0
    return model, density


def test_gain_zero_rhs_and_ssprk3_are_exact_legacy_density_branch():
    model, density, line_state = _components(PhysicalFeedbackConfig(gain=0.0))
    state = DimensionalEMState(density.copy(), line_state)

    expected_rhs, expected_diag = model.legacy_li_model.evaluate_rhs(density)
    actual_rhs, actual_diag = model.evaluate_rhs(0.0, state)
    assert np.array_equal(actual_rhs.density_m3, expected_rhs)
    assert np.count_nonzero(actual_rhs.alfven.w_plus_v) == 0
    assert np.count_nonzero(actual_rhs.alfven.w_minus_v) == 0
    assert actual_diag.electrostatic == expected_diag
    assert actual_diag.fac_closure is None

    expected_step, expected_step_diag = model.legacy_li_model.ssprk3_step(
        density, model.config.dt_s
    )
    actual_step, actual_step_diag = model.ssprk3_step(
        0.0, state, model.config.dt_s
    )
    assert np.array_equal(actual_step.density_m3, expected_step)
    assert np.array_equal(actual_step.alfven.w_plus_v, line_state.w_plus_v)
    assert np.array_equal(actual_step.alfven.w_minus_v, line_state.w_minus_v)
    assert actual_step_diag.electrostatic == expected_step_diag


@pytest.mark.parametrize("line_spatial_order", [1, 2])
def test_initial_line_profile_matches_both_voltage_boundaries_without_jump(
    line_spatial_order,
):
    model, density, line_state = _components(
        PhysicalFeedbackConfig(gain=1.0),
        line_spatial_order=line_spatial_order,
    )
    potential, _ = model.closure.solve(density, forcing=None)
    left_voltage = model.basis.project(B0_T * potential)
    line_voltage = model.line.potential_coefficients_v(line_state)
    fraction = (model.line.s_m / model.line.length_m)[:, None]
    expected = (
        (1.0 - fraction) * left_voltage[None, :]
        + fraction * model.right_boundary_voltage_v[None, :]
    )
    np.testing.assert_allclose(line_voltage, expected, rtol=0.0, atol=2.0e-17)
    assert np.count_nonzero(model.line.apar_coefficients_t_m(line_state)) == 0

    # With the first-order characteristic flux used by this fixture, a
    # cell-centred linear profile has a spatially uniform initial derivative;
    # no endpoint cell contains a hidden discontinuity pulse.
    derivative = model.line.rhs(
        line_state,
        left_voltage_v=left_voltage,
        right_voltage_v=model.right_boundary_voltage_v,
    )
    np.testing.assert_allclose(
        derivative.w_plus_v,
        np.broadcast_to(derivative.w_plus_v[0], derivative.w_plus_v.shape),
        rtol=2.0e-13,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        derivative.w_minus_v,
        np.broadcast_to(derivative.w_minus_v[0], derivative.w_minus_v.shape),
        rtol=2.0e-13,
        atol=2.0e-15,
    )
    if line_spatial_order == 2:
        response = model.line.left_boundary_response(
            line_state,
            model.basis,
            boundary_voltage_coefficients_v=left_voltage,
        )
        np.testing.assert_allclose(
            response.current_gradient.d_j_parallel_ds_a_m3,
            0.0,
            rtol=0.0,
            atol=1.0e-24,
        )


@pytest.mark.parametrize(
    ("boundary_model", "conductance_factor", "expected_reflection"),
    [
        ("matched", None, 0.0),
        ("conductance", 0.0, 1.0),
        ("conductance", 3.0, -0.5),
    ],
)
def test_passive_remote_boundary_is_used_by_line_and_has_steady_initial_state(
    boundary_model, conductance_factor, expected_reflection
):
    original, density, _ = _components(PhysicalFeedbackConfig(gain=1.0))
    conductance = (
        None
        if conductance_factor is None
        else conductance_factor * original.line.alfven_conductance_s
    )
    model = DimensionalBidirectionalModel(
        original.legacy_li_model,
        original.closure,
        original.basis,
        original.line,
        original.feedback,
        right_boundary_model=boundary_model,
        right_boundary_conductance_s=conductance,
    )
    assert model.right_boundary_reflection_coefficient == pytest.approx(
        expected_reflection
    )

    line_state = model.initial_alfven_state_from_li_potential(density)
    potential, _ = model.closure.solve(density, forcing=None)
    voltage = model.basis.project(B0_T * potential)
    derivative = model.line.rhs(
        line_state,
        left_voltage_v=voltage,
        **model._remote_boundary_kwargs(),
    )
    np.testing.assert_allclose(derivative.w_plus_v, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(derivative.w_minus_v, 0.0, atol=2.0e-14)
    _, diagnostics = model.evaluate_rhs(
        0.0, DimensionalEMState(density, line_state)
    )
    assert diagnostics.remote_boundary_reflection_coefficient == pytest.approx(
        expected_reflection
    )
    if abs(expected_reflection) < 1.0:
        assert diagnostics.remote_boundary_power_modal_w > 0.0
    else:
        assert diagnostics.remote_boundary_power_modal_w == pytest.approx(
            0.0, abs=1.0e-20
        )


def test_remote_boundary_configuration_rejects_active_or_ambiguous_inputs():
    original, _, _ = _components(PhysicalFeedbackConfig(gain=1.0))
    common = (
        original.legacy_li_model,
        original.closure,
        original.basis,
        original.line,
        original.feedback,
    )
    with pytest.raises(ValueError, match="require zero"):
        DimensionalBidirectionalModel(
            *common,
            right_boundary_voltage_v=1.0,
            right_boundary_model="matched",
        )
    with pytest.raises(ValueError, match="required"):
        DimensionalBidirectionalModel(
            *common,
            right_boundary_model="conductance",
        )
    with pytest.raises(ValueError, match="non-negative"):
        DimensionalBidirectionalModel(
            *common,
            right_boundary_model="conductance",
            right_boundary_conductance_s=-1.0,
        )


def test_positive_gain_ramp_keeps_es_limit_but_not_line_propagation_at_start():
    model, density, line_state = _components(
        PhysicalFeedbackConfig(gain=0.8, ramp_time_s=10.0)
    )
    state = DimensionalEMState(density.copy(), line_state)
    tendency, diagnostics = model.evaluate_rhs(0.0, state)
    legacy_rhs, _ = model.legacy_li_model.evaluate_rhs(density)
    assert np.array_equal(tendency.density_m3, legacy_rhs)
    expected_line = model.line.rhs(
        line_state,
        left_voltage_v=np.zeros(model.basis.mode_count),
        right_voltage_v=model.right_boundary_voltage_v,
    )
    np.testing.assert_allclose(
        tendency.alfven.w_plus_v,
        expected_line.w_plus_v,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tendency.alfven.w_minus_v,
        expected_line.w_minus_v,
        rtol=0.0,
        atol=0.0,
    )
    assert diagnostics.effective_gain == 0.0
    assert diagnostics.activation_fraction == 0.0
    assert diagnostics.alfven_cfl_rate_s_inv == pytest.approx(
        model.line.alfven_speed_m_s / model.line.spacing_m
    )
    assert diagnostics.boundary_power_modal_w == 0.0

    _, middle = model.evaluate_rhs(5.0, state)
    assert middle.effective_gain == pytest.approx(0.4)
    assert middle.activation_fraction == pytest.approx(0.5)
    assert middle.alfven_cfl_rate_s_inv == pytest.approx(
        diagnostics.alfven_cfl_rate_s_inv
    )


def test_port_ramp_has_c_squared_load_and_equal_line_li_power():
    """The ramp scales effort and flow once each without slowing the line."""

    model, density = _passive_single_mode_components(line_cells=16)
    model.feedback = PhysicalFeedbackConfig(
        gain=1.0,
        ramp_time_s=10.0,
        solver="modal_woodbury",
        fac_source_model="symmetric_flux_tube",
        parallel_averaging_length_m=2.0 * model.line.length_m,
        hemisphere_count=2,
        relative_tolerance=2.0e-10,
    )
    incident_v = 120.0
    shape = (model.line.cells, model.line.mode_count)
    line_state = ModalAlfvenState(
        np.full(shape, incident_v),
        np.full(shape, incident_v),
    )
    coupling = model.feedback.effective_gain(5.0)
    assert coupling == pytest.approx(0.5)
    potential, response, _, residual = model._coupled_potential(
        density, line_state, coupling
    )
    li_voltage = model.basis.project(B0_T * potential)[0]

    collision = model.closure.collision_frequency_s[0]
    scale = density[0, 0] * collision
    eigenvalue = model.basis.eigenvalues_m2[0]
    modal_admittance = (
        model.feedback.hemisphere_count
        * eigenvalue
        / (
            model.feedback.parallel_averaging_length_m
            * MU0
            * model.line.alfven_speed_m_s
        )
    )
    load_coefficient = (
        B0_T**2
        * modal_admittance
        / (model.closure.ion_mass_kg * scale)
    )
    expected_li_ratio = (
        coupling * load_coefficient
        / (eigenvalue + coupling**2 * load_coefficient)
    )
    assert li_voltage / incident_v == pytest.approx(
        expected_li_ratio, rel=3.0e-12, abs=2.0e-14
    )
    line_voltage = coupling * li_voltage
    assert response.incoming_w_plus_v[0] == pytest.approx(
        2.0 * line_voltage - incident_v,
        rel=3.0e-12,
        abs=2.0e-14,
    )
    assert residual <= model.feedback.relative_tolerance

    source = coupling * model._fac_source_from_response(response)
    li_power = model.feedback.parallel_averaging_length_m * (
        model.basis.field_inner_product(
            model.basis.lift(np.array([li_voltage])), source
        )
    )
    current = model.basis.parallel_current_coefficients_a_m2(
        response.boundary_apar_coefficients_t_m
    )
    line_power = model.feedback.hemisphere_count * (
        model.basis.modal_inner_product(
            np.array([line_voltage]), current
        )
    )
    assert li_power == pytest.approx(line_power, rel=3.0e-13, abs=1.0e-15)

    state = DimensionalEMState(density, line_state)
    _, diagnostics = model.evaluate_rhs(5.0, state)
    assert diagnostics.boundary_power_modal_w == pytest.approx(
        line_power, rel=3.0e-12, abs=1.0e-15
    )
    assert diagnostics.alfven_cfl_rate_s_inv == pytest.approx(
        model.line.alfven_speed_m_s / model.line.spacing_m
    )


def test_symmetric_flux_tube_is_a_passive_single_mode_load_and_grid_invariant():
    """The flux-tube closure has ``K + Y``, a bounded reflection coefficient.

    A spatially uniform incident ``w_minus=W`` makes the one-mode algebra
    analytic.  The result must not depend on the numerical spacing of the
    half-line because the FAC volume source is a flux-tube average, not a
    one-sided half-cell derivative.
    """

    incident_v = 120.0
    voltage_ratios = []
    for line_cells in (8, 16, 32):
        model, density = _passive_single_mode_components(
            line_cells=line_cells
        )
        shape = (model.line.cells, model.line.mode_count)
        line_state = ModalAlfvenState(
            np.full(shape, incident_v),
            np.full(shape, incident_v),
        )
        potential, response, _, residual = model._coupled_potential(
            density, line_state, 1.0
        )
        voltage_v = model.basis.project(B0_T * potential)[0]

        collision = model.closure.collision_frequency_s[0]
        scale = density[0, 0] * collision
        eigenvalue = model.basis.eigenvalues_m2[0]
        modal_admittance = (
            model.feedback.hemisphere_count
            * eigenvalue
            / (
                model.feedback.parallel_averaging_length_m
                * MU0
                * model.line.alfven_speed_m_s
            )
        )
        load_coefficient = (
            B0_T**2
            * modal_admittance
            / (model.closure.ion_mass_kg * scale)
        )
        expected_ratio = load_coefficient / (
            eigenvalue + load_coefficient
        )
        voltage_ratios.append(voltage_v / incident_v)
        assert voltage_v / incident_v == pytest.approx(
            expected_ratio, rel=3.0e-12, abs=2.0e-14
        )

        reflection = response.incoming_w_plus_v[0] / incident_v
        assert reflection == pytest.approx(
            2.0 * expected_ratio - 1.0,
            rel=3.0e-12,
            abs=2.0e-14,
        )
        assert abs(reflection) <= 1.0
        current = model.basis.parallel_current_coefficients_a_m2(
            response.boundary_apar_coefficients_t_m
        )[0]
        assert voltage_v * current <= 0.0
        assert residual <= model.feedback.relative_tolerance

        state = DimensionalEMState(density, line_state)
        _, diagnostics = model.evaluate_rhs(0.0, state)
        assert diagnostics.boundary_power_modal_w < 0.0
        assert diagnostics.boundary_power_grid_w < 0.0

    np.testing.assert_allclose(
        voltage_ratios,
        np.full(3, voltage_ratios[0]),
        rtol=2.0e-13,
        atol=2.0e-15,
    )


@pytest.mark.parametrize("coupling", [0.35, 1.0])
def test_symmetric_flux_tube_full_multimode_scattering_is_passive(coupling):
    """The complete retained-mode reflection matrix is energy contractive.

    The heterogeneous positive density mixes modes, so checking the diagonal
    reflection factors alone would miss an indefinite off-diagonal coupling.
    In characteristic variables the Alfvén energy metric is proportional to
    ``diag(k_perp**2)``.  Passivity therefore requires every singular value of
    ``sqrt(D) R inv(sqrt(D))`` to be no greater than one.
    """

    model, density = _passive_multimode_components(line_cells=8)
    mode_count = model.basis.mode_count
    reflection = np.empty((mode_count, mode_count), dtype=np.float64)
    for incident_mode in range(mode_count):
        incident = np.zeros(
            (model.line.cells, mode_count), dtype=np.float64
        )
        incident[:, incident_mode] = 1.0
        line_state = ModalAlfvenState(np.zeros_like(incident), incident)
        _, response, _, residual = model._coupled_potential(
            density, line_state, coupling
        )
        reflection[:, incident_mode] = response.incoming_w_plus_v
        assert residual <= model.feedback.relative_tolerance

    metric_sqrt = np.sqrt(model.basis.eigenvalues_m2)
    energy_reflection = (
        metric_sqrt[:, None]
        * reflection
        / metric_sqrt[None, :]
    )
    singular_values = np.linalg.svd(
        energy_reflection, compute_uv=False
    )
    absorption = (
        np.eye(mode_count) - energy_reflection.T @ energy_reflection
    )

    # This is genuinely a full-matrix test rather than fourteen independent
    # copies of the existing one-mode reflection test.
    assert np.linalg.norm(
        energy_reflection - np.diag(np.diag(energy_reflection))
    ) > 1.0e-5
    np.testing.assert_allclose(
        energy_reflection,
        energy_reflection.T,
        rtol=0.0,
        atol=2.0e-11,
    )
    assert np.max(singular_values) < 1.0
    assert np.min(np.linalg.eigvalsh(absorption)) > 0.0


def test_incident_multimode_wave_packet_closes_the_line_energy_budget():
    """An incident packet is absorbed at the passive port without energy gain.

    For the semi-discrete upwind line, the exact continuum balance is
    ``E(t)-E(0)=integral(P_port dt)-E_numerical``.  The positive remainder is
    finite-volume dissipation and must decrease rapidly on line refinement.
    Density is deliberately frozen and gravity is zero, isolating the
    corrected symmetric source/port from nonlinear ES evolution.
    """

    relative_budget_defects = []
    for line_cells in (64, 128):
        model, density = _passive_multimode_components(
            line_cells=line_cells
        )
        line = model.line
        basis = model.basis
        mode_index = np.arange(basis.mode_count, dtype=np.float64)
        amplitudes = 80.0 * (-1.0) ** mode_index / (1.0 + 0.25 * mode_index)
        envelope = np.exp(
            -(
                (line.s_m - 0.35 * line.length_m)
                / (0.12 * line.length_m)
            )
            ** 2
        )
        initial_minus = envelope[:, None] * amplitudes[None, :]
        state = ModalAlfvenState(
            np.zeros_like(initial_minus), initial_minus.copy()
        )

        energy_coefficient = (
            model.feedback.hemisphere_count
            * basis.domain_area_m2
            * line.spacing_m
            / (4.0 * MU0 * line.alfven_speed_m_s**2)
        )

        def line_energy(line_state):
            characteristic_energy = (
                line_state.w_plus_v**2 + line_state.w_minus_v**2
            ) * basis.eigenvalues_m2[None, :]
            return float(energy_coefficient * np.sum(characteristic_energy))

        def loaded_rhs_and_power(line_state):
            potential, response, _, residual = model._coupled_potential(
                density, line_state, 1.0
            )
            assert residual <= model.feedback.relative_tolerance
            voltage = B0_T * basis.project(potential)
            derivative = line.rhs(
                line_state,
                left_voltage_v=voltage,
                right_voltage_v=0.0,
            )
            current = basis.parallel_current_coefficients_a_m2(
                response.boundary_apar_coefficients_t_m
            )
            power = model.feedback.hemisphere_count * (
                basis.modal_inner_product(voltage, current)
            )
            return derivative, power

        def add_scaled(line_state, derivative, scale):
            return ModalAlfvenState(
                line_state.w_plus_v + scale * derivative.w_plus_v,
                line_state.w_minus_v + scale * derivative.w_minus_v,
            )

        def loaded_ssprk3_step(line_state, dt_s):
            first_rhs, first_power = loaded_rhs_and_power(line_state)
            first = add_scaled(line_state, first_rhs, dt_s)
            second_rhs, second_power = loaded_rhs_and_power(first)
            second_euler = add_scaled(first, second_rhs, dt_s)
            second = ModalAlfvenState(
                0.75 * line_state.w_plus_v
                + 0.25 * second_euler.w_plus_v,
                0.75 * line_state.w_minus_v
                + 0.25 * second_euler.w_minus_v,
            )
            third_rhs, third_power = loaded_rhs_and_power(second)
            third_euler = add_scaled(second, third_rhs, dt_s)
            updated = ModalAlfvenState(
                line_state.w_plus_v / 3.0
                + 2.0 * third_euler.w_plus_v / 3.0,
                line_state.w_minus_v / 3.0
                + 2.0 * third_euler.w_minus_v / 3.0,
            )
            integrated_power = dt_s * (
                first_power / 6.0
                + second_power / 6.0
                + 2.0 * third_power / 3.0
            )
            return updated, integrated_power, max(
                first_power, second_power, third_power
            )

        initial_energy = line_energy(state)
        integrated_port_power = 0.0
        maximum_port_power = -np.inf
        duration_s = 7.5
        step_count = int(
            np.ceil(duration_s / (0.4 * line.maximum_stable_dt_s))
        )
        dt_s = duration_s / step_count
        for _ in range(step_count):
            state, power_increment, stage_maximum = loaded_ssprk3_step(
                state, dt_s
            )
            integrated_port_power += power_increment
            maximum_port_power = max(maximum_port_power, stage_maximum)

        final_energy = line_energy(state)
        actual_energy_loss = initial_energy - final_energy
        port_absorption = -integrated_port_power
        numerical_dissipation = actual_energy_loss - port_absorption
        relative_budget_defects.append(
            numerical_dissipation / initial_energy
        )

        assert maximum_port_power <= 1.0e-8
        assert port_absorption > 0.0
        assert actual_energy_loss > 0.0
        assert numerical_dissipation >= -1.0e-10 * initial_energy
        assert (
            np.linalg.norm(state.w_minus_v)
            / np.linalg.norm(initial_minus)
            < 1.0e-4
        )

    assert relative_budget_defects[1] < 2.5e-3
    assert relative_budget_defects[1] < relative_budget_defects[0] / 4.0


def test_symmetric_flux_tube_configuration_requires_explicit_geometry():
    with pytest.raises(ValueError, match="parallel_averaging_length_m"):
        PhysicalFeedbackConfig(fac_source_model="symmetric_flux_tube")
    with pytest.raises(ValueError, match="either a scalar"):
        PhysicalFeedbackConfig(
            fac_source_model="symmetric_flux_tube",
            parallel_averaging_length_m=1.0,
            gravity_column_weight_m=np.ones(3),
            collision_column_weight_m=np.ones(3),
        )
    with pytest.raises(ValueError, match="supplied together"):
        PhysicalFeedbackConfig(
            fac_source_model="symmetric_flux_tube",
            gravity_column_weight_m=np.ones(3),
        )
    with pytest.raises(ValueError, match="one value"):
        original, _, _ = _components(PhysicalFeedbackConfig(gain=1.0))
        profile = np.ones(2)
        DimensionalBidirectionalModel(
            original.legacy_li_model,
            original.closure,
            original.basis,
            original.line,
            PhysicalFeedbackConfig(
                fac_source_model="symmetric_flux_tube",
                gravity_column_weight_m=profile,
                collision_column_weight_m=profile,
            ),
        )
    with pytest.raises(ValueError, match="positive finite"):
        PhysicalFeedbackConfig(
            fac_source_model="symmetric_flux_tube",
            gravity_column_weight_m=np.array([1.0, -1.0]),
            collision_column_weight_m=np.ones(2),
        )
    with pytest.raises(ValueError, match="only valid"):
        PhysicalFeedbackConfig(
            fac_source_model="boundary_gradient",
            gravity_column_weight_m=np.ones(3),
            collision_column_weight_m=np.ones(3),
        )
    with pytest.raises(ValueError, match="hemisphere_count"):
        PhysicalFeedbackConfig(
            fac_source_model="symmetric_flux_tube",
            parallel_averaging_length_m=1.0,
            hemisphere_count=0,
        )


@pytest.mark.parametrize("solver", ["direct_krylov", "fixed_point"])
def test_symmetric_flux_tube_source_is_identical_in_all_coupling_solvers(solver):
    reference, density = _passive_single_mode_components(
        line_cells=8, solver="modal_woodbury"
    )
    candidate, _ = _passive_single_mode_components(
        line_cells=8, solver=solver
    )
    shape = (reference.line.cells, reference.line.mode_count)
    line_state = ModalAlfvenState(
        np.full(shape, 80.0),
        np.full(shape, 120.0),
    )
    coupling = 0.37
    expected, expected_response, _, _ = reference._coupled_potential(
        density, line_state, coupling
    )
    actual, actual_response, _, _ = candidate._coupled_potential(
        density, line_state, coupling
    )
    np.testing.assert_allclose(actual, expected, rtol=3.0e-10, atol=2.0e-8)
    np.testing.assert_allclose(
        actual_response.current_gradient.d_j_parallel_ds_a_m3,
        expected_response.current_gradient.d_j_parallel_ds_a_m3,
        rtol=3.0e-10,
        atol=2.0e-18,
    )
    np.testing.assert_allclose(
        candidate._fac_source_from_response(actual_response),
        reference._fac_source_from_response(expected_response),
        rtol=3.0e-10,
        atol=2.0e-18,
    )


@pytest.mark.parametrize("solver", ["direct_krylov", "fixed_point"])
def test_variable_column_source_is_identical_in_all_coupling_solvers(solver):
    profile_m = np.geomspace(1.5e7, 6.0e7, _config().ny)
    reference, density = _passive_multimode_components(
        line_cells=8,
        solver="modal_woodbury",
        parallel_profile_m=profile_m,
    )
    candidate, _ = _passive_multimode_components(
        line_cells=8,
        solver=solver,
        parallel_profile_m=profile_m,
    )
    rng = np.random.default_rng(20260820)
    shape = (reference.line.cells, reference.line.mode_count)
    line_state = ModalAlfvenState(
        30.0 * rng.standard_normal(shape),
        40.0 * rng.standard_normal(shape),
    )
    coupling = 0.43
    expected, expected_response, _, expected_residual = (
        reference._coupled_potential(density, line_state, coupling)
    )
    actual, actual_response, _, actual_residual = candidate._coupled_potential(
        density, line_state, coupling
    )
    assert expected_residual <= 5.0 * reference.feedback.relative_tolerance
    assert actual_residual <= 5.0 * candidate.feedback.relative_tolerance
    np.testing.assert_allclose(actual, expected, rtol=5.0e-9, atol=4.0e-8)
    np.testing.assert_allclose(
        candidate._fac_source_from_response(actual_response),
        reference._fac_source_from_response(expected_response),
        rtol=5.0e-9,
        atol=3.0e-18,
    )


def test_variable_column_uses_surface_current_li_power_pairing():
    profile_m = np.geomspace(8.0e6, 5.0e7, _config().ny)
    model, density = _passive_multimode_components(
        line_cells=8, parallel_profile_m=profile_m
    )
    rng = np.random.default_rng(31)
    shape = (model.line.cells, model.line.mode_count)
    line_state = ModalAlfvenState(
        25.0 * rng.standard_normal(shape),
        35.0 * rng.standard_normal(shape),
    )
    state = DimensionalEMState(density, line_state)
    _, diagnostics = model.evaluate_rhs(0.0, state)
    assert diagnostics.relative_power_pairing_defect < 3.0e-13
    assert abs(diagnostics.net_d_j_parallel_ds_a_m) < 1.0e-8
    assert diagnostics.fac_closure.relative_net_current_gradient < 3.0e-13
    assert diagnostics.fac_forcing_representation == "surface_current"
    assert diagnostics.fac_forcing_units == "A m^-2"

    potential, response, _, _ = model._coupled_potential(density, line_state, 1.0)
    voltage = model.basis.project(B0_T * potential)
    source = model._fac_source_from_response(response)
    li_column_power = model.basis.field_inner_product(
        model.basis.lift(voltage), source,
    )
    current = model.basis.parallel_current_coefficients_a_m2(
        response.boundary_apar_coefficients_t_m
    )
    line_power = model.feedback.hemisphere_count * model.basis.modal_inner_product(
        voltage, current
    )
    assert li_column_power == pytest.approx(line_power, rel=3.0e-13, abs=1.0e-15)


def test_variable_column_full_multimode_scattering_is_energy_contractive():
    profile_m = np.geomspace(5.0e6, 8.0e7, _config().ny)
    model, density = _passive_multimode_components(
        line_cells=8, parallel_profile_m=profile_m
    )
    count = model.basis.mode_count
    reflection = np.empty((count, count), dtype=np.float64)
    for incident_mode in range(count):
        incident = np.zeros((model.line.cells, count), dtype=np.float64)
        incident[:, incident_mode] = 1.0
        line_state = ModalAlfvenState(np.zeros_like(incident), incident)
        _, response, _, residual = model._coupled_potential(
            density, line_state, 1.0
        )
        assert residual <= 5.0 * model.feedback.relative_tolerance
        reflection[:, incident_mode] = response.incoming_w_plus_v

    metric = np.sqrt(model.basis.eigenvalues_m2)
    energy_reflection = metric[:, None] * reflection / metric[None, :]
    singular_values = np.linalg.svd(energy_reflection, compute_uv=False)
    assert np.max(singular_values) < 1.0


def test_variable_column_density_rhs_conserves_flux_tube_particles():
    profile_m = np.geomspace(4.0e6, 5.0e7, _config().ny)
    model, density = _passive_multimode_components(
        line_cells=8, parallel_profile_m=profile_m
    )
    # Use an x-independent state and only kx=0 incident modes.  Then v_y=0
    # at every altitude boundary, so the finite domain has no particle flux.
    density = np.broadcast_to(
        np.mean(density, axis=1, keepdims=True), density.shape
    ).copy()
    shape = (model.line.cells, model.line.mode_count)
    minus = np.zeros(shape)
    for index, mode in enumerate(model.basis.modes):
        if mode.x_wavenumber == 0:
            minus[:, index] = 18.0
    line_state = ModalAlfvenState(np.zeros(shape), minus)
    tendency, diagnostics = model.evaluate_rhs(
        0.0, DimensionalEMState(density, line_state)
    )
    weighted_tendency = (
        model.gravity_column_weight_field_m * tendency.density_m3
    )
    absolute = np.sum(np.abs(weighted_tendency), dtype=np.float64)
    relative_balance = abs(float(np.sum(weighted_tendency))) / max(
        float(absolute), np.finfo(float).tiny
    )
    assert relative_balance < 3.0e-14
    assert diagnostics.density_conservation_measure == (
        "gravity_column_weight_times_density"
    )


def test_variable_column_geometry_is_continuous_through_zero_port_gain():
    profile_m = np.geomspace(8.0e6, 4.0e7, _config().ny)
    model, density = _passive_multimode_components(
        line_cells=8, parallel_profile_m=profile_m
    )
    zeros = np.zeros((model.line.cells, model.line.mode_count))
    line_state = ModalAlfvenState(zeros.copy(), zeros.copy())
    zero, _, _, zero_residual = model._coupled_potential(
        density, line_state, 0.0
    )
    small, _, _, small_residual = model._coupled_potential(
        density, line_state, 1.0e-7
    )
    assert zero_residual == 0.0
    assert small_residual <= 5.0 * model.feedback.relative_tolerance
    relative = np.linalg.norm(small - zero) / max(np.linalg.norm(zero), 1.0)
    assert relative < 2.0e-12


def test_column_es_control_uses_geometry_without_fac_feedback():
    profile_m = np.geomspace(8.0e6, 4.0e7, _config().ny)
    model, density = _passive_multimode_components(
        line_cells=8, parallel_profile_m=profile_m
    )
    actual, diagnostics = model.evaluate_column_es_rhs(density)
    potential = model._unforced_coupled_geometry_potential(density)
    velocity_x, velocity_y = model.legacy_li_model.velocity_from_potential(
        potential
    )
    expected = model._density_advection_rhs(density, velocity_x, velocity_y)
    assert np.array_equal(actual, expected)
    assert diagnostics.cfl_rate >= 0.0
    assert model.closure.last_fac_diagnostics.fac_to_gravity_ratio == 0.0

    scalar, _ = _passive_multimode_components(line_cells=8)
    with pytest.raises(ValueError, match="column-integrated"):
        scalar.evaluate_column_es_rhs(density)


def test_full_gain_uses_dimensional_fac_and_closes_boundary_and_power():
    feedback = PhysicalFeedbackConfig(
        gain=1.0,
        under_relaxation=0.4,
        max_iterations=160,
        relative_tolerance=2.0e-10,
    )
    model, density, line_state = _components(feedback)
    seeded = _seed_boundary_gradient(line_state)
    state = DimensionalEMState(density.copy(), seeded)
    tendency, diagnostics = model.evaluate_rhs(1.0, state)

    assert diagnostics.effective_gain == 1.0
    assert diagnostics.activation_fraction == 1.0
    assert diagnostics.iterations > 0
    assert diagnostics.relative_coupling_residual <= feedback.relative_tolerance
    assert diagnostics.fac_closure is not None
    assert diagnostics.fac_closure.relative_net_current_gradient < 2.0e-13
    assert diagnostics.maximum_abs_d_j_parallel_ds_a_m3 > 0.0
    assert abs(diagnostics.net_d_j_parallel_ds_a_m) < 1.0e-12
    assert diagnostics.boundary_characteristic_defect_v < 1.0e-15
    assert diagnostics.relative_power_pairing_defect < 2.0e-14
    assert np.all(np.isfinite(tendency.density_m3))
    assert np.all(np.isfinite(tendency.alfven.w_plus_v))
    assert np.all(np.isfinite(tendency.alfven.w_minus_v))

    legacy_rhs, _ = model.legacy_li_model.evaluate_rhs(density)
    assert not np.array_equal(tendency.density_m3, legacy_rhs)
    assert np.linalg.norm(tendency.density_m3 - legacy_rhs) > 0.0


def test_short_full_gain_nonlinear_density_trajectory_differs_from_es():
    coupled, density, line_state = _components(
        PhysicalFeedbackConfig(
            gain=1.0,
            under_relaxation=0.4,
            max_iterations=160,
            relative_tolerance=5.0e-10,
        )
    )
    baseline, _, _ = _components(PhysicalFeedbackConfig(gain=0.0))
    seeded = _seed_boundary_gradient(line_state)
    coupled_state = DimensionalEMState(density.copy(), seeded)
    baseline_state = DimensionalEMState(density.copy(), seeded)
    initial_hash_bytes = density.tobytes()
    assert baseline_state.density_m3.tobytes() == initial_hash_bytes
    assert coupled_state.density_m3.tobytes() == initial_hash_bytes

    dt = 0.02
    time = 0.0
    for _ in range(5):
        baseline_state, _ = baseline.ssprk3_step(time, baseline_state, dt)
        coupled_state, diagnostics = coupled.ssprk3_step(
            time, coupled_state, dt
        )
        time += dt
    assert np.min(baseline_state.density_m3) > 0.0
    assert np.min(coupled_state.density_m3) > 0.0
    difference = float(
        np.linalg.norm(coupled_state.density_m3 - baseline_state.density_m3)
    )
    assert difference > 0.0
    assert diagnostics.relative_coupling_residual <= 5.0e-10
    assert density_mode_amplitude(coupled_state.density_m3) != pytest.approx(
        density_mode_amplitude(baseline_state.density_m3), rel=0.0, abs=0.0
    )


def test_direct_low_rank_solve_matches_physical_fixed_point_with_fewer_iterations():
    direct, density, line_state = _components(
        PhysicalFeedbackConfig(
            gain=1.0,
            relative_tolerance=2.0e-10,
            solver="direct_krylov",
        )
    )
    fixed, _, _ = _components(
        PhysicalFeedbackConfig(
            gain=1.0,
            under_relaxation=0.4,
            max_iterations=180,
            relative_tolerance=2.0e-10,
            solver="fixed_point",
        )
    )
    seeded = _seed_boundary_gradient(line_state)
    state = DimensionalEMState(density.copy(), seeded)
    direct_rhs, direct_diag = direct.evaluate_rhs(1.0, state)
    fixed_rhs, fixed_diag = fixed.evaluate_rhs(1.0, state)
    np.testing.assert_allclose(
        direct_rhs.density_m3,
        fixed_rhs.density_m3,
        rtol=2.0e-9,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        direct_rhs.alfven.w_plus_v,
        fixed_rhs.alfven.w_plus_v,
        rtol=2.0e-9,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        direct_rhs.alfven.w_minus_v,
        fixed_rhs.alfven.w_minus_v,
        rtol=2.0e-9,
        atol=1.0e-10,
    )
    assert direct_diag.iterations < fixed_diag.iterations


def test_modal_woodbury_matches_direct_krylov_solution():
    woodbury, density, line_state = _components(
        PhysicalFeedbackConfig(
            gain=1.0,
            relative_tolerance=2.0e-10,
            solver="modal_woodbury",
        )
    )
    direct, _, _ = _components(
        PhysicalFeedbackConfig(
            gain=1.0,
            relative_tolerance=2.0e-10,
            solver="direct_krylov",
        )
    )
    state = DimensionalEMState(
        density.copy(), _seed_boundary_gradient(line_state)
    )
    woodbury_rhs, woodbury_diag = woodbury.evaluate_rhs(1.0, state)
    direct_rhs, direct_diag = direct.evaluate_rhs(1.0, state)
    np.testing.assert_allclose(
        woodbury_rhs.density_m3,
        direct_rhs.density_m3,
        rtol=3.0e-10,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        woodbury_rhs.alfven.w_plus_v,
        direct_rhs.alfven.w_plus_v,
        rtol=3.0e-10,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        woodbury_rhs.alfven.w_minus_v,
        direct_rhs.alfven.w_minus_v,
        rtol=3.0e-10,
        atol=2.0e-10,
    )
    assert woodbury_diag.iterations == 1
    assert direct_diag.iterations > woodbury_diag.iterations


def test_nonconverged_physical_feedback_rejects_without_mutating_state():
    model, density, line_state = _components(
        PhysicalFeedbackConfig(
            gain=1.0,
            under_relaxation=0.05,
            max_iterations=1,
            relative_tolerance=1.0e-16,
            solver="fixed_point",
        )
    )
    state = DimensionalEMState(density.copy(), _seed_boundary_gradient(line_state))
    before_density = state.density_m3.copy()
    before_plus = state.alfven.w_plus_v.copy()
    before_minus = state.alfven.w_minus_v.copy()
    with pytest.raises(PhysicalCouplingConvergenceError):
        model.evaluate_rhs(1.0, state)
    assert np.array_equal(state.density_m3, before_density)
    assert np.array_equal(state.alfven.w_plus_v, before_plus)
    assert np.array_equal(state.alfven.w_minus_v, before_minus)


def test_feedback_startup_ramp_is_c1_and_reaches_full_gain():
    config = PhysicalFeedbackConfig(gain=0.8, ramp_time_s=10.0)
    assert config.effective_gain(0.0) == 0.0
    assert config.effective_gain(5.0) == pytest.approx(0.4)
    assert config.effective_gain(10.0) == pytest.approx(0.8)
    assert config.effective_gain(100.0) == pytest.approx(0.8)
