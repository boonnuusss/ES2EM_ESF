from __future__ import annotations

import numpy as np
import pytest

from esf_sim.controlled_em import (
    BoundaryFixedPointSolver,
    ChannelMap,
    CouplingConvergenceError,
    FeedbackConfig,
    area_weighted_l2,
    area_weighted_mean,
    condition_fac_source,
    dispatch_feedback,
    project_area_weighted_zero_mean,
)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("gain", -1.0, ValueError),
        ("gain", 1.01, ValueError),
        ("gain", np.inf, ValueError),
        ("ramp_time_s", -1.0, ValueError),
        ("under_relaxation", 0.0, ValueError),
        ("under_relaxation", 1.01, ValueError),
        ("max_iterations", 0, ValueError),
        ("max_iterations", 3.5, TypeError),
        ("relative_tolerance", 0.0, ValueError),
        ("max_source_ratio", 0.0, ValueError),
    ],
)
def test_feedback_config_rejects_invalid_controls(field, value, error):
    values = {
        "gain": 0.2,
        "ramp_time_s": 10.0,
        "under_relaxation": 0.7,
        "max_iterations": 100,
        "relative_tolerance": 1.0e-9,
        "max_source_ratio": 0.2,
    }
    values[field] = value
    with pytest.raises(error):
        FeedbackConfig(**values)


def test_feedback_ramp_is_clipped_and_gain_is_linear():
    config = FeedbackConfig(gain=0.2, ramp_time_s=10.0)
    assert config.effective_gain(0.0) == 0.0
    assert config.effective_gain(5.0) == pytest.approx(0.1)
    assert config.effective_gain(20.0) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        config.effective_gain(-1.0)


def test_fac_projection_has_exact_discrete_zero_net_current_and_no_mutation():
    fac = np.array([1.0, 2.0, 8.0, -3.0])
    area = np.array([1.0, 3.0, 2.0, 4.0])
    fac_before = fac.copy()
    area_before = area.copy()

    projected = project_area_weighted_zero_mean(fac, area)

    assert area_weighted_mean(projected, area) == pytest.approx(0.0, abs=2e-16)
    assert np.sum(area * projected) == pytest.approx(0.0, abs=2e-15)
    np.testing.assert_array_equal(fac, fac_before)
    np.testing.assert_array_equal(area, area_before)


def test_conditioned_fac_is_zero_mean_and_source_ratio_limited():
    fac = np.array([[5.0, -1.0], [3.0, 9.0]])
    area = np.array([[1.0, 2.0], [3.0, 4.0]])
    reference = np.ones_like(fac)
    config = FeedbackConfig(gain=1.0, max_source_ratio=0.125)

    result = condition_fac_source(
        fac, reference, area, config, time_s=2.0
    )

    assert area_weighted_mean(result.source, area) == pytest.approx(0.0, abs=1e-15)
    ratio = area_weighted_l2(result.source, area) / area_weighted_l2(
        reference, area
    )
    assert ratio == pytest.approx(config.max_source_ratio)
    assert result.diagnostics.limited
    assert result.diagnostics.applied_scale < 1.0
    assert result.diagnostics.final_source_ratio == pytest.approx(ratio)
    assert result.diagnostics.input_weighted_mean == pytest.approx(
        area_weighted_mean(fac, area)
    )
    assert result.diagnostics.projected_weighted_mean == pytest.approx(
        0.0, abs=1e-15
    )


def test_small_gain_feedback_scales_linearly_before_limiter():
    fac = np.array([2.0, -1.0, 4.0, 0.5])
    area = np.array([1.0, 2.0, 1.0, 3.0])
    reference = np.full(4, 100.0)
    low = condition_fac_source(
        fac,
        reference,
        area,
        FeedbackConfig(gain=1.0e-4, max_source_ratio=1.0),
        time_s=1.0,
    )
    high = condition_fac_source(
        fac,
        reference,
        area,
        FeedbackConfig(gain=2.0e-4, max_source_ratio=1.0),
        time_s=1.0,
    )

    assert not low.diagnostics.limited
    assert not high.diagnostics.limited
    np.testing.assert_allclose(high.source, 2.0 * low.source, rtol=0.0, atol=0.0)


def test_zero_reference_norm_safely_forces_zero_feedback():
    result = condition_fac_source(
        np.array([1.0, -1.0]),
        np.zeros(2),
        np.ones(2),
        FeedbackConfig(gain=1.0, max_source_ratio=0.5),
        time_s=1.0,
    )
    np.testing.assert_array_equal(result.source, np.zeros(2))
    assert result.diagnostics.limited
    assert np.isinf(result.diagnostics.unlimited_source_ratio)
    assert result.diagnostics.final_source_ratio == 0.0


def test_gain_zero_is_literal_unforced_branch_and_never_calls_forced():
    calls = {"unforced": 0, "forced": 0}

    def unforced():
        calls["unforced"] += 1
        return "electrostatic"

    def forced(_gain):
        calls["forced"] += 1
        raise AssertionError("forced callback must not be reached")

    value = dispatch_feedback(
        FeedbackConfig(gain=0.0),
        time_s=123.0,
        unforced=unforced,
        forced=forced,
    )

    assert value == "electrostatic"
    assert calls == {"unforced": 1, "forced": 0}


def test_nonzero_gain_is_passed_to_forced_callback_after_ramp():
    received = []
    value = dispatch_feedback(
        FeedbackConfig(gain=0.4, ramp_time_s=20.0),
        time_s=5.0,
        unforced=lambda: pytest.fail("unforced branch must not be reached"),
        forced=lambda gain: received.append(gain) or "coupled",
    )
    assert value == "coupled"
    assert received == pytest.approx([0.1])


def test_boundary_fixed_point_matches_closed_form_without_mutating_inputs():
    phi0 = np.array([1.0, -2.0, 0.5])
    j_free = np.array([0.2, 0.3, -0.1])
    k = np.array([0.25, 0.1, -0.2])
    admittance = np.array([0.4, -0.25, 0.3])
    initial = np.array([8.0, 8.0, 8.0])
    originals = [array.copy() for array in (phi0, j_free, k, admittance, initial)]
    solver = BoundaryFixedPointSolver(
        FeedbackConfig(
            under_relaxation=0.7,
            max_iterations=200,
            relative_tolerance=1.0e-13,
        )
    )

    result = solver.solve(
        phi0, j_free, k, admittance, initial_potential=initial
    )
    expected_phi = (phi0 + k * j_free) / (1.0 - k * admittance)
    expected_j = j_free + admittance * expected_phi

    np.testing.assert_allclose(result.potential, expected_phi, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.fac, expected_j, rtol=2e-13, atol=2e-13)
    assert result.relative_residual < 1.0e-13
    for actual, original in zip((phi0, j_free, k, admittance, initial), originals):
        np.testing.assert_array_equal(actual, original)


def test_boundary_nonconvergence_raises_and_does_not_modify_any_input():
    phi0 = np.array([1.0, 2.0])
    j_free = np.array([0.1, -0.1])
    k = np.array([2.0, 2.0])
    admittance = np.ones(2)
    initial = np.zeros(2)
    arrays = (phi0, j_free, k, admittance, initial)
    originals = tuple(array.copy() for array in arrays)
    solver = BoundaryFixedPointSolver(
        FeedbackConfig(
            under_relaxation=1.0,
            max_iterations=5,
            relative_tolerance=1.0e-14,
        )
    )

    with pytest.raises(CouplingConvergenceError) as error:
        solver.solve(
            phi0, j_free, k, admittance, initial_potential=initial
        )

    assert error.value.iterations == 5
    assert np.isfinite(error.value.relative_residual)
    for actual, original in zip(arrays, originals):
        np.testing.assert_array_equal(actual, original)


def test_channel_map_satisfies_weighted_power_adjoint_identity():
    mapping = ChannelMap(
        matrix=np.array(
            [
                [1.0, 0.0, 0.5],
                [0.0, -2.0, 0.3],
                [0.7, 0.2, -1.0],
                [-0.1, 0.8, 0.0],
            ]
        ),
        cell_area=np.array([1.0, 3.0, 2.0, 4.0]),
        channel_weight=np.array([2.0, 0.5, 1.5]),
    )
    channels = np.array([0.2, -1.1, 0.7])
    grid_effort = np.array([3.0, -0.5, 1.2, 0.1])

    grid_power = mapping.grid_pairing(mapping.to_grid(channels), grid_effort)
    channel_power = mapping.channel_pairing(
        channels, mapping.adjoint_to_channels(grid_effort)
    )

    assert grid_power == pytest.approx(channel_power, rel=2e-15, abs=2e-15)
