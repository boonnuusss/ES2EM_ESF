import numpy as np

from esf_sim.figure3_comparison import CASES, _field_agreement


def test_figure3_comparison_case_contract():
    assert [case.alpha for case in CASES] == [0.2, 0.4, 0.6, 0.8]
    assert [case.time_s for case in CASES] == [3150.0, 3150.0, 3050.0, 2860.0]
    assert [case.recovered for case in CASES] == [True, False, False, False]
    assert CASES[0].expected_operator == "exponential_conservative"
    assert all(
        case.expected_operator == "point_centered_equation_2_21_log_coefficients"
        for case in CASES[1:]
    )


def test_field_agreement_is_exact_for_identical_arrays():
    field = np.arange(24, dtype=float).reshape(4, 6)
    agreement = _field_agreement(field, field)
    assert np.isclose(agreement["spatial_correlation"], 1.0)
    assert agreement["rmse_m3"] == 0.0
    assert agreement["normalized_rmse_by_author_range"] == 0.0
    assert agreement["relative_l2_error"] == 0.0
    assert agreement["mean_bias_m3"] == 0.0
    assert agreement["maximum_absolute_difference_m3"] == 0.0
