import numpy as np

from esf_sim.experimental_time import variable_step_predictor


def test_variable_step_predictor_is_exact_for_quadratic_history():
    times = [1.0, 0.7, 0.2]
    states = [np.array([time**2]) for time in times]
    target = 1.1
    predicted = variable_step_predictor(states, times, target)
    assert np.allclose(predicted, target**2)


def test_uniform_history_recovers_published_coefficients():
    states = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    predicted = variable_step_predictor(states, [2.0, 1.0, 0.0], 2.5)
    expected = 15.0 / 8.0 * states[0] - 5.0 / 4.0 * states[1] + 3.0 / 8.0 * states[2]
    assert np.allclose(predicted, expected)
