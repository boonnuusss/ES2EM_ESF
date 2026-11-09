import numpy as np

from esf_sim.high_order_time import variable_ab3_weights


def test_constant_step_ab3_weights():
    weights = variable_ab3_weights([2.0, 1.0, 0.0], 3.0)
    np.testing.assert_allclose(weights, [23/12, -16/12, 5/12], atol=2e-15)


def test_variable_ab3_integrates_quadratic_rhs_exactly():
    times = [2.0, 1.3, -0.2]
    target = 2.45
    weights = variable_ab3_weights(times, target)
    rhs = np.asarray([t*t-2*t+3 for t in times])
    numerical = float(weights@rhs)
    antiderivative = lambda t: t**3/3-t*t+3*t
    exact = antiderivative(target)-antiderivative(times[0])
    np.testing.assert_allclose(numerical, exact, atol=2e-15)


def test_ab3_scalar_ode_has_third_order_global_convergence():
    def integrate(h):
        # Exact starting values isolate the multistep truncation error.
        times = [2*h, h, 0.0]
        states = [np.exp(t) for t in times]
        while times[0] < 1-1e-14:
            step = min(h, 1-times[0])
            weights = variable_ab3_weights(times, times[0]+step)
            states.insert(0, states[0]+float(weights@np.asarray(states[:3])))
            times.insert(0, times[0]+step)
            del states[3:]; del times[3:]
        return abs(states[0]-np.e)
    errors = [integrate(h) for h in (0.05, 0.025, 0.0125)]
    assert errors[0]/errors[1] > 7.0
    assert errors[1]/errors[2] > 7.0
