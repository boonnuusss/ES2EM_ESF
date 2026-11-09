from esf_sim.validation import (
    elliptic_manufactured_solution_test,
    temporal_accuracy_test,
)


def test_elliptic_operator_is_second_order():
    report = elliptic_manufactured_solution_test()
    assert report["second_order_pass"]


def test_published_time_integrator_accuracy_is_measured():
    report = temporal_accuracy_test()
    assert report["accuracy_pass"]
    assert report["observed_orders"][-1] > 1.95
