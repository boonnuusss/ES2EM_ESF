import numpy as np

from esf_sim.arithmetic_mean_ab import ArithmeticMeanESFModel, arm_width_profile
from esf_sim.config import SimulationConfig
from esf_sim._model_impl import ESFModel
from esf_sim.potential import build_elliptic_matrix


def test_default_harmonic_operator_is_unchanged_and_arithmetic_is_isolated():
    coefficient = np.array(
        [[1.0, 2.0, 4.0, 8.0], [1.5, 3.0, 6.0, 12.0], [2.0, 4.0, 8.0, 16.0]]
    )
    default, default_scale = build_elliptic_matrix(coefficient, 2.0, 3.0)
    harmonic, harmonic_scale = build_elliptic_matrix(
        coefficient, 2.0, 3.0, face_average="harmonic"
    )
    arithmetic, arithmetic_scale = build_elliptic_matrix(
        coefficient, 2.0, 3.0, face_average="arithmetic"
    )
    np.testing.assert_allclose(default.toarray(), harmonic.toarray())
    assert default_scale == harmonic_scale == arithmetic_scale
    assert not np.allclose(harmonic.toarray(), arithmetic.toarray())
    np.testing.assert_allclose(arithmetic.toarray(), arithmetic.toarray().T)


def test_constant_coefficient_gives_identical_face_operators():
    coefficient = np.ones((5, 7))
    harmonic, _ = build_elliptic_matrix(
        coefficient, 1.0, 1.0, face_average="harmonic"
    )
    arithmetic, _ = build_elliptic_matrix(
        coefficient, 1.0, 1.0, face_average="arithmetic"
    )
    np.testing.assert_allclose(harmonic.toarray(), arithmetic.toarray())


def test_ab_models_differ_only_in_potential_face_average():
    config = SimulationConfig(nx=9, ny=9, x_max_km=10.0, y_max_km=258.0)
    baseline = ESFModel(config)
    arithmetic = ArithmeticMeanESFModel(config)
    assert baseline.potential_solver.face_average == "harmonic"
    assert arithmetic.potential_solver.face_average == "arithmetic"
    assert type(baseline).velocity_from_potential is type(arithmetic).velocity_from_potential


def test_arm_width_profile_recovers_two_symmetric_troughs():
    x = np.linspace(-125.0, 125.0, 201)
    y = np.array([450.0])
    density = np.ones((1, x.size))
    density[0, (x >= -60.0) & (x <= -40.0)] = 0.2
    density[0, (x >= 40.0) & (x <= 60.0)] = 0.2
    profile = arm_width_profile(density, x, y)
    assert 20.0 < profile["mean_km"][0] < 22.0
