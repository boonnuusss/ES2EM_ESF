import numpy as np

from esf_sim.experimental import STAGES, build_point_centered_matrix


def test_centered_operator_has_expected_shape_and_constant_null_action_inside():
    coefficient = np.ones((7, 8))
    matrix = build_point_centered_matrix(
        coefficient, 1.0, 1.0, boundary_y="dirichlet_zero"
    )
    assert matrix.shape == (40, 40)
    assert np.all(matrix.diagonal() > 0.0)


def test_stage_one_contract():
    stage = STAGES["centered_dirichlet"]
    assert stage.potential_operator == "point_centered_five_point"
    assert stage.potential_boundary_y == "dirichlet_zero"
