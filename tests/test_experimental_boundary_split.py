import numpy as np

from esf_sim.experimental_boundary_split import (
    build_arithmetic_dirichlet_matrix,
    zero_normal_current_boundary_flux,
)


def test_arithmetic_dirichlet_matrix_is_symmetric_positive_definite() -> None:
    matrix, scale = build_arithmetic_dirichlet_matrix(
        np.ones((5, 4)), dx=2.0, dy=3.0
    )

    assert scale == 1.0
    assert matrix.shape == (12, 12)
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T)
    assert np.min(np.linalg.eigvalsh(matrix.toarray())) > 0.0
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().reshape(3, 4)
    np.testing.assert_allclose(row_sums[0], 1.0 / 9.0)
    np.testing.assert_allclose(row_sums[1], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(row_sums[2], 1.0 / 9.0)


def test_equal_potential_ghosts_are_zero_normal_current_boundary() -> None:
    rng = np.random.default_rng(8)
    coefficient = 0.1 + rng.random((9, 7))
    potential = rng.standard_normal((9, 7))

    bottom, top = zero_normal_current_boundary_flux(
        coefficient, potential, dy=1000.0
    )

    np.testing.assert_array_equal(bottom, np.zeros(7))
    np.testing.assert_array_equal(top, np.zeros(7))
