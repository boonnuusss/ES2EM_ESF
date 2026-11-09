import numpy as np

from esf_sim.experimental_combination import (
    ACTIVE_HORIZONTAL_CELLS,
    author_storage_with_four_ghosts,
    build_arithmetic_neumann_matrix,
    expand_author_horizontal,
)


def test_inferred_active_core_reproduces_sciencedb_ghost_indices() -> None:
    active = np.arange(ACTIVE_HORIZONTAL_CELLS, dtype=float)[None, :]
    display = expand_author_horizontal(active)
    stored = author_storage_with_four_ghosts(active)

    assert display.shape[-1] == 201
    assert stored.shape[-1] == 209
    np.testing.assert_array_equal(stored[..., :4], stored[..., 198:202])
    np.testing.assert_array_equal(stored[..., 205:209], stored[..., 7:11])


def test_arithmetic_neumann_matrix_has_gauge_and_constant_null_solution() -> None:
    coefficient = np.ones((7, 9))
    matrix, scale = build_arithmetic_neumann_matrix(coefficient, 2.0, 3.0)
    assert scale == 1.0
    assert matrix.shape == (63, 63)
    row = matrix.getrow(0).toarray().ravel()
    assert row[0] == 1.0
    assert np.count_nonzero(row) == 1

