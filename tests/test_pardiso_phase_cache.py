from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from esf_sim.point_centered_potential import LogPointCenteredPotentialSolver


class FakePardiso:
    def __init__(self) -> None:
        self.phase = None
        self.phases: list[int] = []

    def set_phase(self, phase: int) -> None:
        self.phase = int(phase)

    def _solve_dense(self, matrix, rhs):
        self.phases.append(self.phase)
        return np.linalg.solve(matrix.toarray(), np.asarray(rhs))

    def solve(self, matrix, rhs):
        return self._solve_dense(matrix, rhs)

    def _call_pardiso(self, matrix, rhs):
        return self._solve_dense(matrix, rhs)


def _solver_with_fake_backend():
    solver = LogPointCenteredPotentialSolver.__new__(
        LogPointCenteredPotentialSolver
    )
    solver.solver = FakePardiso()
    solver._pattern_indptr = None
    solver._pattern_indices = None
    solver._factorized_matrix = None
    solver._factorized_data = None
    return solver


def test_same_matrix_reuses_numeric_factorization_for_new_right_hand_sides():
    solver = _solver_with_fake_backend()
    matrix = csr_matrix(np.array([[3.0, -1.0], [-1.0, 2.0]]))
    first, rebuilt = solver._solve(matrix, np.array([1.0, 0.0]))
    second, rebuilt_again = solver._solve(matrix, np.array([0.0, 1.0]))
    np.testing.assert_allclose(first, np.linalg.solve(matrix.toarray(), [1.0, 0.0]))
    np.testing.assert_allclose(second, np.linalg.solve(matrix.toarray(), [0.0, 1.0]))
    assert rebuilt
    assert not rebuilt_again
    assert solver.solver.phases == [13, 33]


def test_new_values_refactor_once_then_return_to_solve_only_phase():
    solver = _solver_with_fake_backend()
    first = csr_matrix(np.array([[3.0, -1.0], [-1.0, 2.0]]))
    second = csr_matrix(np.array([[4.0, -1.0], [-1.0, 2.5]]))
    solver._solve(first, np.array([1.0, 0.0]))
    solver._solve(second, np.array([1.0, 0.0]))
    solver._solve(second, np.array([0.0, 1.0]))
    second.data[0] += 0.25
    solver._solve(second, np.array([1.0, 1.0]))
    assert solver.solver.phases == [13, 23, 33, 23]
