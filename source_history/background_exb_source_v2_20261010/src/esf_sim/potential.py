from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg

try:
    import pyamg  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only in minimal environments
    pyamg = None


FloatArray = NDArray[np.float64]


def harmonic_mean(left: FloatArray, right: FloatArray) -> FloatArray:
    denominator = left + right
    result = np.zeros_like(denominator)
    np.divide(2.0 * left * right, denominator, out=result, where=denominator > 0.0)
    return result


def arithmetic_mean(left: FloatArray, right: FloatArray) -> FloatArray:
    """Return the symmetric arithmetic face value used by the isolated A/B test."""

    return 0.5 * (left + right)


def build_elliptic_matrix(
    coefficient: FloatArray,
    dx: float,
    dy: float,
    *,
    face_average: str = "harmonic",
) -> tuple[csr_matrix, float]:
    """Build ``-div(a grad)`` on a periodic-x, Dirichlet-y grid.

    The first and last y rows are fixed to zero and are not included among the
    unknowns.  Scaling by ``max(a)`` improves conditioning without changing the
    solution when the right-hand side is scaled by the returned factor.
    """

    if face_average == "harmonic":
        average = harmonic_mean
    elif face_average == "arithmetic":
        average = arithmetic_mean
    else:
        raise ValueError("face_average must be 'harmonic' or 'arithmetic'")

    a_raw = np.asarray(coefficient, dtype=np.float64)
    if a_raw.ndim != 2 or a_raw.shape[0] < 3:
        raise ValueError("coefficient must be a two-dimensional array with at least 3 y rows")
    scale = float(np.max(a_raw))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("elliptic coefficient must contain positive finite values")
    a = a_raw / scale

    interior = a[1:-1]
    nyi, nx = interior.shape
    rows_grid = np.arange(nyi * nx, dtype=np.int64).reshape(nyi, nx)

    east = average(interior, np.roll(interior, -1, axis=1))
    west = average(interior, np.roll(interior, 1, axis=1))
    north = average(interior, a[2:])
    south = average(interior, a[:-2])

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    diagonal = (east + west) * inv_dx2 + (north + south) * inv_dy2

    row_parts = [rows_grid.ravel()]
    col_parts = [rows_grid.ravel()]
    data_parts = [diagonal.ravel()]

    row_parts.extend([rows_grid.ravel(), rows_grid.ravel()])
    col_parts.extend(
        [np.roll(rows_grid, -1, axis=1).ravel(), np.roll(rows_grid, 1, axis=1).ravel()]
    )
    data_parts.extend([(-east * inv_dx2).ravel(), (-west * inv_dx2).ravel()])

    if nyi > 1:
        row_parts.append(rows_grid[:-1].ravel())
        col_parts.append(rows_grid[1:].ravel())
        data_parts.append((-north[:-1] * inv_dy2).ravel())

        row_parts.append(rows_grid[1:].ravel())
        col_parts.append(rows_grid[:-1].ravel())
        data_parts.append((-south[1:] * inv_dy2).ravel())

    matrix = csr_matrix(
        (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(nyi * nx, nyi * nx),
    )
    return matrix, scale


@dataclass
class PotentialDiagnostics:
    iterations: int
    relative_residual: float
    rebuilt_preconditioner: bool
    used_pyamg: bool


class PotentialSolveError(RuntimeError):
    pass


class PotentialSolver:
    """Warm-started CG solver for the polarization potential."""

    def __init__(
        self,
        *,
        dx: float,
        dy: float,
        gravity: float,
        collision_frequency: FloatArray,
        density_floor: float = 1.0,
        rtol: float = 1.0e-8,
        maxiter: int = 400,
        refresh_steps: int = 25,
        iteration_limit: int = 80,
        face_average: str = "harmonic",
    ) -> None:
        self.dx = dx
        self.dy = dy
        self.gravity = gravity
        self.collision_frequency = np.asarray(collision_frequency, dtype=np.float64)
        self.density_floor = density_floor
        self.rtol = rtol
        self.maxiter = maxiter
        self.refresh_steps = refresh_steps
        self.iteration_limit = iteration_limit
        if face_average not in {"harmonic", "arithmetic"}:
            raise ValueError("face_average must be 'harmonic' or 'arithmetic'")
        self.face_average = face_average

        self._preconditioner: LinearOperator | None = None
        self._preconditioner_age = refresh_steps
        self._last_solution: FloatArray | None = None
        self._warned_no_pyamg = False

    @property
    def has_pyamg(self) -> bool:
        return pyamg is not None

    def _make_preconditioner(self, matrix: csr_matrix) -> LinearOperator:
        if pyamg is not None:
            hierarchy = pyamg.smoothed_aggregation_solver(matrix, symmetry="symmetric")
            return hierarchy.aspreconditioner(cycle="V")
        if not self._warned_no_pyamg:
            warnings.warn(
                "PyAMG is unavailable; using a Jacobi preconditioner. Full-resolution runs will be slower.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_no_pyamg = True
        inverse_diagonal = 1.0 / matrix.diagonal()
        return LinearOperator(matrix.shape, matvec=lambda value: inverse_diagonal * value)

    def _cg_once(
        self, matrix: csr_matrix, rhs: FloatArray, *, force_rebuild: bool
    ) -> tuple[FloatArray, int, float, int, bool]:
        rebuilt = force_rebuild or self._preconditioner is None or self._preconditioner_age >= self.refresh_steps
        if rebuilt:
            self._preconditioner = self._make_preconditioner(matrix)
            self._preconditioner_age = 0

        x0 = self._last_solution
        if x0 is not None and x0.shape != rhs.shape:
            x0 = None
        iterations = 0

        def callback(_: FloatArray) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            matrix,
            rhs,
            x0=x0,
            M=self._preconditioner,
            rtol=self.rtol,
            atol=0.0,
            maxiter=self.maxiter,
            callback=callback,
        )
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        return solution, iterations, relative_residual, int(info), rebuilt

    def solve(self, density: FloatArray) -> tuple[FloatArray, PotentialDiagnostics]:
        n = np.asarray(density, dtype=np.float64)
        if n.ndim != 2:
            raise ValueError("density must be two-dimensional")
        if self.collision_frequency.shape not in {(n.shape[0],), n.shape}:
            raise ValueError("collision frequency must have shape (ny,) or density.shape")

        nu = self.collision_frequency
        if nu.ndim == 1:
            nu = nu[:, None]
        coefficient = np.maximum(n, self.density_floor) * nu
        matrix, scale = build_elliptic_matrix(
            coefficient,
            self.dx,
            self.dy,
            face_average=self.face_average,
        )

        density_dx = (np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)) / (2.0 * self.dx)
        rhs = (self.gravity * density_dx[1:-1] / scale).ravel()

        force_rebuild = self._preconditioner_age >= self.refresh_steps
        solution, iterations, relative_residual, info, rebuilt = self._cg_once(
            matrix, rhs, force_rebuild=force_rebuild
        )
        needs_retry = (
            info != 0
            or not np.isfinite(relative_residual)
            or relative_residual > 10.0 * self.rtol
            or iterations > self.iteration_limit
        )
        if needs_retry and not rebuilt:
            solution, iterations, relative_residual, info, rebuilt = self._cg_once(
                matrix, rhs, force_rebuild=True
            )

        if info != 0 or not np.isfinite(relative_residual) or relative_residual > 10.0 * self.rtol:
            raise PotentialSolveError(
                f"CG failed: info={info}, iterations={iterations}, relative residual={relative_residual:.3e}"
            )

        self._last_solution = solution
        self._preconditioner_age += 1
        potential = np.zeros_like(n)
        potential[1:-1] = solution.reshape(n.shape[0] - 2, n.shape[1])
        return potential, PotentialDiagnostics(
            iterations=iterations,
            relative_residual=relative_residual,
            rebuilt_preconditioner=rebuilt,
            used_pyamg=pyamg is not None,
        )

