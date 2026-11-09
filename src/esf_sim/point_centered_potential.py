"""Production point-centered five-point potential solver for equation (2.21)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .config import SimulationConfig
from .potential import PotentialDiagnostics


_DLL_HANDLES: list[object] = []


def _load_pardiso():
    workspace = Path(__file__).resolve().parents[2]
    binary_directory = workspace / ".deps" / "Library" / "bin"
    runtime = binary_directory / "mkl_rt.3.dll"
    if runtime.exists():
        os.environ.setdefault("PYPARDISO_MKL_RT", str(runtime))
        os.environ["PATH"] = str(binary_directory) + os.pathsep + os.environ.get(
            "PATH", ""
        )
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(str(binary_directory)))
    try:
        from pypardiso import PyPardisoSolver
    except ImportError as error:
        raise RuntimeError(
            "pypardiso is required by the point-centered best-baseline solver"
        ) from error
    return PyPardisoSolver


def build_log_point_centered_matrix(
    coefficient: np.ndarray,
    dx: float,
    dy: float,
) -> csr_matrix:
    """Discretize ``-laplacian(psi) - grad(log(a)).grad(psi)`` on five points.

    This is the point-centered form of equation (2.21), with the exact
    continuum identity ``grad(a)/a = grad(log(a))`` evaluated before centered
    differencing. Horizontal rows are periodic and the first/last vertical
    points are homogeneous Dirichlet boundaries excluded from the unknowns.
    """

    raw = np.asarray(coefficient, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 3:
        raise ValueError("coefficient must be two-dimensional with at least 3 rows")
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("coefficient must be positive and finite")
    log_a = np.log(raw)
    interior = log_a[1:-1]
    nyi, nx = interior.shape
    identifiers = np.arange(nyi * nx, dtype=np.int64).reshape(nyi, nx)
    drift_x = (
        np.roll(interior, -1, axis=1) - np.roll(interior, 1, axis=1)
    ) / (2.0 * dx)
    drift_y = (log_a[2:] - log_a[:-2]) / (2.0 * dy)

    rows = [identifiers.ravel(), identifiers.ravel(), identifiers.ravel()]
    columns = [
        identifiers.ravel(),
        np.roll(identifiers, -1, axis=1).ravel(),
        np.roll(identifiers, 1, axis=1).ravel(),
    ]
    data = [
        np.full(identifiers.size, 2.0 / dx**2 + 2.0 / dy**2),
        (-1.0 / dx**2 - drift_x / (2.0 * dx)).ravel(),
        (-1.0 / dx**2 + drift_x / (2.0 * dx)).ravel(),
    ]
    if nyi > 1:
        rows.extend([identifiers[:-1].ravel(), identifiers[1:].ravel()])
        columns.extend([identifiers[1:].ravel(), identifiers[:-1].ravel()])
        data.extend(
            [
                (-1.0 / dy**2 - drift_y[:-1] / (2.0 * dy)).ravel(),
                (-1.0 / dy**2 + drift_y[1:] / (2.0 * dy)).ravel(),
            ]
        )
    matrix = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(identifiers.size, identifiers.size),
    )
    matrix.sort_indices()
    return matrix


class LogPointCenteredPotentialSolver:
    """PARDISO solve of the stable point-centered equation (2.21) stencil."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.collision_frequency = config.collision_nu0_s * np.exp(
            -config.y_coordinates_m / config.collision_frequency_scale_height_m
        )
        self.solver = _load_pardiso()(mtype=11)
        self._pattern_indptr: np.ndarray | None = None
        self._pattern_indices: np.ndarray | None = None
        self._factorized_matrix: csr_matrix | None = None
        self._factorized_data: np.ndarray | None = None

    def _solve(self, matrix: csr_matrix, rhs: np.ndarray) -> tuple[np.ndarray, bool]:
        right = np.asarray(rhs, dtype=np.float64)
        if right.ndim == 2 and not right.flags.f_contiguous:
            # ``PyPardisoSolver.solve`` performs this conversion internally,
            # but direct phase-23/33 calls do not.  MKL interprets multiple
            # right-hand sides in column-major order.
            right = np.asfortranarray(right)
        same_pattern = (
            self._pattern_indptr is not None
            and self._pattern_indices is not None
            and np.array_equal(matrix.indptr, self._pattern_indptr)
            and np.array_equal(matrix.indices, self._pattern_indices)
        )
        if not same_pattern:
            self._pattern_indptr = matrix.indptr.copy()
            self._pattern_indices = matrix.indices.copy()
            self.solver.set_phase(13)
            solution = self.solver.solve(matrix, right)
            self._factorized_matrix = matrix
            self._factorized_data = matrix.data.copy()
            return np.asarray(solution, dtype=np.float64), True

        same_numeric_factorization = (
            matrix is self._factorized_matrix
            and self._factorized_data is not None
            and np.array_equal(matrix.data, self._factorized_data)
        )
        if same_numeric_factorization:
            # GMRES may apply the exact Li preconditioner dozens of times to
            # different right-hand sides while the matrix is unchanged.  Phase
            # 33 reuses both symbolic analysis and numeric factors; phase 23
            # would needlessly refactor the same 2-D operator every iteration.
            self.solver.set_phase(33)
            solution = self.solver._call_pardiso(matrix, right)
            return np.asarray(solution, dtype=np.float64), False

        self.solver.set_phase(23)
        solution = self.solver._call_pardiso(matrix, right)
        self._factorized_matrix = matrix
        self._factorized_data = matrix.data.copy()
        return np.asarray(solution, dtype=np.float64), False

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        positive_n = np.maximum(n, self.config.density_floor_m3)
        coefficient = positive_n * self.collision_frequency[:, None]
        matrix = build_log_point_centered_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        log_n = np.log(positive_n)
        log_density_dx = (
            np.roll(log_n, -1, axis=1) - np.roll(log_n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        rhs = (
            self.config.gravity_m_s2
            * log_density_dx[1:-1]
            / self.collision_frequency[1:-1, None]
        ).ravel()
        solution, rebuilt = self._solve(matrix, rhs)
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            self._pattern_indptr = None
            self._pattern_indices = None
            self.solver.set_phase(13)
            solution = np.asarray(self.solver.solve(matrix, rhs), dtype=np.float64)
            residual = matrix @ solution - rhs
            relative_residual = float(np.linalg.norm(residual) / denominator)
            rebuilt = True
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            raise RuntimeError(
                f"PARDISO residual {relative_residual:.3e} exceeds 1e-8"
            )
        potential = np.zeros_like(n)
        potential[1:-1] = solution.reshape(n.shape[0] - 2, n.shape[1])
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=rebuilt,
            used_pyamg=False,
        )



def build_equal_extrapolated_log_point_centered_matrix(
    coefficient: np.ndarray,
    dx: float,
    dy: float,
    *,
    impose_gauge: bool = True,
) -> csr_matrix:
    """Discretize equation (2.21) with equal-value altitude ghost cells.

    All vertical grid rows remain active unknowns. The ghost contribution is
    folded into each boundary-row diagonal. The ungauged operator annihilates
    constants; one identity row fixes the arbitrary additive potential.
    """

    raw = np.asarray(coefficient, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) < 2:
        raise ValueError("coefficient must be two-dimensional with at least 2 rows")
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("coefficient must be positive and finite")

    log_a = np.log(raw)
    ny, nx = log_a.shape
    identifiers = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    drift_x = (
        np.roll(log_a, -1, axis=1) - np.roll(log_a, 1, axis=1)
    ) / (2.0 * dx)
    padded_y = np.pad(log_a, ((1, 1), (0, 0)), mode="edge")
    drift_y = (padded_y[2:] - padded_y[:-2]) / (2.0 * dy)

    diagonal = np.full((ny, nx), 2.0 / dx**2 + 2.0 / dy**2)
    diagonal[0] = 2.0 / dx**2 + 1.0 / dy**2 + drift_y[0] / (2.0 * dy)
    diagonal[-1] = 2.0 / dx**2 + 1.0 / dy**2 - drift_y[-1] / (2.0 * dy)
    rows = [identifiers.ravel(), identifiers.ravel(), identifiers.ravel()]
    columns = [
        identifiers.ravel(),
        np.roll(identifiers, -1, axis=1).ravel(),
        np.roll(identifiers, 1, axis=1).ravel(),
    ]
    data = [
        diagonal.ravel(),
        (-1.0 / dx**2 - drift_x / (2.0 * dx)).ravel(),
        (-1.0 / dx**2 + drift_x / (2.0 * dx)).ravel(),
    ]
    rows.extend([identifiers[:-1].ravel(), identifiers[1:].ravel()])
    columns.extend([identifiers[1:].ravel(), identifiers[:-1].ravel()])
    data.extend(
        [
            (-1.0 / dy**2 - drift_y[:-1] / (2.0 * dy)).ravel(),
            (-1.0 / dy**2 + drift_y[1:] / (2.0 * dy)).ravel(),
        ]
    )
    matrix = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(ny * nx, ny * nx),
    )
    matrix.sort_indices()
    if impose_gauge:
        start, stop = matrix.indptr[0], matrix.indptr[1]
        matrix.data[start:stop] = 0.0
        local_diagonal = np.flatnonzero(matrix.indices[start:stop] == 0)
        if local_diagonal.size != 1:
            raise RuntimeError("failed to locate the potential gauge diagonal")
        matrix.data[start + int(local_diagonal[0])] = 1.0
    return matrix


class EqualExtrapolatedLogPointCenteredPotentialSolver(
    LogPointCenteredPotentialSolver
):
    """PARDISO solve using equal-value altitude ghosts for the potential."""

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        positive_n = np.maximum(n, self.config.density_floor_m3)
        coefficient = positive_n * self.collision_frequency[:, None]
        matrix = build_equal_extrapolated_log_point_centered_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        log_n = np.log(positive_n)
        log_density_dx = (
            np.roll(log_n, -1, axis=1) - np.roll(log_n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        rhs = (
            self.config.gravity_m_s2
            * log_density_dx
            / self.collision_frequency[:, None]
        ).ravel()
        rhs[0] = 0.0
        solution, rebuilt = self._solve(matrix, rhs)
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            self._pattern_indptr = None
            self._pattern_indices = None
            self.solver.set_phase(13)
            solution = np.asarray(self.solver.solve(matrix, rhs), dtype=np.float64)
            residual = matrix @ solution - rhs
            relative_residual = float(np.linalg.norm(residual) / denominator)
            rebuilt = True
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            raise RuntimeError(
                f"PARDISO residual {relative_residual:.3e} exceeds 1e-8"
            )
        potential = solution.reshape(n.shape)
        potential -= float(np.mean(potential))
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=rebuilt,
            used_pyamg=False,
        )


__all__ = [
    "EqualExtrapolatedLogPointCenteredPotentialSolver",
    "LogPointCenteredPotentialSolver",
    "build_equal_extrapolated_log_point_centered_matrix",
    "build_log_point_centered_matrix",
]
