"""Conservative high-resolution calibration of the ESF potential equation."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .config import SimulationConfig
from .point_centered_potential import LogPointCenteredPotentialSolver
from .potential import PotentialDiagnostics, harmonic_mean


def build_harmonic_equal_extrapolated_matrix(
    coefficient: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[csr_matrix, float]:
    """Build ``-div(a grad(psi))`` with periodic x and equal-value y ghosts."""

    raw = np.asarray(coefficient, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) < 2:
        raise ValueError("coefficient must be two-dimensional with at least 2 rows")
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("coefficient must be positive and finite")

    scale = float(np.max(raw))
    a = raw / scale
    ny, nx = a.shape
    identifiers = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    east = harmonic_mean(a, np.roll(a, -1, axis=1))
    west = harmonic_mean(a, np.roll(a, 1, axis=1))
    north = np.zeros_like(a)
    north[:-1] = harmonic_mean(a[:-1], a[1:])
    south = np.zeros_like(a)
    south[1:] = north[:-1]
    diagonal = (east + west) / dx**2 + (north + south) / dy**2

    rows = [identifiers.ravel(), identifiers.ravel(), identifiers.ravel()]
    columns = [
        identifiers.ravel(),
        np.roll(identifiers, -1, axis=1).ravel(),
        np.roll(identifiers, 1, axis=1).ravel(),
    ]
    data = [
        diagonal.ravel(),
        (-east / dx**2).ravel(),
        (-west / dx**2).ravel(),
    ]
    rows.extend([identifiers[:-1].ravel(), identifiers[1:].ravel()])
    columns.extend([identifiers[1:].ravel(), identifiers[:-1].ravel()])
    data.extend([(-north[:-1] / dy**2).ravel(), (-south[1:] / dy**2).ravel()])
    matrix = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(ny * nx, ny * nx),
    )
    matrix.sort_indices()

    # The equal-extrapolation operator has one constant null mode.
    start, stop = matrix.indptr[0], matrix.indptr[1]
    matrix.data[start:stop] = 0.0
    local_diagonal = np.flatnonzero(matrix.indices[start:stop] == 0)
    if local_diagonal.size != 1:
        raise RuntimeError("failed to locate the potential gauge diagonal")
    matrix.data[start + int(local_diagonal[0])] = 1.0
    return matrix, scale


class HarmonicMimeticLogPotentialSolver(LogPointCenteredPotentialSolver):
    """Conservative harmonic-face left operator with the calibrated log RHS."""

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        positive_n = np.maximum(n, self.config.density_floor_m3)
        coefficient = positive_n * self.collision_frequency[:, None]
        matrix, scale = build_harmonic_equal_extrapolated_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        log_n = np.log(positive_n)
        log_density_dx = (
            np.roll(log_n, -1, axis=1) - np.roll(log_n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        divided_rhs = (
            self.config.gravity_m_s2
            * log_density_dx
            / self.collision_frequency[:, None]
        )
        rhs = (coefficient / scale * divided_rhs).ravel()
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
    "HarmonicMimeticLogPotentialSolver",
    "build_harmonic_equal_extrapolated_matrix",
]
