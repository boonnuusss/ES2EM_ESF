"""Monotone potential operators for strong-gradient ESF states.

The production point-centred stencil is an accurate calibration target while
its local cell Peclet number is below one.  At sharper interfaces, one of its
off-diagonal coefficients can become positive and the discrete elliptic
operator can acquire a non-physical near-zero mode.  This module contains two
isolated alternatives used to diagnose and remove that failure mode.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .config import SimulationConfig
from .point_centered_potential import LogPointCenteredPotentialSolver
from .potential import PotentialDiagnostics


def exponential_face_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the positive exponential-profile transmissibility mean.

    If the coefficient varies exponentially between two cell centres, this is
    the exact face conductance.  It is the Scharfetter-Gummel (exponentially
    fitted) counterpart of a centred face average and remains positive for an
    arbitrarily large coefficient jump.
    """

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("left and right face arrays must have equal shapes")
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("face coefficients must be positive")
    log_ratio = np.log(b) - np.log(a)
    result = np.empty_like(a)
    close = np.abs(log_ratio) < 1.0e-7
    # The geometric mean is the symmetric limiting value and avoids the 0/0
    # cancellation in the direct expression for nearly equal coefficients.
    result[close] = np.sqrt(a[close] * b[close])
    result[~close] = (
        a[~close]
        * b[~close]
        * log_ratio[~close]
        / (b[~close] - a[~close])
    )
    return result


def build_exponential_equal_extrapolated_matrix(
    coefficient: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[csr_matrix, float]:
    """Build ``-div(a grad(psi))`` with positive fitted face conductances."""

    raw = np.asarray(coefficient, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) < 2:
        raise ValueError("coefficient must be a two-dimensional positive array")
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("coefficient must be positive and finite")
    scale = float(np.max(raw))
    a = raw / scale
    ny, nx = a.shape
    identifiers = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)

    east = exponential_face_mean(a, np.roll(a, -1, axis=1))
    west = np.roll(east, 1, axis=1)
    north = np.zeros_like(a)
    north[:-1] = exponential_face_mean(a[:-1], a[1:])
    south = np.zeros_like(a)
    south[1:] = north[:-1]
    diagonal = (east + west) / dx**2 + (north + south) / dy**2

    rows = [
        identifiers.ravel(),
        identifiers.ravel(),
        identifiers.ravel(),
        identifiers[:-1].ravel(),
        identifiers[1:].ravel(),
    ]
    columns = [
        identifiers.ravel(),
        np.roll(identifiers, -1, axis=1).ravel(),
        np.roll(identifiers, 1, axis=1).ravel(),
        identifiers[1:].ravel(),
        identifiers[:-1].ravel(),
    ]
    data = [
        diagonal.ravel(),
        (-east / dx**2).ravel(),
        (-west / dx**2).ravel(),
        (-north[:-1] / dy**2).ravel(),
        (-south[1:] / dy**2).ravel(),
    ]
    matrix = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(ny * nx, ny * nx),
    )
    matrix.sort_indices()
    return matrix, scale


def point_centered_peclet_diagnostics(
    coefficient: np.ndarray,
    dx: float,
    dy: float,
) -> dict[str, float | int]:
    """Measure the monotonicity violation of the centred drift stencil."""

    log_a = np.log(np.asarray(coefficient, dtype=np.float64))
    drift_x = (
        np.roll(log_a, -1, axis=1) - np.roll(log_a, 1, axis=1)
    ) / (2.0 * dx)
    padded_y = np.pad(log_a, ((1, 1), (0, 0)), mode="edge")
    drift_y = (padded_y[2:] - padded_y[:-2]) / (2.0 * dy)
    peclet_x = np.abs(drift_x) * dx / 2.0
    peclet_y = np.abs(drift_y) * dy / 2.0
    return {
        "maximum_peclet_x": float(np.max(peclet_x)),
        "maximum_peclet_y": float(np.max(peclet_y)),
        "cells_peclet_x_above_one": int(np.count_nonzero(peclet_x > 1.0)),
        "cells_peclet_y_above_one": int(np.count_nonzero(peclet_y > 1.0)),
    }


class ExponentialConservativePotentialSolver(LogPointCenteredPotentialSolver):
    """Positive conservative solve of the same continuum potential equation."""

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        positive_n = np.maximum(n, self.config.density_floor_m3)
        coefficient = positive_n * self.collision_frequency[:, None]
        matrix, scale = build_exponential_equal_extrapolated_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )

        density_dx = (
            np.roll(positive_n, -1, axis=1)
            - np.roll(positive_n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        rhs = (self.config.gravity_m_s2 * density_dx / scale).ravel()

        # The conservative periodic RHS sums to roundoff zero.  Eliminate one
        # gauge degree of freedom instead of deleting an incompatible physical
        # equation; the omitted row follows from conservation.
        reduced_matrix = matrix[1:, 1:].tocsr()
        reduced_rhs = rhs[1:]
        reduced_solution, rebuilt = self._solve(reduced_matrix, reduced_rhs)
        solution = np.empty_like(rhs)
        solution[0] = 0.0
        solution[1:] = reduced_solution

        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.isfinite(relative_residual) or relative_residual > self.config.potential_rtol:
            self._pattern_indptr = None
            self._pattern_indices = None
            reduced_solution, rebuilt = self._solve(reduced_matrix, reduced_rhs)
            solution[1:] = reduced_solution
            residual = matrix @ solution - rhs
            relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.isfinite(relative_residual) or relative_residual > self.config.potential_rtol:
            raise RuntimeError(
                "exponential conservative PARDISO residual "
                f"{relative_residual:.3e} exceeds {self.config.potential_rtol:g}"
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
    "ExponentialConservativePotentialSolver",
    "build_exponential_equal_extrapolated_matrix",
    "exponential_face_mean",
    "point_centered_peclet_diagnostics",
]
