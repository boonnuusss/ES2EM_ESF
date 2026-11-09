"""Cached PARDISO potential solvers and an automatic monotonicity safeguard.

Only the numerical values of an ESF five-point operator change in time.  The
legacy implementation rebuilt and sorted its sparse topology at every RHS
evaluation.  These solvers build that topology once, update ``CSR.data`` in
place, and retain PARDISO's symbolic analysis.

``AdaptiveMonotonePotentialSolver`` starts with the calibrated centred form
and permanently switches to the positive exponential-fitted conservative form
before the centred stencil loses its M-matrix property.  A one-way switch is
intentional: it avoids temporal chatter and discontinuous repeated changes of
the semi-discrete operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .config import SimulationConfig
from .point_centered_potential import _load_pardiso
from .potential import PotentialDiagnostics
from .stabilized_potential import exponential_face_mean


@dataclass(frozen=True)
class AdaptivePotentialDiagnostics:
    mode: str
    maximum_peclet: float
    switched_this_solve: bool


class _PardisoCachedBase:
    def __init__(self, config: SimulationConfig, matrix: csr_matrix):
        self.config = config
        self.matrix = matrix
        self.solver = _load_pardiso()(mtype=11)
        self._analysed = False

    def _linear_solve(self, rhs: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self._analysed:
            self.solver.set_phase(13)
            result = self.solver.solve(self.matrix, rhs)
            self._analysed = True
            return np.asarray(result, dtype=np.float64), True
        self.solver.set_phase(23)
        return np.asarray(
            self.solver._call_pardiso(self.matrix, rhs), dtype=np.float64
        ), False

    def _retry(self, rhs: np.ndarray) -> np.ndarray:
        self.solver.set_phase(13)
        result = np.asarray(self.solver.solve(self.matrix, rhs), dtype=np.float64)
        self._analysed = True
        return result


def _point_topology(ny: int, nx: int):
    size = ny * nx
    indptr = [0]
    indices: list[int] = []
    positions = {name: np.full(size, -1, dtype=np.int64) for name in (
        "diag", "east", "west", "north", "south"
    )}
    for node in range(size):
        if node == 0:  # gauge row
            candidates = [(node, "diag")]
        else:
            j, i = divmod(node, nx)
            candidates = [
                (node, "diag"),
                (j * nx + (i + 1) % nx, "east"),
                (j * nx + (i - 1) % nx, "west"),
            ]
            if j + 1 < ny:
                candidates.append((node + nx, "north"))
            if j:
                candidates.append((node - nx, "south"))
        for column, name in sorted(candidates):
            positions[name][node] = len(indices)
            indices.append(column)
        indptr.append(len(indices))
    matrix = csr_matrix(
        (np.zeros(len(indices)), np.asarray(indices), np.asarray(indptr)),
        shape=(size, size),
    )
    return matrix, positions


def _reduced_conservative_topology(ny: int, nx: int):
    size = ny * nx
    indptr = [0]
    indices: list[int] = []
    positions = {name: np.full(size, -1, dtype=np.int64) for name in (
        "diag", "east", "west", "north", "south"
    )}
    for node in range(1, size):
        j, i = divmod(node, nx)
        candidates = [
            (node, "diag"),
            (j * nx + (i + 1) % nx, "east"),
            (j * nx + (i - 1) % nx, "west"),
        ]
        if j + 1 < ny:
            candidates.append((node + nx, "north"))
        if j:
            candidates.append((node - nx, "south"))
        for global_column, name in sorted(candidates):
            if global_column == 0:
                continue
            positions[name][node] = len(indices)
            indices.append(global_column - 1)
        indptr.append(len(indices))
    matrix = csr_matrix(
        (np.zeros(len(indices)), np.asarray(indices), np.asarray(indptr)),
        shape=(size - 1, size - 1),
    )
    return matrix, positions


def _put(data: np.ndarray, positions: np.ndarray, values: np.ndarray) -> None:
    valid = positions >= 0
    data[positions[valid]] = values[valid]


class CachedPointCenteredPotentialSolver(_PardisoCachedBase):
    """Exact legacy point-centred operator with cached CSR topology."""

    def __init__(self, config: SimulationConfig):
        self.collision_frequency = config.collision_nu0_s * np.exp(
            -config.y_coordinates_m / config.collision_frequency_scale_height_m
        )
        matrix, self.positions = _point_topology(config.ny, config.nx)
        super().__init__(config, matrix)

    def _assemble(self, positive_n: np.ndarray) -> tuple[np.ndarray, float]:
        c = self.config
        log_a = np.log(positive_n * self.collision_frequency[:, None])
        drift_x = (np.roll(log_a, -1, axis=1) - np.roll(log_a, 1, axis=1)) / (2*c.dx_m)
        padded = np.pad(log_a, ((1, 1), (0, 0)), mode="edge")
        drift_y = (padded[2:] - padded[:-2]) / (2*c.dy_m)
        diag = np.full_like(log_a, 2/c.dx_m**2 + 2/c.dy_m**2)
        diag[0] = 2/c.dx_m**2 + 1/c.dy_m**2 + drift_y[0]/(2*c.dy_m)
        diag[-1] = 2/c.dx_m**2 + 1/c.dy_m**2 - drift_y[-1]/(2*c.dy_m)
        values = {
            "diag": diag.ravel(),
            "east": (-1/c.dx_m**2 - drift_x/(2*c.dx_m)).ravel(),
            "west": (-1/c.dx_m**2 + drift_x/(2*c.dx_m)).ravel(),
            "north": (-1/c.dy_m**2 - drift_y/(2*c.dy_m)).ravel(),
            "south": (-1/c.dy_m**2 + drift_y/(2*c.dy_m)).ravel(),
        }
        for name, array in values.items():
            _put(self.matrix.data, self.positions[name], array)
        self.matrix.data[self.positions["diag"][0]] = 1.0
        log_n = np.log(positive_n)
        grad = (np.roll(log_n, -1, axis=1)-np.roll(log_n, 1, axis=1))/(2*c.dx_m)
        rhs = (c.gravity_m_s2*grad/self.collision_frequency[:, None]).ravel()
        rhs[0] = 0.0
        peclet = max(float(np.max(np.abs(drift_x)*c.dx_m/2)),
                      float(np.max(np.abs(drift_y)*c.dy_m/2)))
        return rhs, peclet

    def solve(self, density: np.ndarray):
        positive = np.maximum(np.asarray(density, dtype=np.float64), self.config.density_floor_m3)
        rhs, _ = self._assemble(positive)
        solution, rebuilt = self._linear_solve(rhs)
        residual = self.matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative = float(np.linalg.norm(residual)/denominator)
        if not np.isfinite(relative) or relative > self.config.potential_rtol:
            solution = self._retry(rhs)
            relative = float(np.linalg.norm(self.matrix @ solution-rhs)/denominator)
            rebuilt = True
        if not np.isfinite(relative) or relative > self.config.potential_rtol:
            raise RuntimeError(f"cached point-centred residual {relative:.3e} exceeds {self.config.potential_rtol:g}")
        potential = solution.reshape(positive.shape)
        potential -= float(np.mean(potential))
        return potential, PotentialDiagnostics(1, relative, rebuilt, False)

    def peclet(self, density: np.ndarray) -> float:
        positive = np.maximum(np.asarray(density), self.config.density_floor_m3)
        _, peclet = self._assemble(positive)
        return peclet


class CachedExponentialPotentialSolver(_PardisoCachedBase):
    """Positive conservative exponential-fitted operator with cached CSR."""

    def __init__(self, config: SimulationConfig):
        self.collision_frequency = config.collision_nu0_s * np.exp(
            -config.y_coordinates_m / config.collision_frequency_scale_height_m
        )
        matrix, self.positions = _reduced_conservative_topology(config.ny, config.nx)
        super().__init__(config, matrix)

    def _assemble(self, positive_n: np.ndarray):
        c = self.config
        raw = positive_n*self.collision_frequency[:, None]
        scale = float(np.max(raw))
        a = raw/scale
        east = exponential_face_mean(a, np.roll(a, -1, axis=1))
        west = np.roll(east, 1, axis=1)
        north = np.zeros_like(a)
        north[:-1] = exponential_face_mean(a[:-1], a[1:])
        south = np.zeros_like(a)
        south[1:] = north[:-1]
        values = {
            "diag": ((east+west)/c.dx_m**2+(north+south)/c.dy_m**2).ravel(),
            "east": (-east/c.dx_m**2).ravel(),
            "west": (-west/c.dx_m**2).ravel(),
            "north": (-north/c.dy_m**2).ravel(),
            "south": (-south/c.dy_m**2).ravel(),
        }
        for name, array in values.items():
            _put(self.matrix.data, self.positions[name], array)
        grad = (np.roll(positive_n,-1,axis=1)-np.roll(positive_n,1,axis=1))/(2*c.dx_m)
        rhs = (c.gravity_m_s2*grad/scale).ravel()
        return rhs, values

    def solve(self, density: np.ndarray):
        positive = np.maximum(np.asarray(density, dtype=np.float64), self.config.density_floor_m3)
        full_rhs, values = self._assemble(positive)
        solution_reduced, rebuilt = self._linear_solve(full_rhs[1:])
        solution = np.empty_like(full_rhs)
        solution[0], solution[1:] = 0.0, solution_reduced
        reduced_residual = self.matrix @ solution_reduced-full_rhs[1:]
        nx = self.config.nx
        r0 = (values["east"][0]*solution[1]
              + values["west"][0]*solution[nx-1]
              + values["north"][0]*solution[nx] - full_rhs[0])
        denominator = max(float(np.linalg.norm(full_rhs)), np.finfo(float).tiny)
        relative = float(np.sqrt(np.dot(reduced_residual,reduced_residual)+r0*r0)/denominator)
        if not np.isfinite(relative) or relative > self.config.potential_rtol:
            solution_reduced = self._retry(full_rhs[1:])
            solution[1:] = solution_reduced
            reduced_residual = self.matrix @ solution_reduced-full_rhs[1:]
            r0 = (values["east"][0]*solution[1]+values["west"][0]*solution[nx-1]
                  + values["north"][0]*solution[nx]-full_rhs[0])
            relative = float(np.sqrt(np.dot(reduced_residual,reduced_residual)+r0*r0)/denominator)
            rebuilt = True
        if not np.isfinite(relative) or relative > self.config.potential_rtol:
            raise RuntimeError(f"cached exponential residual {relative:.3e} exceeds {self.config.potential_rtol:g}")
        potential = solution.reshape(positive.shape)
        potential -= float(np.mean(potential))
        return potential, PotentialDiagnostics(1, relative, rebuilt, False)


class AdaptiveMonotonePotentialSolver:
    """Accurate centred solve followed by a one-way monotone safety switch."""

    def __init__(self, config: SimulationConfig, switch_peclet: float = 0.75):
        if not 0.0 < switch_peclet < 1.0:
            raise ValueError("switch_peclet must lie in (0, 1)")
        self.config = config
        self.switch_peclet = switch_peclet
        self.centered = CachedPointCenteredPotentialSolver(config)
        self.exponential: CachedExponentialPotentialSolver | None = None
        self.mode = "point_centered"
        self.last_adaptive_diagnostics = AdaptivePotentialDiagnostics(self.mode, 0.0, False)

    def solve(self, density: np.ndarray):
        peclet = self.centered.peclet(density)
        switched = False
        if self.mode == "point_centered" and peclet >= self.switch_peclet:
            self.mode = "exponential_conservative"
            self.exponential = CachedExponentialPotentialSolver(self.config)
            switched = True
        self.last_adaptive_diagnostics = AdaptivePotentialDiagnostics(self.mode, peclet, switched)
        if self.mode == "point_centered":
            return self.centered.solve(density)
        assert self.exponential is not None
        return self.exponential.solve(density)


__all__ = [
    "AdaptiveMonotonePotentialSolver",
    "AdaptivePotentialDiagnostics",
    "CachedExponentialPotentialSolver",
    "CachedPointCenteredPotentialSolver",
]
