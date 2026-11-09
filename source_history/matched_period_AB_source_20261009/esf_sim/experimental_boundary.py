"""Stage 2: test equal-extrapolation potential boundaries."""

from __future__ import annotations

import json

import numpy as np
from scipy.sparse import csr_matrix

from . import _experimental_impl as _core
from . import _model_impl
from .config import SimulationConfig
from .experimental import run_stage
from .potential import PotentialDiagnostics, harmonic_mean


_BASE_MODEL = _model_impl.ESFModel


STAGE = _core.StageSpec(
    name="harmonic_extrapolate",
    potential_operator="conservative_harmonic",
    potential_boundary_y="equal_extrapolation",
    face_velocity="cell_center_then_average",
    time_history="restart_with_cfl_recovery",
    description=(
        "Retain the conservative harmonic operator and replace zero potential "
        "at the altitude boundaries by zero-normal-gradient extrapolation."
    ),
)
_core.STAGES[STAGE.name] = STAGE


def build_harmonic_neumann_matrix(
    coefficient: np.ndarray, dx: float, dy: float
) -> tuple[csr_matrix, float]:
    raw = np.asarray(coefficient, dtype=np.float64)
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
    # Replace one redundant conservation equation by a gauge condition psi=0.
    start, end = matrix.indptr[0], matrix.indptr[1]
    matrix.data[start:end] = 0.0
    gauge = np.where(matrix.indices[start:end] == 0)[0]
    matrix.data[start + int(gauge[0])] = 1.0
    return matrix, scale


class HarmonicExtrapolationPotentialSolver:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.collision_frequency = config.collision_nu0_s * np.exp(
            -config.y_coordinates_m / config.collision_frequency_scale_height_m
        )
        solver_class = _core._load_pardiso()
        self.solver = solver_class(mtype=11)

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        coefficient = np.maximum(n, self.config.density_floor_m3)
        coefficient = coefficient * self.collision_frequency[:, None]
        matrix, scale = build_harmonic_neumann_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        density_dx = (
            np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        rhs = (self.config.gravity_m_s2 * density_dx / scale).ravel()
        rhs[0] = 0.0
        solution = np.asarray(self.solver.solve(matrix, rhs), dtype=np.float64)
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            raise RuntimeError(
                f"PARDISO residual {relative_residual:.3e} exceeds 1e-8"
            )
        potential = solution.reshape(n.shape)
        potential -= np.mean(potential)
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=False,
            used_pyamg=False,
        )


class BoundaryExperimentalModel(_BASE_MODEL):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.potential_solver = HarmonicExtrapolationPotentialSolver(config)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m esf_sim.experimental_boundary")
    parser.add_argument("--config", required=True)
    parser.add_argument("--alpha", type=float, action="append")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    _core.ExperimentalESFModel = BoundaryExperimentalModel
    base = SimulationConfig.from_json(args.config)
    summary = run_stage(
        base,
        STAGE,
        args.alpha or [0.6, 0.2, 0.4, 0.8],
        resume=not args.no_resume,
        render=args.render,
    )
    if summary is not None:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
