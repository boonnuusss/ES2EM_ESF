"""Efficient high-resolution Figure 6 solve with an updated AMG fine level.

The elliptic matrix still changes every right-hand-side evaluation and CG is
still applied to that current matrix.  Only the expensive AMG interpolation
hierarchy is reused; PyAMG's public ``change_solve_matrix`` method updates the
fine-grid operator and relaxation routines before each solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import cg

from . import _model_impl
from .config import SimulationConfig
from .figure6 import (
    FIGURE6_CASES,
    _archive_config,
    _normalize_snapshot,
    case_config,
)
from .model import run_simulation
from .potential import (
    PotentialDiagnostics,
    PotentialSolveError,
    build_elliptic_matrix,
    pyamg,
)


class UpdatedHierarchyPotentialSolver:
    """CG solver that updates the AMG fine operator without rebuilding coarsening."""

    def __init__(self, config: SimulationConfig):
        if pyamg is None:
            raise RuntimeError("PyAMG is required for the high-resolution workflow")
        self.config = config
        self.collision_frequency = config.collision_nu0_s * np.exp(
            -config.y_coordinates_m / config.collision_frequency_scale_height_m
        )
        self.hierarchy = None
        self.last_solution = None
        self.hierarchy_age = 0

    def _solve_once(self, matrix, rhs, *, rebuild: bool):
        if rebuild or self.hierarchy is None:
            self.hierarchy = pyamg.smoothed_aggregation_solver(
                matrix, symmetry="symmetric"
            )
            self.hierarchy_age = 0
        else:
            self.hierarchy.change_solve_matrix(matrix)
        preconditioner = self.hierarchy.aspreconditioner(cycle="V")
        iterations = 0

        def callback(_):
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            matrix,
            rhs,
            x0=self.last_solution,
            M=preconditioner,
            rtol=self.config.potential_rtol,
            atol=0.0,
            maxiter=self.config.potential_maxiter,
            callback=callback,
        )
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        return solution, int(info), iterations, relative_residual

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        coefficient = np.maximum(n, self.config.density_floor_m3)
        coefficient *= self.collision_frequency[:, None]
        matrix, scale = build_elliptic_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        density_dx = (
            np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        rhs = (self.config.gravity_m_s2 * density_dx[1:-1] / scale).ravel()

        periodic_rebuild = self.hierarchy_age >= self.config.preconditioner_refresh_steps
        rebuilt = self.hierarchy is None or periodic_rebuild
        solution, info, iterations, relative_residual = self._solve_once(
            matrix, rhs, rebuild=rebuilt
        )
        needs_rebuild = (
            info != 0
            or not np.isfinite(relative_residual)
            or relative_residual > 10.0 * self.config.potential_rtol
            or iterations > self.config.preconditioner_iteration_limit
        )
        if needs_rebuild and not rebuilt:
            solution, info, iterations, relative_residual = self._solve_once(
                matrix, rhs, rebuild=True
            )
            rebuilt = True
        if (
            info != 0
            or not np.isfinite(relative_residual)
            or relative_residual > 10.0 * self.config.potential_rtol
        ):
            raise PotentialSolveError(
                f"updated-hierarchy CG failed: info={info}, iterations={iterations}, "
                f"relative residual={relative_residual:.3e}"
            )
        self.last_solution = solution
        self.hierarchy_age += 1
        potential = np.zeros_like(n)
        potential[1:-1] = solution.reshape(n.shape[0] - 2, n.shape[1])
        return potential, PotentialDiagnostics(
            iterations=iterations,
            relative_residual=relative_residual,
            rebuilt_preconditioner=rebuilt,
            used_pyamg=True,
        )


class UpdatedHierarchyESFModel(_model_impl.ESFModel):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.potential_solver = UpdatedHierarchyPotentialSolver(config)


def run_high_resolution_case(base: SimulationConfig, *, resume: bool = True) -> Path:
    case = FIGURE6_CASES[-1]
    root = base.output_directory / "figure6"
    run_root = root / "runs" / case.key
    tuned = base.with_updates(
        preconditioner_refresh_steps=100000,
        preconditioner_iteration_limit=160,
    )
    config = case_config(tuned, case, run_root)
    config.validate()
    config_path = root / "configs" / f"{case.key}.json"
    _archive_config(config, config_path, case)
    archived = json.loads(config_path.read_text(encoding="utf-8"))
    archived["elliptic_solver_optimization"] = {
        "method": "PyAMG change_solve_matrix on every RHS",
        "updated_each_rhs": "fine-grid matrix and relaxation routines",
        "reused": "coarse interpolation hierarchy",
        "full_rebuild_trigger": "CG iterations > 160 or failed residual",
        "scientific_effect": "none; CG solves the current matrix to the configured tolerance",
    }
    config_path.write_text(json.dumps(archived, indent=2), encoding="utf-8")

    original_model = _model_impl.ESFModel
    _model_impl.ESFModel = UpdatedHierarchyESFModel
    try:
        source = run_simulation(config, resume=resume)
    finally:
        _model_impl.ESFModel = original_model
    stable = root / "simulations" / f"density_{case.key}.npz"
    return _normalize_snapshot(source, stable, case)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.figure6_highres")
    parser.add_argument("--config", default="configs/figure6.json")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig.from_json(args.config)
    print(run_high_resolution_case(base, resume=not args.no_resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
