"""High-resolution Figure 6 solve with Galerkin-updated AMG operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pyamg.multilevel import coarse_grid_solver
from pyamg.relaxation import smoothing
from scipy.sparse.linalg import cg

from . import _model_impl
from .config import SimulationConfig
from .figure6 import FIGURE6_CASES, _archive_config, _normalize_snapshot, case_config
from .figure6_highres import UpdatedHierarchyPotentialSolver
from .model import run_simulation


class GalerkinUpdatedPotentialSolver(UpdatedHierarchyPotentialSolver):
    """Reuse P/R aggregates while updating every level to the current matrix."""

    def _update_galerkin_hierarchy(self, matrix) -> None:
        levels = self.hierarchy.levels
        levels[0].A = matrix
        for index in range(len(levels) - 1):
            levels[index + 1].A = (
                levels[index].R @ levels[index].A @ levels[index].P
            ).tocsr()
        for level in levels[:-1]:
            smoothing.rebuild_smoother(level)
        # Dense coarse solvers cache their factorization; reset that cache for
        # the newly projected coarsest operator.
        self.hierarchy.coarse_solver = coarse_grid_solver("pinv")

    def _solve_once(self, matrix, rhs, *, rebuild: bool):
        if rebuild or self.hierarchy is None:
            self.hierarchy = self._build_hierarchy(matrix)
            self.hierarchy_age = 0
        else:
            self._update_galerkin_hierarchy(matrix)
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

    def _build_hierarchy(self, matrix):
        from .potential import pyamg

        return pyamg.smoothed_aggregation_solver(matrix, symmetry="symmetric")


class GalerkinUpdatedESFModel(_model_impl.ESFModel):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.potential_solver = GalerkinUpdatedPotentialSolver(config)


def run_galerkin_high_resolution_case(
    base: SimulationConfig, *, resume: bool = True
) -> Path:
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
        "method": "Galerkin update of a reused PyAMG aggregation hierarchy",
        "updated_each_rhs": "all level matrices, relaxation routines, and coarse factor",
        "reused": "aggregation, restriction, and interpolation operators",
        "full_rebuild_trigger": "CG iterations > 160 or failed residual",
        "scientific_effect": "none; CG solves the current matrix to the configured tolerance",
    }
    config_path.write_text(json.dumps(archived, indent=2), encoding="utf-8")

    original_model = _model_impl.ESFModel
    _model_impl.ESFModel = GalerkinUpdatedESFModel
    try:
        source = run_simulation(config, resume=resume)
    finally:
        _model_impl.ESFModel = original_model
    stable = root / "simulations" / f"density_{case.key}.npz"
    return _normalize_snapshot(source, stable, case)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.figure6_galerkin")
    parser.add_argument("--config", default="configs/figure6.json")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig.from_json(args.config)
    print(run_galerkin_high_resolution_case(base, resume=not args.no_resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
