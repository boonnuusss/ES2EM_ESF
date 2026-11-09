"""Cumulative numerical experiments for diagnosing Figure 3 morphology."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix

from . import _model_impl
from .config import SimulationConfig
from .figure3 import (
    FIGURE3_CASES,
    create_figure3,
    write_figure3_metrics,
    write_figure3_source_data,
)
from .model import run_simulation
from .potential import PotentialDiagnostics


@dataclass(frozen=True)
class StageSpec:
    name: str
    potential_operator: str
    potential_boundary_y: str
    face_velocity: str
    time_history: str
    description: str


STAGES = {
    "centered_dirichlet": StageSpec(
        name="centered_dirichlet",
        potential_operator="point_centered_five_point",
        potential_boundary_y="dirichlet_zero",
        face_velocity="cell_center_then_average",
        time_history="restart_after_dt_halving",
        description=(
            "Equation 2.21 is discretized directly with centered first and "
            "second derivatives and solved by Intel MKL PARDISO."
        ),
    ),
}


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
            "pypardiso is required for the paper-matched direct solver stage"
        ) from error
    return PyPardisoSolver


def build_point_centered_matrix(
    coefficient: np.ndarray,
    dx: float,
    dy: float,
    *,
    boundary_y: str,
) -> csr_matrix:
    """Discretize equation 2.21 directly on cell points."""

    a = np.asarray(coefficient, dtype=np.float64)
    if boundary_y != "dirichlet_zero":
        raise NotImplementedError(boundary_y)
    interior = a[1:-1]
    nyi, nx = interior.shape
    identifiers = np.arange(nyi * nx, dtype=np.int64).reshape(nyi, nx)
    gradient_x = (
        np.roll(interior, -1, axis=1) - np.roll(interior, 1, axis=1)
    ) / (2.0 * dx)
    gradient_y = (a[2:] - a[:-2]) / (2.0 * dy)
    drift_x = gradient_x / interior
    drift_y = gradient_y / interior

    row_parts = [identifiers.ravel(), identifiers.ravel(), identifiers.ravel()]
    column_parts = [
        identifiers.ravel(),
        np.roll(identifiers, -1, axis=1).ravel(),
        np.roll(identifiers, 1, axis=1).ravel(),
    ]
    data_parts = [
        np.full(identifiers.size, 2.0 / dx**2 + 2.0 / dy**2),
        (-1.0 / dx**2 - drift_x / (2.0 * dx)).ravel(),
        (-1.0 / dx**2 + drift_x / (2.0 * dx)).ravel(),
    ]
    if nyi > 1:
        row_parts.extend([identifiers[:-1].ravel(), identifiers[1:].ravel()])
        column_parts.extend([identifiers[1:].ravel(), identifiers[:-1].ravel()])
        data_parts.extend(
            [
                (-1.0 / dy**2 - drift_y[:-1] / (2.0 * dy)).ravel(),
                (-1.0 / dy**2 + drift_y[1:] / (2.0 * dy)).ravel(),
            ]
        )
    matrix = csr_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(identifiers.size, identifiers.size),
    )
    matrix.sort_indices()
    return matrix


class PointCenteredDirectPotentialSolver:
    """Paper-style point-centered operator with a persistent PARDISO solver."""

    def __init__(self, config: SimulationConfig, stage: StageSpec):
        self.config = config
        self.stage = stage
        self.collision_frequency = config.collision_nu0_s * np.exp(
            -config.y_coordinates_m / config.collision_frequency_scale_height_m
        )
        solver_class = _load_pardiso()
        self.solver = solver_class(mtype=11)
        self._pattern_indptr: np.ndarray | None = None
        self._pattern_indices: np.ndarray | None = None

    def _solve_reusing_symbolic_analysis(
        self, matrix: csr_matrix, rhs: np.ndarray
    ) -> np.ndarray:
        """Refactor changing values while retaining PARDISO's fixed sparsity analysis."""

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
            return np.asarray(self.solver.solve(matrix, rhs), dtype=np.float64)

        # Phase 23 repeats numerical factorization and solve without redoing the
        # symbolic analysis. Matrix values still change at every ESF time step.
        self.solver.set_phase(23)
        return np.asarray(
            self.solver._call_pardiso(matrix, rhs), dtype=np.float64
        )

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        coefficient = np.maximum(n, self.config.density_floor_m3)
        coefficient = coefficient * self.collision_frequency[:, None]
        matrix = build_point_centered_matrix(
            coefficient,
            self.config.dx_m,
            self.config.dy_m,
            boundary_y=self.stage.potential_boundary_y,
        )
        density_dx = (
            np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        rhs = (
            self.config.gravity_m_s2
            * density_dx[1:-1]
            / coefficient[1:-1]
        ).ravel()
        solution = self._solve_reusing_symbolic_analysis(matrix, rhs)
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        rebuilt_analysis = False
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            # A fixed ordering can become unsuitable after the coefficient field
            # develops extreme gradients. Retry with a complete PARDISO analysis,
            # reordering, numerical factorization, and solve before giving up.
            self._pattern_indptr = None
            self._pattern_indices = None
            self.solver.set_phase(13)
            solution = np.asarray(self.solver.solve(matrix, rhs), dtype=np.float64)
            residual = matrix @ solution - rhs
            relative_residual = float(np.linalg.norm(residual) / denominator)
            rebuilt_analysis = True
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            raise RuntimeError(
                f"PARDISO residual {relative_residual:.3e} exceeds 1e-8 after full rebuild"
            )
        potential = np.zeros_like(n)
        potential[1:-1] = solution.reshape(n.shape[0] - 2, n.shape[1])
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=rebuilt_analysis,
            used_pyamg=False,
        )


_ACTIVE_STAGE = STAGES["centered_dirichlet"]


class ExperimentalESFModel(_model_impl.ESFModel):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        if _ACTIVE_STAGE.potential_operator == "point_centered_five_point":
            self.potential_solver = PointCenteredDirectPotentialSolver(
                config, _ACTIVE_STAGE
            )


def _case_for_alpha(alpha: float):
    for case in FIGURE3_CASES:
        if np.isclose(case.alpha, alpha):
            return case
    raise ValueError(f"unsupported Figure 3 alpha {alpha}")


def _snapshot_complete(path: Path, expected_time_s: float) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as saved:
            return bool(
                saved["density"].shape == (401, 201)
                and np.isclose(float(saved["time_s"]), expected_time_s)
            )
    except (OSError, KeyError, ValueError):
        return False


def run_stage_case(
    base: SimulationConfig,
    stage: StageSpec,
    alpha: float,
    *,
    resume: bool = True,
) -> Path:
    global _ACTIVE_STAGE
    _ACTIVE_STAGE = stage
    _model_impl.ESFModel = ExperimentalESFModel
    case = _case_for_alpha(alpha)
    root = base.output_directory / "figure3_stages" / stage.name
    run_directory = root / "runs" / case.key
    config = base.with_updates(
        scheme="pdm7",
        pdm_alpha=case.alpha,
        t_end_s=case.time_s,
        output_dir=str(run_directory),
    )
    archive = config.to_dict()
    archive["experimental_stage"] = asdict(stage)
    archive_path = root / "configs" / f"{case.key}.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(
        json.dumps(archive, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stable = root / "simulations" / f"density_{case.key}.npz"
    if _snapshot_complete(stable, case.time_s):
        return stable
    raw = run_directory / "simulations" / "density_pdm7.npz"
    if not _snapshot_complete(raw, case.time_s):
        print(
            f"[experiment] {stage.name} alpha={case.alpha:g} "
            f"to t={case.time_s:g} s",
            flush=True,
        )
        raw = run_simulation(config, resume=resume)
    with np.load(raw) as saved:
        stable.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            stable,
            density=np.asarray(saved["density"], dtype=np.float64),
            x_km=np.asarray(saved["x_km"], dtype=np.float64),
            y_km=np.asarray(saved["y_km"], dtype=np.float64),
            time_s=float(saved["time_s"]),
            scheme="pdm7",
            pdm_alpha=case.alpha,
            experimental_stage=stage.name,
        )
    return stable


def render_stage(base: SimulationConfig, stage: StageSpec) -> dict[str, object]:
    root = base.output_directory / "figure3_stages" / stage.name
    snapshots = {
        case.key: root / "simulations" / f"density_{case.key}.npz"
        for case in FIGURE3_CASES
    }
    for case in FIGURE3_CASES:
        if not _snapshot_complete(snapshots[case.key], case.time_s):
            raise FileNotFoundError(snapshots[case.key])
    source = write_figure3_source_data(
        snapshots, root / f"figure3_{stage.name}_source_data.npz"
    )
    figures = create_figure3(snapshots, root)
    renamed = {}
    for extension, path in figures.items():
        destination = root / f"figure3_{stage.name}.{extension}"
        path.replace(destination)
        renamed[extension] = destination
    report = write_figure3_metrics(
        snapshots,
        reference_directory=Path(base.reference_mat).parent,
        output_path=root / f"figure3_{stage.name}_metrics.json",
    )
    shutil.copyfile(
        root / f"figure3_{stage.name}_metrics.json",
        root / f"figure3_{stage.name}_acceptance.json",
    )
    summary = {
        "stage": asdict(stage),
        "figures": {key: str(value) for key, value in renamed.items()},
        "source_data": str(source),
        "reference_thresholds_pass": report["all_reference_thresholds_pass"],
        "alpha_trend_pass": report[
            "all_scientific_alpha_trend_criteria_pass"
        ],
    }
    (root / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def run_stage(
    base: SimulationConfig,
    stage: StageSpec,
    alphas: Iterable[float],
    *,
    resume: bool = True,
    render: bool = False,
) -> dict[str, object] | None:
    for alpha in alphas:
        run_stage_case(base, stage, alpha, resume=resume)
    return render_stage(base, stage) if render else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.experimental")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--alpha", type=float, action="append")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = SimulationConfig.from_json(args.config)
    stage = STAGES[args.stage]
    alphas = args.alpha or [0.6, 0.2, 0.4, 0.8]
    summary = run_stage(
        base,
        stage,
        alphas,
        resume=not args.no_resume,
        render=args.render,
    )
    if summary is not None:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
