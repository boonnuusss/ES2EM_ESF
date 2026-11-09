"""Stable public interface for staged Figure 3 experiments."""

from __future__ import annotations

import json

from . import _experimental_impl as _core
from ._experimental_public_impl import (
    STAGES,
    ExperimentalESFModel,
    PointCenteredDirectPotentialSolver,
    StageSpec,
    build_point_centered_matrix,
    render_stage,
    run_simulation_with_cfl_recovery,
)
from .config import SimulationConfig


_run_stage_case_impl = _core.run_stage_case


def run_stage_case(*args, **kwargs):
    _core.run_simulation = run_simulation_with_cfl_recovery
    return _run_stage_case_impl(*args, **kwargs)


def run_stage(base, stage, alphas, *, resume=True, render=False):
    for alpha in alphas:
        run_stage_case(base, stage, alpha, resume=resume)
    return render_stage(base, stage) if render else None


def build_parser():
    return _core.build_parser()


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


__all__ = [
    "STAGES",
    "ExperimentalESFModel",
    "PointCenteredDirectPotentialSolver",
    "StageSpec",
    "build_point_centered_matrix",
    "render_stage",
    "run_simulation_with_cfl_recovery",
    "run_stage",
    "run_stage_case",
]


if __name__ == "__main__":
    raise SystemExit(main())
