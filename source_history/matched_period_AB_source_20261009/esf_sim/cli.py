"""Unified command-line interface for the Figure 2 and Figure 3 workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import _cli_figure2
from .config import SimulationConfig
from .em_validation import run_em_validation
from .em_growth import EMGrowthConfig, run_em_growth_experiment
from .figure3 import reproduce_figure3
from .figure3_timeseries import (
    reproduce_figure3_timeseries,
    run_selected_timeseries_cases,
)


def _figure3_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m esf_sim reproduce-figure3",
        description=(
            "Run four PDM parameter cases and independently reproduce "
            "Li et al. (2021) Figure 3"
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    return parser


def _figure3_timeseries_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m esf_sim reproduce-figure3-timeseries",
        description=(
            "Reproduce Figure 3 on the formal v2 baseline and archive "
            "one density field per simulated second"
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        choices=("alpha_0p2", "alpha_0p4", "alpha_0p6", "alpha_0p8"),
        help="run and validate only this case; may be repeated",
    )
    return parser


def _validate_em_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m esf_sim validate-em",
        description=(
            "Run the nine fast reduced-EM validation stages without formal "
            "Li snapshots, PARDISO, or a long EPB simulation"
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "optional JSON file or directory; a directory receives "
            "em_validation_report.json"
        ),
    )
    return parser


def _run_em_growth_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m esf_sim run-em-growth",
        description=(
            "Run checkpointed paired Li electrostatic and dimensional "
            "2-D+1-D reduced-EM nonlinear growth trajectories"
        ),
    )
    parser.add_argument("--config", required=True, help="Li SimulationConfig JSON")
    parser.add_argument(
        "--em-config", required=True, help="dimensional EMGrowthConfig JSON"
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def run_em_growth_command(argv: list[str]) -> int:
    args = _run_em_growth_parser().parse_args(argv)
    li_config = SimulationConfig.from_json(args.config)
    em_config = EMGrowthConfig.from_json(args.em_config)
    report = run_em_growth_experiment(
        li_config, em_config, resume=not args.no_resume
    )
    print(
        json.dumps(
            {
                "report": str(
                    Path(em_config.output_dir) / "em_growth_report.json"
                ),
                "completed_time_s": report["completed_time_s"],
                "steps": report["steps"],
                "growth_rate_difference_s_inv": report[
                    "growth_rate_difference_s_inv"
                ],
                "relative_final_density_trajectory_difference": report[
                    "relative_final_density_trajectory_difference"
                ],
            },
            indent=2,
        )
    )
    return 0


def validate_em_command(argv: list[str]) -> int:
    args = _validate_em_parser().parse_args(argv)
    report = run_em_validation(args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_pass"] else 1


def reproduce_figure3_timeseries_command(argv: list[str]) -> int:
    args = _figure3_timeseries_parser().parse_args(argv)
    base = SimulationConfig.from_json(args.config)
    if args.case:
        selected_report = run_selected_timeseries_cases(
            base,
            tuple(dict.fromkeys(args.case)),
            simulate=not args.skip_simulation,
            resume=not args.no_resume,
        )
        print(json.dumps(selected_report, indent=2))
        return 0
    figures, report = reproduce_figure3_timeseries(
        base,
        simulate=not args.skip_simulation,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "figures": {key: str(value) for key, value in figures.items()},
                "total_frame_count": report["total_frame_count"],
                "alpha_0p4_matches_v2": report["alpha_0p4_v2_equivalence"].get(
                    "array_equal"
                ),
                "reference_thresholds_pass": report["reference_thresholds_pass"],
                "alpha_trend_pass": report["alpha_trend_pass"],
            },
            indent=2,
        )
    )
    return 0


def reproduce_figure3_command(argv: list[str]) -> int:
    args = _figure3_parser().parse_args(argv)
    base = SimulationConfig.from_json(args.config)
    figures, report = reproduce_figure3(
        base,
        simulate=not args.skip_simulation,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "figures": {key: str(value) for key, value in figures.items()},
                "reference_thresholds_pass": report[
                    "all_reference_thresholds_pass"
                ],
                "alpha_trend_pass": report[
                    "all_scientific_alpha_trend_criteria_pass"
                ],
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "validate-em":
        return validate_em_command(arguments[1:])
    if arguments and arguments[0] == "run-em-growth":
        return run_em_growth_command(arguments[1:])
    if arguments and arguments[0] == "reproduce-figure3":
        return reproduce_figure3_command(arguments[1:])
    if arguments and arguments[0] == "reproduce-figure3-timeseries":
        return reproduce_figure3_timeseries_command(arguments[1:])
    return _cli_figure2.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
