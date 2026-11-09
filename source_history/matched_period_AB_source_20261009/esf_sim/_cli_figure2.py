"""Command-line interface for validation and Figure 2 reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from .config import (
    CONSERVATIVE_HARMONIC_OPERATOR,
    DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
    SimulationConfig,
    VALID_SCHEMES,
)
from .model import run_simulation
from .plotting import create_figure2
from .report import write_metrics_report
from .validation import run_validation_suite


def _snapshot_matches_config(path: Path, config: SimulationConfig) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as saved:
            stored_operator = (
                str(saved["potential_operator"])
                if "potential_operator" in saved.files
                else CONSERVATIVE_HARMONIC_OPERATOR
            )
            stored_boundary = (
                str(saved["potential_boundary_y"])
                if "potential_boundary_y" in saved.files
                else DIRICHLET_ZERO_POTENTIAL_BOUNDARY
            )
            return bool(
                saved["density"].shape == (config.ny, config.nx)
                and np.isclose(float(saved["time_s"]), config.t_end_s)
                and str(saved["scheme"]) == config.scheme
                and stored_operator == config.potential_operator
                and stored_boundary == config.potential_boundary_y
            )
    except (OSError, KeyError, ValueError):
        return False


def _archive_config(config: SimulationConfig, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config.scheme}.json"
    payload = config.to_dict()
    payload["implementation_note"] = (
        "The Chinese thesis writes a 600 km upper boundary in one location; "
        "the English paper, 401-by-1-km grid, and plotted axis require 650 km."
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _sensitivity_report(
    baseline_path: Path,
    perturbed_path: Path,
    output_path: Path,
) -> None:
    with np.load(baseline_path) as baseline, np.load(perturbed_path) as perturbed:
        first = np.asarray(baseline["density"], dtype=np.float64)
        second = np.asarray(perturbed["density"], dtype=np.float64)
    difference = second - first
    report = {
        "relative_initial_perturbation": 1.0e-12,
        "relative_l2_final_difference": float(
            np.linalg.norm(difference) / np.linalg.norm(first)
        ),
        "maximum_relative_pointwise_difference": float(
            np.max(np.abs(difference)) / np.max(np.abs(first))
        ),
        "interpretation": (
            "Round-off-scale perturbations may change nonlinear fine-branch positions; "
            "branch-by-branch pixel alignment is therefore excluded from acceptance."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def reproduce_figure2(args: argparse.Namespace) -> int:
    base = SimulationConfig.from_json(args.config)
    output = base.output_directory
    snapshots: dict[str, Path] = {}
    for scheme in VALID_SCHEMES:
        config = base.with_updates(scheme=scheme)
        _archive_config(config, output / "configs")
        snapshot = output / "simulations" / f"density_{scheme}.npz"
        if not args.skip_simulation:
            print(f"[esf_sim] running {scheme} to t={config.t_end_s:g} s", flush=True)
            snapshot = run_simulation(config, resume=not args.no_resume)
        if not _snapshot_matches_config(snapshot, config):
            raise FileNotFoundError(
                f"missing or incompatible {snapshot}; run without "
                "--skip-simulation using the selected potential operator first"
            )
        snapshots[scheme] = snapshot

    if not args.skip_sensitivity:
        sensitivity_config = base.with_updates(
            scheme="pdm7",
            relative_noise=1.0e-12,
            output_dir=str(output / "sensitivity"),
        )
        _archive_config(sensitivity_config, output / "sensitivity" / "configs")
        perturbed = (
            sensitivity_config.output_directory
            / "simulations"
            / "density_pdm7.npz"
        )
        if not args.skip_simulation:
            print("[esf_sim] running 1e-12 PDM7 sensitivity case", flush=True)
            perturbed = run_simulation(
                sensitivity_config,
                resume=not args.no_resume,
            )
        if perturbed.exists():
            _sensitivity_report(
                snapshots["pdm7"],
                perturbed,
                output / "sensitivity" / "sensitivity_report.json",
            )

    print("[esf_sim] rendering Figure 2 outputs", flush=True)
    figure_paths = create_figure2(snapshots, output)
    report = write_metrics_report(
        snapshots,
        reference_directory=Path(base.reference_mat).parent,
        output_path=output / "figure2_metrics.json",
    )
    acceptance_path = output / "figure2_acceptance_report.json"
    shutil.copyfile(output / "figure2_metrics.json", acceptance_path)
    print(
        json.dumps(
            {
                "figures": {key: str(value) for key, value in figure_paths.items()},
                "reference_thresholds_pass": report["all_reference_thresholds_pass"],
                "hierarchy_pass": report["all_hierarchy_thresholds_pass"],
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m esf_sim",
        description="Independent numerical reproduction of Li et al. (2021) Figure 2",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="run numerical verification tests")
    validate.add_argument("--output", default="output/validation")

    reproduce = subparsers.add_parser(
        "reproduce-figure2",
        help="run the three ESF schemes and render Figure 2",
    )
    reproduce.add_argument("--config", required=True)
    reproduce.add_argument("--no-resume", action="store_true")
    reproduce.add_argument("--skip-simulation", action="store_true")
    reproduce.add_argument("--skip-sensitivity", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        report = run_validation_suite(args.output)
        print(json.dumps(report, indent=2))
        return 0 if report["all_pass"] else 1
    if args.command == "reproduce-figure2":
        return reproduce_figure2(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
