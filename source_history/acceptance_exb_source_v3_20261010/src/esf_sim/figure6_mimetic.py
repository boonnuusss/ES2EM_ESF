"""Continue the calibrated high-resolution Figure 6 branch from stable anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
    SimulationConfig,
)
from .figure6 import (
    FIGURE6_CASES,
    _normalize_snapshot,
    _snapshot_matches,
    case_by_key,
    case_config,
    create_figure6,
    write_figure6_metrics,
    write_figure6_source_data,
)
from .figure6_stability import _fixed_checkpoint_state, _write_json
from .model import run_simulation


CALIBRATION_ROOT = "figure6_stability_calibration"
RUN_ROOT = "figure6_mimetic"
CALIBRATED_301_ANCHOR_TIME_S = 2400.0


def _paths(base: SimulationConfig, case_key: str) -> dict[str, Path]:
    root = base.output_directory / RUN_ROOT
    run = root / "runs" / case_key
    return {
        "root": root,
        "run": run,
        "checkpoint": run / "checkpoints" / "pdm7.npz",
        "status": root / "status" / f"{case_key}.json",
        "config": root / "configs" / f"{case_key}.json",
        "anchor_archive": (
            root
            / "anchors"
            / f"{case_key}_t{CALIBRATED_301_ANCHOR_TIME_S:.0f}.npz"
        ),
    }


def calibrated_config(
    base: SimulationConfig, case_key: str, *, target_time_s: float | None = None
) -> SimulationConfig:
    case = case_by_key(case_key)
    target = case.time_s if target_time_s is None else float(target_time_s)
    return case_config(base, case, _paths(base, case_key)["run"]).with_updates(
        potential_operator=MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        t_end_s=target,
    )


def _candidate_301_checkpoint(base: SimulationConfig) -> Path:
    return (
        base.output_directory
        / CALIBRATION_ROOT
        / "mimetic_harmonic_log_rhs"
        / "checkpoints"
        / "pdm7.npz"
    )


def prepare_301_anchor(base: SimulationConfig) -> Path:
    """Copy only the verified t=2400 candidate; never use the failed v2 branch."""

    source = _candidate_301_checkpoint(base)
    _, time_s, _, _ = _fixed_checkpoint_state(source)
    if not np.isclose(time_s, CALIBRATED_301_ANCHOR_TIME_S):
        raise ValueError(
            f"calibrated 301x601 anchor is t={time_s:g}, expected "
            f"{CALIBRATED_301_ANCHOR_TIME_S:g}"
        )
    paths = _paths(base, "grid_301x601")
    archive = paths["anchor_archive"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copy2(source, archive)
    checkpoint = paths["checkpoint"]
    if not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, checkpoint)
    else:
        _, current_time_s, _, _ = _fixed_checkpoint_state(checkpoint)
        if current_time_s < CALIBRATED_301_ANCHOR_TIME_S:
            raise ValueError(
                f"continuation checkpoint regressed to t={current_time_s:g}"
            )
    return checkpoint


def _archive_config(
    config: SimulationConfig, path: Path, case_key: str, anchor: str
) -> None:
    payload = config.to_dict()
    payload["figure6_case"] = case_key
    payload["calibration_role"] = "high_resolution_stable_mimetic_branch"
    payload["continuation_anchor"] = anchor
    payload["changed_factor_only"] = (
        "potential left-hand matrix uses conservative harmonic face fluxes; "
        "the logarithmic centered RHS, equal-extrapolation boundary, PARDISO, "
        "PDM7 alpha=0.4, physical parameters and time integrator are unchanged"
    )
    _write_json(path, payload)


def run_case(
    base: SimulationConfig,
    case_key: str,
    *,
    target_time_s: float | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    paths = _paths(base, case_key)
    config = calibrated_config(base, case_key, target_time_s=target_time_s)
    if case_key == "grid_301x601":
        checkpoint = prepare_301_anchor(base)
        anchor_description = str(paths["anchor_archive"])
    else:
        checkpoint = paths["checkpoint"]
        anchor_description = "initial Chapman state at t=0"
    _archive_config(config, paths["config"], case_key, anchor_description)
    try:
        snapshot = run_simulation(config, resume=resume)
        _, time_s, dt_s, step = _fixed_checkpoint_state(checkpoint)
        status: dict[str, Any] = {
            "status": "completed",
            "snapshot": str(snapshot),
            "time_s": time_s,
            "dt_s": dt_s,
            "step": step,
            "potential_operator": config.potential_operator,
            "potential_boundary_y": config.potential_boundary_y,
        }
    except (RuntimeError, FloatingPointError) as error:
        _, time_s, dt_s, step = _fixed_checkpoint_state(checkpoint)
        status = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "time_s": time_s,
            "dt_s": dt_s,
            "step": step,
            "potential_operator": config.potential_operator,
            "potential_boundary_y": config.potential_boundary_y,
        }
    _write_json(paths["status"], status)
    return status


def finalize_figure6(base: SimulationConfig) -> dict[str, Any]:
    root = base.output_directory / RUN_ROOT
    snapshots: dict[str, Path] = {}
    for case in FIGURE6_CASES:
        source = (
            root / "runs" / case.key / "simulations" / "density_pdm7.npz"
        )
        if not _snapshot_matches(
            source,
            case,
            MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
            BEST_BASELINE_POTENTIAL_BOUNDARY,
        ):
            raise FileNotFoundError(f"missing calibrated snapshot {source}")
        stable = root / "simulations" / f"density_{case.key}.npz"
        snapshots[case.key] = _normalize_snapshot(source, stable, case)
    figures = create_figure6(snapshots, root)
    source_data = write_figure6_source_data(
        snapshots, root / "figure6_source_data.npz"
    )
    report = write_figure6_metrics(
        base, snapshots, root / "figure6_metrics.json"
    )
    shutil.copyfile(
        root / "figure6_metrics.json",
        root / "figure6_acceptance_report.json",
    )
    result = {
        "figures": {key: str(value) for key, value in figures.items()},
        "source_data": str(source_data),
        "all_reference_thresholds_pass": report[
            "all_reference_thresholds_pass"
        ],
    }
    _write_json(root / "figure6_outputs.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/figure6.json")
    parser.add_argument("--case", default="grid_301x601")
    parser.add_argument("--target-time", type=float)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    base = SimulationConfig.from_json(args.config)
    status = run_case(
        base,
        args.case,
        target_time_s=args.target_time,
        resume=not args.no_resume,
    )
    print(json.dumps(status, indent=2), flush=True)
    if args.render:
        print(json.dumps(finalize_figure6(base), indent=2), flush=True)


if __name__ == "__main__":
    main()
