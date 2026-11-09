"""Complete the stabilized alpha=0.2 Figure 3 run and its one-second archive."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .alpha02_stability import (
    ANCHOR_TIME_S,
    TEST_END_TIME_S,
    _checkpoint_state,
    _diagnostic_summary,
    _matrix_health,
    _write_json,
    calibration_paths,
)
from .config import SimulationConfig
from .model import run_simulation
from .timeseries import validate_frame_archive


TARGET_TIME_S = 3150.0
TRANSITION_TIME_S = ANCHOR_TIME_S


def recovery_paths(base: SimulationConfig) -> dict[str, Path]:
    paths = calibration_paths(base)
    candidate = paths["candidate"]
    return {
        **paths,
        "frames": paths["formal"] / "time_series" / "alpha_0p2_recovered",
        "recovery_status": candidate / "recovery_status.json",
        "recovery_report": paths["root"] / "alpha02_recovery_report.json",
        "recovery_metadata": paths["formal"]
        / "time_series"
        / "alpha_0p2_recovered"
        / "recovery_metadata.json",
    }


def _recovery_config(base: SimulationConfig) -> SimulationConfig:
    paths = recovery_paths(base)
    return base.with_updates(
        pdm_alpha=0.2,
        potential_operator="exponential_conservative",
        potential_boundary_y="equal_extrapolation",
        t_end_s=TARGET_TIME_S,
        output_dir=str(paths["candidate"]),
        checkpoint_interval_s=100.0,
    )


def _copy_original_prefix(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("frames_??????_??????.npz")):
        with np.load(path, allow_pickle=False) as saved:
            times = np.asarray(saved["time_s"], dtype=np.float64)
        if float(times[-1]) <= TRANSITION_TIME_S - 1.0:
            target = destination / path.name
            if not target.exists():
                shutil.copy2(path, target)


def _write_transition_chunk(base: SimulationConfig) -> Path:
    paths = recovery_paths(base)
    destination = paths["frames"] / "frames_002300_002350.npz"
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            times = np.asarray(saved["time_s"], dtype=np.float64)
        if times.size == 51 and math.isclose(float(times[0]), 2300.0) and math.isclose(
            float(times[-1]), 2350.0
        ):
            return destination
        raise RuntimeError(f"inconsistent transition chunk: {destination}")

    densities: list[np.ndarray] = []
    times = np.arange(int(TRANSITION_TIME_S), int(TEST_END_TIME_S) + 1, dtype=float)
    for time_s in times:
        if math.isclose(time_s, TRANSITION_TIME_S):
            checkpoint = paths["anchor"]
        else:
            checkpoint = (
                paths["candidate"]
                / "checkpoints"
                / "stable"
                / f"pdm7_t{time_s:010.3f}.npz"
            )
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing transition checkpoint: {checkpoint}")
        density, saved_time, _, _ = _checkpoint_state(checkpoint)
        if not math.isclose(saved_time, time_s, abs_tol=1.0e-9):
            raise RuntimeError(f"checkpoint {checkpoint} stores t={saved_time:g}")
        densities.append(density.astype(np.float32))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        density_m3=np.stack(densities),
        time_s=times,
        x_km=base.x_coordinates_m / 1.0e3 - 125.0,
        y_km=base.y_coordinates_m / 1.0e3,
        interval_s=1.0,
        solver_dtype="float64",
        storage_dtype="float32",
        scheme=base.scheme,
        pdm_alpha=0.2,
        potential_operator="hybrid_transition_to_exponential_conservative",
        potential_boundary_y="equal_extrapolation",
    )
    temporary.replace(destination)
    return destination


def prepare_recovery_archive(base: SimulationConfig) -> Path:
    paths = recovery_paths(base)
    short_status = json.loads(paths["status"].read_text(encoding="utf-8"))
    if short_status.get("status") != "completed":
        raise RuntimeError("the t=2300--2350 stability calibration has not completed")
    density, time_s, _, _ = _checkpoint_state(paths["candidate_checkpoint"])
    if density.shape != (base.ny, base.nx) or not math.isclose(
        time_s, TEST_END_TIME_S, abs_tol=1.0e-9
    ):
        raise RuntimeError("candidate checkpoint is not the validated t=2350 state")
    _copy_original_prefix(paths["control_frames"], paths["frames"])
    _write_transition_chunk(base)
    metadata = {
        "format": "esf_alpha02_recovered_timeseries_v1",
        "transition_time_s": TRANSITION_TIME_S,
        "prefix": {
            "time_range_s": [0.0, TRANSITION_TIME_S],
            "potential_operator": "point_centered_equation_2_21_log_coefficients",
        },
        "recovered_segment": {
            "time_range_s": [TRANSITION_TIME_S, TARGET_TIME_S],
            "potential_operator": "exponential_conservative",
            "reason": "remove the non-physical near-zero point-centred elliptic mode",
        },
        "common_settings": {
            "scheme": "pdm7",
            "pdm_alpha": 0.2,
            "potential_boundary_y": "equal_extrapolation",
            "linear_solver": "Intel MKL PARDISO",
            "frame_interval_s": 1.0,
        },
    }
    _write_json(paths["recovery_metadata"], metadata)
    return paths["frames"]


def run_recovery(base: SimulationConfig) -> dict[str, Any]:
    paths = recovery_paths(base)
    if paths["recovery_status"].exists():
        saved = json.loads(paths["recovery_status"].read_text(encoding="utf-8"))
        if saved.get("status") == "completed":
            return saved
    frames = prepare_recovery_archive(base)
    config = _recovery_config(base)
    payload = config.to_dict()
    payload.update(
        {
            "recovery_transition_time_s": TRANSITION_TIME_S,
            "recovery_from_validated_short_window_s": TEST_END_TIME_S,
            "frame_archive": str(frames),
        }
    )
    _write_json(paths["candidate"] / "recovery_config.json", payload)
    try:
        snapshot = run_simulation(
            config,
            resume=True,
            frame_archive_directory=frames,
            frame_interval_s=1.0,
            frame_chunk_size=100,
        )
        status: dict[str, Any] = {
            "status": "completed",
            "snapshot": str(snapshot),
            "target_time_s": TARGET_TIME_S,
        }
    except (RuntimeError, FloatingPointError) as error:
        _, time_s, dt_s, step = _checkpoint_state(paths["candidate_checkpoint"])
        status = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "last_time_s": time_s,
            "last_dt_s": dt_s,
            "last_step": step,
        }
    _write_json(paths["recovery_status"], status)
    return status


def write_recovery_report(base: SimulationConfig, status: dict[str, Any]) -> Path:
    paths = recovery_paths(base)
    density, time_s, dt_s, step = _checkpoint_state(paths["candidate_checkpoint"])
    archive = validate_frame_archive(
        paths["frames"],
        expected_times=(
            np.arange(int(TARGET_TIME_S) + 1, dtype=np.float64)
            if status.get("status") == "completed"
            else None
        ),
        expected_shape=(base.ny, base.nx),
        full=status.get("status") == "completed",
    )
    report = {
        "status": status,
        "transition_time_s": TRANSITION_TIME_S,
        "checkpoint": {
            "time_s": time_s,
            "dt_s": dt_s,
            "step": step,
            "density_finite": bool(np.all(np.isfinite(density))),
            "minimum_density_m3": float(np.min(density)),
        },
        "diagnostics": _diagnostic_summary(
            paths["candidate"] / "simulations" / "pdm7_diagnostics.jsonl"
        ),
        "point_centered_health_of_final_density": _matrix_health(
            density, _recovery_config(base)
        ),
        "frame_archive": archive,
        "scientific_scope": (
            "This is a documented recovery trajectory.  The density is identical "
            "to the formal v2 run through t=2300 s; the potential discretization "
            "then switches to the positive conservative fitted operator."
        ),
    }
    _write_json(paths["recovery_report"], report)
    return paths["recovery_report"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/figure3.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    base = SimulationConfig.from_json(args.config).with_updates(pdm_alpha=0.2)
    paths = recovery_paths(base)
    status = (
        json.loads(paths["recovery_status"].read_text(encoding="utf-8"))
        if args.report_only
        else run_recovery(base)
    )
    report = write_recovery_report(base, status)
    print(f"[alpha02-recovery] status={status['status']}", flush=True)
    print(f"[alpha02-recovery] report={report}", flush=True)


if __name__ == "__main__":
    main()
