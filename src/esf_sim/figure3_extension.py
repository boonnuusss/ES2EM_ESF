"""Continue the four Figure 3 trajectories beyond their published snapshots.

The continuation is deliberately isolated from the canonical Figure 3 products.
It resumes the full three-level AB3 checkpoints, stores only the non-zero-time
extension frames, and pauses safely when CFL control would require an impractically
small base step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np

from ._model_impl import (
    CFLViolation,
    ESFModel,
    _archive_stable_checkpoint,
    _load_checkpoint,
    _save_checkpoint,
    _write_jsonl,
)
from .config import SimulationConfig
from .timeseries import ChunkedFrameArchive, validate_frame_archive


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = WORKSPACE_ROOT / "output" / "best_baseline" / "figure3_v2_timeseries"
DEFAULT_OUTPUT_ROOT = SOURCE_ROOT / "extended_1000s"


@dataclass(frozen=True)
class ExtensionCase:
    name: str
    alpha: float
    start_time_s: float
    target_time_s: float
    config_path: Path
    checkpoint_path: Path


def extension_cases() -> tuple[ExtensionCase, ...]:
    return (
        ExtensionCase(
            "alpha_0p2",
            0.2,
            3150.0,
            4150.0,
            SOURCE_ROOT
            / "alpha02_stability_calibration"
            / "exponential_conservative"
            / "recovery_config.json",
            SOURCE_ROOT
            / "alpha02_stability_calibration"
            / "exponential_conservative"
            / "checkpoints"
            / "pdm7.npz",
        ),
        ExtensionCase(
            "alpha_0p4",
            0.4,
            3150.0,
            4150.0,
            SOURCE_ROOT / "configs" / "alpha_0p4.json",
            SOURCE_ROOT / "runs" / "alpha_0p4" / "checkpoints" / "pdm7.npz",
        ),
        ExtensionCase(
            "alpha_0p6",
            0.6,
            3050.0,
            4050.0,
            SOURCE_ROOT / "configs" / "alpha_0p6.json",
            SOURCE_ROOT / "runs" / "alpha_0p6" / "checkpoints" / "pdm7.npz",
        ),
        ExtensionCase(
            "alpha_0p8",
            0.8,
            2860.0,
            3860.0,
            SOURCE_ROOT / "configs" / "alpha_0p8.json",
            SOURCE_ROOT / "runs" / "alpha_0p8" / "checkpoints" / "pdm7.npz",
        ),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ContinuationFrameArchive(ChunkedFrameArchive):
    """A chunked archive whose exact-time sequence begins after t=0."""

    def __init__(
        self,
        directory: str | Path,
        config: SimulationConfig,
        *,
        start_time_s: float,
        interval_s: float = 1.0,
        chunk_frames: int = 100,
    ) -> None:
        super().__init__(
            directory,
            config,
            interval_s=interval_s,
            chunk_frames=chunk_frames,
        )
        span = (config.t_end_s - start_time_s) / interval_s
        if start_time_s < 0.0 or span < 0.0 or not math.isclose(
            span, round(span), rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError(
                "continuation times must form an integer number of archive intervals"
            )
        self.start_time_s = float(start_time_s)
        self.expected_times = self.start_time_s + np.arange(
            int(round(span)) + 1, dtype=np.float64
        ) * self.interval_s

    def _write_manifest(self, *, completed: bool) -> None:
        chunks = []
        for path in self._chunk_paths():
            times = self._chunk_times(path)
            chunks.append(
                {
                    "file": path.name,
                    "first_time_s": float(times[0]),
                    "last_time_s": float(times[-1]),
                    "frame_count": int(times.size),
                    "bytes": path.stat().st_size,
                }
            )
        payload = {
            "format": "esf_continuation_npz_timeseries_v1",
            "completed": bool(completed and self.frame_count == self.expected_times.size),
            "frame_interval_s": self.interval_s,
            "frame_count": self.frame_count,
            "expected_frame_count": int(self.expected_times.size),
            "first_time_s": self.start_time_s,
            "target_time_s": float(self.config.t_end_s),
            "density_shape": [self.config.ny, self.config.nx],
            "solver_dtype": "float64",
            "storage_dtype": self.storage_dtype.name,
            "density_units": "m^-3",
            "pdm_alpha": self.config.pdm_alpha,
            "scheme": self.config.scheme,
            "potential_operator": self.config.potential_operator,
            "potential_boundary_y": self.config.potential_boundary_y,
            "chunks": chunks,
        }
        _atomic_json(self.manifest_path, payload)

    def finalize(self, *, completed: bool) -> dict[str, Any]:
        self.flush()
        self._write_manifest(completed=completed)
        return validate_frame_archive(
            self.directory,
            expected_times=self.expected_times if completed else None,
            expected_shape=(self.config.ny, self.config.nx),
            full=completed,
        )


def _status_payload(
    case: ExtensionCase,
    config: SimulationConfig,
    *,
    state: str,
    time_s: float,
    dt_s: float,
    step: int,
    minimum_dt_s: float,
    frame_count: int,
    reason: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "case": case.name,
        "state": state,
        "reason": reason,
        "start_time_s": case.start_time_s,
        "time_s": float(time_s),
        "target_time_s": case.target_time_s,
        "extension_progress_s": float(time_s - case.start_time_s),
        "dt_s": float(dt_s),
        "minimum_dt_s": float(minimum_dt_s),
        "step": int(step),
        "frame_count": int(frame_count),
        "expected_frame_count": int(case.target_time_s - case.start_time_s) + 1,
        "pdm_alpha": case.alpha,
        "scheme": config.scheme,
        "potential_operator": config.potential_operator,
        "potential_boundary_y": config.potential_boundary_y,
    }
    if diagnostics:
        payload.update(diagnostics)
    return payload


def _write_snapshot(
    config: SimulationConfig,
    density: np.ndarray,
    *,
    time_s: float,
    dt_s: float,
    start_mass: float,
    archive_directory: Path,
    state: str,
    reason: str | None,
) -> Path:
    path = config.output_directory / "simulations" / f"density_{config.scheme}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        density=density,
        x_km=config.x_coordinates_m / 1.0e3 - 125.0,
        y_km=config.y_coordinates_m / 1.0e3,
        time_s=time_s,
        dt_s=dt_s,
        scheme=config.scheme,
        pdm_alpha=config.pdm_alpha,
        potential_operator=config.potential_operator,
        potential_boundary_y=config.potential_boundary_y,
        extension_start_mass=start_mass,
        final_mass=float(np.sum(density) * config.dx_m * config.dy_m),
        frame_archive=str(archive_directory),
        frame_interval_s=1.0,
        extension_state=state,
        pause_reason=reason or "",
    )
    return path


def run_case(
    case: ExtensionCase,
    output_root: Path,
    *,
    minimum_dt_s: float,
    frame_chunk_size: int,
) -> dict[str, Any]:
    """Resume one case and return its terminal status record."""

    run_directory = output_root / "runs" / case.name
    checkpoint_path = run_directory / "checkpoints" / "pdm7.npz"
    archive_directory = output_root / "time_series" / case.name
    status_path = run_directory / "status.json"
    diagnostics_path = run_directory / "simulations" / "pdm7_diagnostics.jsonl"
    log_path = run_directory / "simulations" / "pdm7.log"
    run_directory.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    source_config = SimulationConfig.from_json(case.config_path)
    config = source_config.with_updates(
        output_dir=str(run_directory),
        t_end_s=case.target_time_s,
        checkpoint_interval_s=25.0,
    )
    config.validate()
    if not math.isclose(config.pdm_alpha, case.alpha, abs_tol=1.0e-15):
        raise ValueError(f"{case.name}: source config alpha does not match case definition")

    if not checkpoint_path.exists():
        if not case.checkpoint_path.exists():
            raise FileNotFoundError(case.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with np.load(case.checkpoint_path, allow_pickle=False) as saved:
            source_time = float(saved["time_s"])
        if not math.isclose(source_time, case.start_time_s, abs_tol=1.0e-8):
            raise ValueError(
                f"{case.name}: source checkpoint t={source_time:g}, expected {case.start_time_s:g}"
            )
        shutil.copy2(case.checkpoint_path, checkpoint_path)
        _atomic_json(
            run_directory / "provenance.json",
            {
                "case": case.name,
                "source_checkpoint": str(case.checkpoint_path),
                "source_checkpoint_sha256": _sha256(case.checkpoint_path),
                "source_config": str(case.config_path),
                "start_time_s": case.start_time_s,
                "target_time_s": case.target_time_s,
                "continuation_seconds": case.target_time_s - case.start_time_s,
                "minimum_dt_s": minimum_dt_s,
                "small_step_policy": (
                    "pause before retrying a CFL-rejected state when the required "
                    "halved base step would be below minimum_dt_s"
                ),
            },
        )
        _atomic_json(run_directory / "config.json", config.to_dict())

    density_history, time_s, dt_s, step = _load_checkpoint(checkpoint_path)
    if time_s < case.start_time_s - 1.0e-8 or time_s > case.target_time_s + 1.0e-8:
        raise ValueError(f"{case.name}: continuation checkpoint has invalid t={time_s:g}")

    logger = logging.getLogger(f"esf_sim.figure3_extension.{case.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.info("Continuation at t=%g s, target=%g s, dt=%g s", time_s, case.target_time_s, dt_s)

    archive = ContinuationFrameArchive(
        archive_directory,
        config,
        start_time_s=case.start_time_s,
        interval_s=1.0,
        chunk_frames=frame_chunk_size,
    )
    archive.prepare(resume_time_s=time_s, density=density_history[0], resume=True)
    start_mass = float(np.sum(density_history[0]) * config.dx_m * config.dy_m)
    model = ESFModel(config)
    next_checkpoint_time = (
        math.floor(time_s / config.checkpoint_interval_s) + 1
    ) * config.checkpoint_interval_s
    state = "running"
    reason: str | None = None
    last_diagnostics: dict[str, Any] = {}
    _atomic_json(
        status_path,
        _status_payload(
            case,
            config,
            state=state,
            time_s=time_s,
            dt_s=dt_s,
            step=step,
            minimum_dt_s=minimum_dt_s,
            frame_count=archive.frame_count,
        ),
    )

    while time_s < case.target_time_s - 1.0e-12:
        if dt_s < minimum_dt_s - 1.0e-15:
            state = "paused_minimum_dt"
            reason = (
                f"checkpoint base dt={dt_s:.12g} s is below the effective-progress "
                f"threshold {minimum_dt_s:.12g} s"
            )
            break
        dt_step = min(dt_s, case.target_time_s - time_s)
        if np.isfinite(archive.next_time_s):
            time_to_frame = archive.next_time_s - time_s
            if time_to_frame > 1.0e-12:
                dt_step = min(dt_step, time_to_frame)
        current = density_history[0]
        try:
            if len(density_history) < 3:
                updated, diagnostics = model.ssprk3_step(current, dt_step)
            else:
                updated, diagnostics = model.ab3_predictor_corrector_step(
                    current, density_history[1], density_history[2], dt_step
                )
        except CFLViolation as error:
            proposed_dt_s = dt_s * 0.5
            density_history = [current]
            dt_s = proposed_dt_s
            _save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            if proposed_dt_s < minimum_dt_s - 1.0e-15:
                state = "paused_minimum_dt"
                reason = (
                    f"{error}; required dt={proposed_dt_s:.12g} s is below the "
                    f"effective-progress threshold {minimum_dt_s:.12g} s"
                )
                logger.warning(reason)
                _archive_stable_checkpoint(checkpoint_path, time_s)
                break
            logger.warning(
                "%s; reducing dt to %g s and rebuilding AB3 history", error, dt_s
            )
            continue
        except Exception as error:
            _save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            state = "failed"
            reason = f"{type(error).__name__}: {error}"
            logger.exception("Continuation failed; checkpoint saved")
            break

        minimum_density = float(np.min(updated))
        if not np.all(np.isfinite(updated)) or minimum_density < -config.positivity_tolerance_m3:
            _save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            state = "failed"
            reason = f"invalid density at t={time_s:g}: minimum={minimum_density:.6e} m^-3"
            logger.error(reason)
            break
        if minimum_density < 0.0:
            updated = np.maximum(updated, 0.0)

        time_s += dt_step
        step += 1
        density_history.insert(0, updated)
        del density_history[3:]
        if math.isclose(time_s, archive.next_time_s, rel_tol=0.0, abs_tol=1.0e-8):
            archive.capture(time_s, updated)

        mass = float(np.sum(updated) * config.dx_m * config.dy_m)
        record = {
            "step": step,
            "time_s": time_s,
            "dt_s": dt_step,
            "base_dt_s": dt_s,
            "cfl": diagnostics.cfl_rate * dt_step,
            "maximum_speed_x_m_s": diagnostics.maximum_speed_x,
            "maximum_speed_y_m_s": diagnostics.maximum_speed_y,
            "potential_iterations": diagnostics.potential.iterations,
            "potential_relative_residual": diagnostics.potential.relative_residual,
            "preconditioner_rebuilt": diagnostics.potential.rebuilt_preconditioner,
            "extension_mass_relative_change": mass / start_mass - 1.0,
            "minimum_density_m3": minimum_density,
        }
        last_diagnostics = record
        if step % config.diagnostics_interval_steps == 0:
            _write_jsonl(diagnostics_path, record)

        checkpoint_due = time_s >= next_checkpoint_time
        frame_due = math.isclose(time_s, round(time_s), rel_tol=0.0, abs_tol=1.0e-8)
        if checkpoint_due or math.isclose(time_s, case.target_time_s, abs_tol=1.0e-8):
            _save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            stable = _archive_stable_checkpoint(checkpoint_path, time_s)
            while next_checkpoint_time <= time_s:
                next_checkpoint_time += config.checkpoint_interval_s
            logger.info(
                "step=%d t=%g base_dt=%g CFL=%g residual=%g stable_checkpoint=%s",
                step,
                time_s,
                dt_s,
                record["cfl"],
                record["potential_relative_residual"],
                stable.name,
            )
        if frame_due:
            _atomic_json(
                status_path,
                _status_payload(
                    case,
                    config,
                    state="running",
                    time_s=time_s,
                    dt_s=dt_s,
                    step=step,
                    minimum_dt_s=minimum_dt_s,
                    frame_count=archive.frame_count,
                    diagnostics=last_diagnostics,
                ),
            )

    if math.isclose(time_s, case.target_time_s, rel_tol=0.0, abs_tol=1.0e-8):
        state = "completed"
        reason = None
    else:
        _save_checkpoint(
            checkpoint_path,
            density_history,
            time_s=time_s,
            dt_s=dt_s,
            step=step,
        )
    archive.finalize(completed=state == "completed")
    snapshot = _write_snapshot(
        config,
        density_history[0],
        time_s=time_s,
        dt_s=dt_s,
        start_mass=start_mass,
        archive_directory=archive_directory,
        state=state,
        reason=reason,
    )
    status = _status_payload(
        case,
        config,
        state=state,
        time_s=time_s,
        dt_s=dt_s,
        step=step,
        minimum_dt_s=minimum_dt_s,
        frame_count=archive.frame_count,
        reason=reason,
        diagnostics=last_diagnostics,
    )
    status["snapshot"] = str(snapshot)
    status["archive"] = str(archive_directory)
    _atomic_json(status_path, status)
    logger.info("Terminal state=%s at t=%g s: %s", state, time_s, reason or "target reached")
    return status


def run_extensions(
    selected_cases: Iterable[str],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    minimum_dt_s: float = 1.0 / 64.0,
    frame_chunk_size: int = 100,
) -> dict[str, Any]:
    definitions = {case.name: case for case in extension_cases()}
    output_root.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, Any] = {}
    for name in selected_cases:
        case = definitions[name]
        try:
            statuses[name] = run_case(
                case,
                output_root,
                minimum_dt_s=minimum_dt_s,
                frame_chunk_size=frame_chunk_size,
            )
        except Exception as error:
            statuses[name] = {
                "case": name,
                "state": "launcher_failed",
                "reason": f"{type(error).__name__}: {error}",
            }
            _atomic_json(output_root / "runs" / name / "status.json", statuses[name])
        _atomic_json(
            output_root / "extension_report.json",
            {
                "format": "esf_figure3_extension_report_v1",
                "continuation_seconds_requested": 1000.0,
                "minimum_dt_s": minimum_dt_s,
                "cases": statuses,
            },
        )
    return statuses


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in extension_cases()],
        help="case to run; repeat for multiple cases (default: all four)",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--minimum-dt",
        type=float,
        default=1.0 / 64.0,
        help="pause when a CFL retry would require a smaller base step",
    )
    parser.add_argument("--frame-chunk-size", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    names = args.case or [case.name for case in extension_cases()]
    statuses = run_extensions(
        names,
        output_root=args.output_root,
        minimum_dt_s=args.minimum_dt,
        frame_chunk_size=args.frame_chunk_size,
    )
    print(json.dumps(statuses, indent=2, ensure_ascii=False))
    failed = any(status["state"] in {"failed", "launcher_failed"} for status in statuses.values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
