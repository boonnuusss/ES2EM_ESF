"""Public staged experiments with recoverable CFL adaptation."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import _experimental_impl as _impl
from . import _model_impl
from ._experimental_impl import (
    STAGES,
    ExperimentalESFModel,
    PointCenteredDirectPotentialSolver,
    StageSpec,
    build_point_centered_matrix,
    render_stage,
)
from .config import SimulationConfig
from .model import initial_density


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_simulation_with_cfl_recovery(
    config: SimulationConfig,
    *,
    resume: bool = True,
    maximum_steps: int | None = None,
) -> Path:
    """Run a stage while allowing a collapsed time step to recover conservatively."""

    config.validate()
    simulation_directory = config.output_directory / "simulations"
    simulation_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_directory / "checkpoints" / f"{config.scheme}.npz"
    diagnostics_path = simulation_directory / f"{config.scheme}_diagnostics.jsonl"
    log_path = simulation_directory / f"{config.scheme}.log"
    logger = logging.getLogger(f"esf_sim.experimental.{config.output_dir}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    if resume and checkpoint_path.exists():
        density_history, time_s, dt_s, step = _model_impl._load_checkpoint(
            checkpoint_path
        )
        logger.info("Resumed at t=%g s, step=%d, dt=%g s", time_s, step, dt_s)
    else:
        density_history = [initial_density(config)]
        time_s = 0.0
        dt_s = config.dt_s
        step = 0
        diagnostics_path.unlink(missing_ok=True)

    model = _model_impl.ESFModel(config)
    mass_reference = float(
        np.sum(initial_density(config)) * config.dx_m * config.dy_m
    )
    low_cfl_steps = 0

    while time_s < config.t_end_s - 1.0e-12:
        if maximum_steps is not None and step >= maximum_steps:
            break
        dt_step = min(dt_s, config.t_end_s - time_s)
        current = density_history[0]
        try:
            if len(density_history) < 3:
                updated, diagnostics = model.ssprk3_step(current, dt_step)
            else:
                updated, diagnostics = model.ab3_predictor_corrector_step(
                    current, density_history[1], density_history[2], dt_step
                )
        except _model_impl.CFLViolation as error:
            dt_s *= 0.5
            low_cfl_steps = 0
            density_history = [current]
            logger.warning(
                "%s; reducing dt to %g s and rebuilding history", error, dt_s
            )
            _model_impl._save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            continue
        except RuntimeError:
            _model_impl._save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            logger.exception("Potential solve failed; checkpoint saved")
            raise

        minimum_density = float(np.min(updated))
        if minimum_density < -config.positivity_tolerance_m3:
            _model_impl._save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            raise FloatingPointError(
                f"negative density {minimum_density:.6e} m^-3 exceeds tolerance"
            )
        if minimum_density < 0.0:
            updated = np.maximum(updated, 0.0)

        time_s += dt_step
        step += 1
        density_history.insert(0, updated)
        del density_history[3:]
        mass = float(np.sum(updated) * config.dx_m * config.dy_m)
        cfl = diagnostics.cfl_rate * dt_step
        record = {
            "step": step,
            "time_s": time_s,
            "dt_s": dt_step,
            "cfl": cfl,
            "maximum_speed_x_m_s": diagnostics.maximum_speed_x,
            "maximum_speed_y_m_s": diagnostics.maximum_speed_y,
            "potential_iterations": diagnostics.potential.iterations,
            "potential_relative_residual": diagnostics.potential.relative_residual,
            "preconditioner_rebuilt": diagnostics.potential.rebuilt_preconditioner,
            "mass_relative_change": mass / mass_reference - 1.0,
            "minimum_density_m3": minimum_density,
        }
        if step % config.diagnostics_interval_steps == 0:
            _write_jsonl(diagnostics_path, record)

        if cfl < 0.08 and dt_s < config.dt_s:
            low_cfl_steps += 1
        else:
            low_cfl_steps = 0
        if low_cfl_steps >= 50:
            dt_s = min(2.0 * dt_s, config.dt_s)
            density_history = [updated]
            low_cfl_steps = 0
            logger.info(
                "CFL remained below 0.08; increasing dt to %g s and rebuilding history",
                dt_s,
            )

        checkpoint_steps = max(
            1, int(round(config.checkpoint_interval_s / max(dt_s, 1.0e-12)))
        )
        if step % checkpoint_steps == 0 or math.isclose(time_s, config.t_end_s):
            _model_impl._save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            logger.info(
                "step=%d t=%g dt=%g CFL=%g residual=%g mass_change=%g",
                step,
                time_s,
                dt_step,
                cfl,
                record["potential_relative_residual"],
                record["mass_relative_change"],
            )

    snapshot_path = simulation_directory / f"density_{config.scheme}.npz"
    np.savez_compressed(
        snapshot_path,
        density=density_history[0],
        x_km=config.x_coordinates_m / 1.0e3 - 125.0,
        y_km=config.y_coordinates_m / 1.0e3,
        time_s=time_s,
        dt_s=dt_s,
        scheme=config.scheme,
        initial_mass=mass_reference,
        final_mass=float(np.sum(density_history[0]) * config.dx_m * config.dy_m),
    )
    return snapshot_path


def run_stage_case(*args, **kwargs):
    _impl.run_simulation = run_simulation_with_cfl_recovery
    return _impl.run_stage_case(*args, **kwargs)


def run_stage(*args, **kwargs):
    _impl.run_simulation = run_simulation_with_cfl_recovery
    _impl.run_stage_case = run_stage_case
    return _impl.run_stage(*args, **kwargs)


def build_parser():
    return _impl.build_parser()


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
