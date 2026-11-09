"""Stage 4: variable-step AB predictor without repeated Runge-Kutta restarts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import shutil

import numpy as np

from . import _model_impl
from .config import SimulationConfig
from .figure3 import (
    FIGURE3_CASES,
    create_figure3,
    write_figure3_metrics,
    write_figure3_source_data,
)
from .model import initial_density


STAGE_NAME = "variable_history_ab3"
STAGE_DESCRIPTION = {
    "name": STAGE_NAME,
    "potential_operator": "from_simulation_config",
    "potential_boundary_y": "from_simulation_config",
    "face_velocity": "uniform_grid_equivalent_staggered",
    "time_history": "variable_step_lagrange_predictor_without_restart",
}


def variable_step_predictor(
    states: list[np.ndarray], times: list[float], target_time: float
) -> np.ndarray:
    if len(states) < 3 or len(times) < 3:
        raise ValueError("three states and times are required")
    selected_states = states[:3]
    selected_times = times[:3]
    result = np.zeros_like(selected_states[0])
    for index in range(3):
        weight = 1.0
        for other in range(3):
            if other == index:
                continue
            weight *= (target_time - selected_times[other]) / (
                selected_times[index] - selected_times[other]
            )
        result += weight * selected_states[index]
    return result


def _save_checkpoint(
    path: Path,
    states: list[np.ndarray],
    times: list[float],
    *,
    dt_s: float,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        states=np.stack(states),
        times=np.asarray(times),
        dt_s=dt_s,
        step=step,
    )
    temporary.replace(path)


def _load_checkpoint(path: Path):
    with np.load(path) as saved:
        states = [item.copy() for item in np.asarray(saved["states"])]
        times = [float(value) for value in np.asarray(saved["times"])]
        return states, times, float(saved["dt_s"]), int(saved["step"])


def convert_fixed_history_checkpoint(source: Path, destination: Path) -> None:
    """Convert a production fixed-step checkpoint to timestamped history.

    The production runner discards its history whenever ``dt`` changes. Thus,
    whenever three states are present, they are separated by the stored
    current time step and can be assigned exact timestamps.
    """

    with np.load(source) as saved:
        history_length = int(saved["history_length"])
        states = [np.asarray(saved["density"], dtype=np.float64).copy()]
        if history_length > 1:
            states.append(
                np.asarray(saved["density_previous"], dtype=np.float64).copy()
            )
        if history_length > 2:
            states.append(
                np.asarray(saved["density_previous2"], dtype=np.float64).copy()
            )
        time_s = float(saved["time_s"])
        dt_s = float(saved["dt_s"])
        step = int(saved["step"])
    times = [time_s - index * dt_s for index in range(len(states))]
    _save_checkpoint(
        destination,
        states,
        times,
        dt_s=dt_s,
        step=step,
    )


def _archive_stable_checkpoint(checkpoint: Path, time_s: float) -> Path:
    """Retain a physical-time checkpoint that a later failure cannot overwrite."""

    archive = (
        checkpoint.parent
        / "stable"
        / f"{checkpoint.stem}_t{time_s:010.3f}.npz"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, archive)
    return archive


def run_variable_history_simulation(
    config: SimulationConfig,
    *,
    resume: bool = True,
    maximum_steps: int | None = None,
) -> Path:
    config.validate()
    simulation_directory = config.output_directory / "simulations"
    simulation_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_directory / "checkpoints" / "variable_history.npz"
    diagnostics_path = simulation_directory / "pdm7_diagnostics.jsonl"
    log_path = simulation_directory / "pdm7.log"
    logger = logging.getLogger(f"esf_sim.variable.{config.output_dir}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    if resume and checkpoint.exists():
        states, times, dt_s, step = _load_checkpoint(checkpoint)
        logger.info("Resumed at t=%g s, dt=%g s", times[0], dt_s)
    else:
        states = [initial_density(config)]
        times = [0.0]
        dt_s = config.dt_s
        step = 0
        diagnostics_path.unlink(missing_ok=True)
    model = _model_impl.ESFModel(config)
    mass_reference = float(
        np.sum(initial_density(config)) * config.dx_m * config.dy_m
    )
    low_cfl_steps = 0
    next_checkpoint_time = (
        math.floor(times[0] / config.checkpoint_interval_s) + 1
    ) * config.checkpoint_interval_s

    while times[0] < config.t_end_s - 1.0e-12:
        if maximum_steps is not None and step >= maximum_steps:
            break
        current_time = times[0]
        dt_step = min(dt_s, config.t_end_s - current_time)
        current = states[0]
        try:
            if len(states) < 3:
                updated, diagnostics = model.ssprk3_step(current, dt_step)
            else:
                predictor = variable_step_predictor(
                    states, times, current_time + 0.5 * dt_step
                )
                rhs, diagnostics = model.evaluate_rhs(predictor)
                _model_impl._check_cfl(
                    diagnostics.cfl_rate, dt_step, config.cfl_limit
                )
                updated = current + dt_step * rhs
        except _model_impl.CFLViolation as error:
            dt_s *= 0.5
            low_cfl_steps = 0
            logger.warning("%s; reducing dt to %g without discarding history", error, dt_s)
            _save_checkpoint(checkpoint, states, times, dt_s=dt_s, step=step)
            continue
        except RuntimeError:
            _save_checkpoint(checkpoint, states, times, dt_s=dt_s, step=step)
            logger.exception("Potential solve failed; checkpoint saved")
            raise

        minimum_density = float(np.min(updated))
        if minimum_density < -config.positivity_tolerance_m3:
            _save_checkpoint(checkpoint, states, times, dt_s=dt_s, step=step)
            raise FloatingPointError(
                f"negative density {minimum_density:.6e} m^-3 exceeds tolerance"
            )
        if minimum_density < 0.0:
            updated = np.maximum(updated, 0.0)
        new_time = current_time + dt_step
        states.insert(0, updated)
        times.insert(0, new_time)
        del states[3:]
        del times[3:]
        step += 1
        mass = float(np.sum(updated) * config.dx_m * config.dy_m)
        cfl = diagnostics.cfl_rate * dt_step
        record = {
            "step": step,
            "time_s": new_time,
            "dt_s": dt_step,
            "cfl": cfl,
            "maximum_speed_x_m_s": diagnostics.maximum_speed_x,
            "maximum_speed_y_m_s": diagnostics.maximum_speed_y,
            "potential_iterations": diagnostics.potential.iterations,
            "potential_relative_residual": diagnostics.potential.relative_residual,
            "mass_relative_change": mass / mass_reference - 1.0,
            "minimum_density_m3": minimum_density,
        }
        with diagnostics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

        if cfl < 0.08 and dt_s < config.dt_s:
            low_cfl_steps += 1
        else:
            low_cfl_steps = 0
        if low_cfl_steps >= 50:
            dt_s = min(2.0 * dt_s, config.dt_s)
            low_cfl_steps = 0
            logger.info("Increasing dt to %g while retaining variable history", dt_s)

        if new_time >= next_checkpoint_time or math.isclose(
            new_time, config.t_end_s
        ):
            _save_checkpoint(checkpoint, states, times, dt_s=dt_s, step=step)
            archive = _archive_stable_checkpoint(checkpoint, new_time)
            while next_checkpoint_time <= new_time:
                next_checkpoint_time += config.checkpoint_interval_s
            logger.info(
                "step=%d t=%g dt=%g CFL=%g residual=%g mass_change=%g "
                "stable_checkpoint=%s",
                step,
                new_time,
                dt_step,
                cfl,
                record["potential_relative_residual"],
                record["mass_relative_change"],
                archive.name,
            )

    snapshot = simulation_directory / "density_pdm7.npz"
    np.savez_compressed(
        snapshot,
        density=states[0],
        x_km=config.x_coordinates_m / 1.0e3 - 125.0,
        y_km=config.y_coordinates_m / 1.0e3,
        time_s=times[0],
        dt_s=dt_s,
        scheme="pdm7",
        potential_operator=config.potential_operator,
        potential_boundary_y=config.potential_boundary_y,
        time_integration="variable_step_lagrange_half_step_ab3",
        initial_mass=mass_reference,
        final_mass=float(np.sum(states[0]) * config.dx_m * config.dy_m),
    )
    return snapshot


def _case(alpha: float):
    for case in FIGURE3_CASES:
        if np.isclose(case.alpha, alpha):
            return case
    raise ValueError(alpha)


def run_case(base: SimulationConfig, alpha: float, *, resume: bool = True) -> Path:
    case = _case(alpha)
    root = base.output_directory / "figure3_stages" / STAGE_NAME
    stable = root / "simulations" / f"density_{case.key}.npz"
    if stable.exists():
        with np.load(stable) as saved:
            if np.isclose(float(saved["time_s"]), case.time_s):
                return stable
    run_directory = root / "runs" / case.key
    config = base.with_updates(
        scheme="pdm7",
        pdm_alpha=case.alpha,
        t_end_s=case.time_s,
        output_dir=str(run_directory),
    )
    archive = config.to_dict()
    archive["experimental_stage"] = STAGE_DESCRIPTION
    archive_path = root / "configs" / f"{case.key}.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(json.dumps(archive, indent=2), encoding="utf-8")
    print(
        f"[experiment] {STAGE_NAME} alpha={case.alpha:g} to t={case.time_s:g} s",
        flush=True,
    )
    raw = run_variable_history_simulation(config, resume=resume)
    with np.load(raw) as saved:
        stable.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            stable,
            density=saved["density"],
            x_km=saved["x_km"],
            y_km=saved["y_km"],
            time_s=float(saved["time_s"]),
            pdm_alpha=case.alpha,
            experimental_stage=STAGE_NAME,
        )
    return stable


def render(base: SimulationConfig) -> dict[str, object]:
    root = base.output_directory / "figure3_stages" / STAGE_NAME
    snapshots = {
        case.key: root / "simulations" / f"density_{case.key}.npz"
        for case in FIGURE3_CASES
    }
    paths = create_figure3(snapshots, root)
    renamed = {}
    for extension, path in paths.items():
        destination = root / f"figure3_{STAGE_NAME}.{extension}"
        path.replace(destination)
        renamed[extension] = destination
    write_figure3_source_data(
        snapshots, root / f"figure3_{STAGE_NAME}_source_data.npz"
    )
    report = write_figure3_metrics(
        snapshots,
        reference_directory=Path(base.reference_mat).parent,
        output_path=root / f"figure3_{STAGE_NAME}_metrics.json",
    )
    shutil.copyfile(
        root / f"figure3_{STAGE_NAME}_metrics.json",
        root / f"figure3_{STAGE_NAME}_acceptance.json",
    )
    return {
        "figures": {key: str(value) for key, value in renamed.items()},
        "reference_thresholds_pass": report["all_reference_thresholds_pass"],
        "alpha_trend_pass": report[
            "all_scientific_alpha_trend_criteria_pass"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.experimental_time")
    parser.add_argument("--config", required=True)
    parser.add_argument("--alpha", type=float, action="append")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig.from_json(args.config)
    for alpha in args.alpha or [0.6, 0.2, 0.4, 0.8]:
        run_case(base, alpha, resume=not args.no_resume)
    if args.render:
        print(json.dumps(render(base), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
