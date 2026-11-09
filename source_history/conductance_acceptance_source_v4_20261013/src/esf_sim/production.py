"""High-stability, high-throughput ESF production integrator.

This module deliberately coexists with the historical reproduction driver.
It combines cached sparse operators, a PDM7 kernel with fewer temporaries, an
automatic centred-to-monotone potential switch, and non-uniform AB3 history so
that a CFL adjustment does not discard two expensive time levels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from ._model_impl import CFLViolation, ESFModel, RhsDiagnostics, _check_cfl
from .config import SimulationConfig
from .optimized_numerics_v2 import advection_rhs
from .optimized_potential import AdaptiveMonotonePotentialSolver


def variable_step_predictor(states, times, target_time):
    """Quadratic state extrapolation to a requested midpoint."""
    if len(states) < 3 or len(times) < 3:
        raise ValueError("three timestamped states are required")
    result = np.zeros_like(states[0])
    for i in range(3):
        weight = 1.0
        for j in range(3):
            if i != j:
                weight *= (target_time-times[j])/(times[i]-times[j])
        result += weight*states[i]
    return result


class ProductionESFModel(ESFModel):
    def __init__(self, config: SimulationConfig, *, switch_peclet: float = 0.75):
        # Avoid constructing the legacy solver which is immediately replaced.
        self.config = config
        self.x = config.x_coordinates_m
        self.y = config.y_coordinates_m
        self.collision_frequency = config.collision_nu0_s*np.exp(
            -self.y/config.collision_frequency_scale_height_m
        )
        self.potential_solver = AdaptiveMonotonePotentialSolver(config, switch_peclet)

    def evaluate_rhs(self, density):
        potential, potential_diagnostics = self.potential_solver.solve(density)
        velocity_x, velocity_y = self.velocity_from_potential(potential)
        rhs = advection_rhs(
            density, velocity_x, velocity_y,
            dx=self.config.dx_m, dy=self.config.dy_m,
            scheme=self.config.scheme, pdm_alpha=self.config.pdm_alpha,
            periodic_y=False,
        )
        rate = float(np.max(
            np.abs(velocity_x)/self.config.dx_m
            + np.abs(velocity_y)/self.config.dy_m
        ))
        return rhs, RhsDiagnostics(
            rate, float(np.max(np.abs(velocity_x))),
            float(np.max(np.abs(velocity_y))), potential_diagnostics,
        )


@dataclass
class ProductionRunSummary:
    start_time_s: float
    final_time_s: float
    accepted_steps: int
    rejected_steps: int
    rhs_evaluations: int
    wall_time_s: float
    minimum_dt_s: float
    maximum_dt_s: float
    final_dt_s: float
    final_cfl: float
    potential_mode: str
    maximum_peclet: float
    mass_relative_change: float
    minimum_density_m3: float


def _load_history(path: Path):
    with np.load(path) as saved:
        if "states" in saved.files:
            states = [x.copy() for x in saved["states"]]
            times = [float(x) for x in saved["times"]]
            dt = float(saved["dt_s"])
        else:
            states = [saved["density"].copy()]
            length = int(saved["history_length"]) if "history_length" in saved.files else 1
            dt = float(saved["dt_s"])
            time_s = float(saved["time_s"])
            if length > 1 and saved["density_previous"].size:
                states.append(saved["density_previous"].copy())
            if length > 2 and saved["density_previous2"].size:
                states.append(saved["density_previous2"].copy())
            times = [time_s-i*dt for i in range(len(states))]
    return states, times, dt


def _save(path: Path, states, times, dt, summary=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    payload = dict(states=np.stack(states), times=np.asarray(times), dt_s=dt)
    if summary is not None:
        payload["summary_json"] = json.dumps(asdict(summary))
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def advance_checkpoint(
    config: SimulationConfig,
    source: str | Path,
    destination: str | Path,
    *,
    duration_s: float,
    maximum_dt_s: float | None = None,
    minimum_dt_s: float = 1.0e-4,
    switch_peclet: float = 0.75,
    cfl_safety: float = 0.90,
) -> ProductionRunSummary:
    """Advance an existing state with adaptive non-uniform history."""
    states, times, stored_dt = _load_history(Path(source))
    start_time = times[0]
    target = start_time+duration_s
    maximum_dt = config.dt_s if maximum_dt_s is None else maximum_dt_s
    # A previously collapsed legacy dt is not a useful initial estimate after
    # switching to the monotone operator. Start optimistically and reject once.
    dt = min(maximum_dt, max(stored_dt, 0.25*maximum_dt))
    model = ProductionESFModel(config, switch_peclet=switch_peclet)
    mass0 = float(np.sum(states[0]))
    accepted = rejected = evaluations = 0
    minimum_used = math.inf
    maximum_used = 0.0
    final_cfl = 0.0
    started = time.perf_counter()

    while times[0] < target-1.0e-12:
        step_dt = min(dt, target-times[0])
        current = states[0]
        try:
            if len(states) < 3:
                # SSPRK3 start-up is retained only when a source checkpoint
                # genuinely lacks history; subsequent dt changes retain it.
                updated, diagnostics = model.ssprk3_step(current, step_dt)
                evaluations += 3
            else:
                predictor = variable_step_predictor(
                    states, times, times[0]+0.5*step_dt
                )
                rhs, diagnostics = model.evaluate_rhs(predictor)
                evaluations += 1
                _check_cfl(diagnostics.cfl_rate, step_dt, config.cfl_limit)
                updated = current+step_dt*rhs
            minimum = float(np.min(updated))
            if not np.all(np.isfinite(updated)) or minimum < -config.positivity_tolerance_m3:
                raise FloatingPointError(f"non-positive trial state: {minimum:.6e}")
        except (CFLViolation, FloatingPointError):
            rejected += 1
            dt *= 0.5
            if dt < minimum_dt_s:
                _save(Path(destination), states, times, dt)
                raise RuntimeError(f"required dt={dt:g} s is below minimum {minimum_dt_s:g} s")
            continue

        updated = np.maximum(updated, 0.0)
        new_time = times[0]+step_dt
        states.insert(0, updated)
        times.insert(0, new_time)
        del states[3:]
        del times[3:]
        accepted += 1
        minimum_used = min(minimum_used, step_dt)
        maximum_used = max(maximum_used, step_dt)
        final_cfl = diagnostics.cfl_rate*step_dt

        cfl_dt = cfl_safety*config.cfl_limit/max(
            diagnostics.cfl_rate, np.finfo(float).tiny
        )
        dt = min(maximum_dt, 1.25*step_dt, cfl_dt)
        dt = max(dt, minimum_dt_s)

    adaptive = model.potential_solver.last_adaptive_diagnostics
    summary = ProductionRunSummary(
        start_time, times[0], accepted, rejected, evaluations,
        time.perf_counter()-started, minimum_used, maximum_used, dt, final_cfl,
        adaptive.mode, adaptive.maximum_peclet,
        float(np.sum(states[0])/mass0-1.0), float(np.min(states[0])),
    )
    _save(Path(destination), states, times, dt, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.production")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--max-dt", type=float)
    parser.add_argument("--min-dt", type=float, default=1.0e-4)
    parser.add_argument("--switch-peclet", type=float, default=0.75)
    args = parser.parse_args()
    config = SimulationConfig.from_json(args.config)
    summary = advance_checkpoint(
        config, args.checkpoint, args.output, duration_s=args.duration,
        maximum_dt_s=args.max_dt, minimum_dt_s=args.min_dt,
        switch_peclet=args.switch_peclet,
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

