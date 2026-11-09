"""Restart-safe alpha=0.4 production run from Chapman state to 5011 s."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from ._model_impl import CFLViolation, _check_cfl, initial_density
from .config import SimulationConfig
from .high_order_time import _startup_step, variable_ab3_weights
from .optimized_potential import CachedExponentialPotentialSolver
from .production import ProductionESFModel, ProductionRunSummary
from .timeseries import ChunkedFrameArchive


DEFAULT_OUTPUT = Path("output/optimization/alpha04_to_5011")


class _ArchiveConfig:
    """Expose honest optimized-operator metadata to the archive writer."""
    def __init__(self, config):
        self._config = config
    def __getattr__(self, name):
        return getattr(self._config, name)
    @property
    def potential_operator(self):
        return "adaptive_point_centered_to_exponential_conservative"


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path, *, states, times, rhs, rhs_times, dt, step,
                     mode, switch_time, low_cfl_steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, states=np.stack(states), times=np.asarray(times),
        rhs=np.stack(rhs) if rhs else np.empty((0, *states[0].shape)),
        rhs_times=np.asarray(rhs_times), dt_s=dt, step=step,
        potential_mode=mode,
        switch_time_s=np.nan if switch_time is None else switch_time,
        low_cfl_steps=low_cfl_steps,
    )
    temporary.replace(path)


def _load_checkpoint(path):
    with np.load(path) as z:
        switch = float(z["switch_time_s"])
        return (
            [x.copy() for x in z["states"]], [float(x) for x in z["times"]],
            [x.copy() for x in z["rhs"]], [float(x) for x in z["rhs_times"]],
            float(z["dt_s"]), int(z["step"]), str(z["potential_mode"]),
            None if np.isnan(switch) else switch, int(z["low_cfl_steps"]),
        )


def run(config_path="configs/figure3.json", output=DEFAULT_OUTPUT, *, resume=True):
    base = SimulationConfig.from_json(config_path)
    config = base.with_updates(
        pdm_alpha=0.4, t_end_s=5011.0, output_dir=str(output)
    )
    output = Path(output)
    checkpoint = output/"checkpoints"/"optimized_ab3.npz"
    diagnostics_path = output/"simulations"/"optimized_diagnostics.jsonl"
    status_path = output/"status.json"
    frame_directory = output/"time_series"
    output.joinpath("simulations").mkdir(parents=True, exist_ok=True)

    if resume and checkpoint.exists():
        (states,times,rhs_history,rhs_times,dt,step,mode,switch_time,
         low_cfl_steps) = _load_checkpoint(checkpoint)
    else:
        states, times = [initial_density(config)], [0.0]
        rhs_history, rhs_times = [], []
        dt, step, mode, switch_time, low_cfl_steps = config.dt_s, 0, "point_centered", None, 0
        diagnostics_path.unlink(missing_ok=True)

    archive = ChunkedFrameArchive(
        frame_directory, _ArchiveConfig(config), interval_s=1.0,
        chunk_frames=100, storage_dtype=np.float32,
    )
    archive.prepare(resume_time_s=times[0], density=states[0], resume=resume)
    model = ProductionESFModel(config, switch_peclet=0.75)
    if mode == "exponential_conservative":
        model.potential_solver.mode = mode
        model.potential_solver.exponential = CachedExponentialPotentialSolver(config)

    start_wall = time.perf_counter()
    start_time = times[0]
    mass_reference = float(np.sum(initial_density(config)))
    accepted = rejected = evaluations = 0
    minimum_dt, maximum_dt, final_cfl = math.inf, 0.0, 0.0
    next_checkpoint = (math.floor(times[0]/100)+1)*100
    last_logged_second = int(math.floor(times[0]))

    _atomic_json(status_path, {
        "state":"running", "time_s":times[0], "dt_s":dt,
        "potential_mode":model.potential_solver.mode,
        "operator_switch_time_s":switch_time,
    })
    while times[0] < config.t_end_s-1e-12:
        step_dt = min(dt, config.t_end_s-times[0])
        current = states[0]
        previous_mode = model.potential_solver.mode
        try:
            if len(rhs_history) < 2:
                updated, diag, rhs_current, current_diag = _startup_step(
                    model, current, step_dt, config.cfl_limit
                )
                evaluations += 3
            else:
                if not rhs_times or not math.isclose(rhs_times[0],times[0],abs_tol=1e-12):
                    rhs_current, current_diag = model.evaluate_rhs(current)
                    evaluations += 1
                    rhs_history.insert(0,rhs_current); rhs_times.insert(0,times[0])
                    del rhs_history[3:]; del rhs_times[3:]
                _check_cfl(current_diag.cfl_rate, step_dt, config.cfl_limit)
                weights = variable_ab3_weights(rhs_times, times[0]+step_dt)
                updated = current+sum(w*f for w,f in zip(weights,rhs_history))
                diag = current_diag
            minimum = float(np.min(updated))
            if not np.all(np.isfinite(updated)) or minimum < -config.positivity_tolerance_m3:
                raise FloatingPointError(f"trial minimum {minimum:.6e}")
        except (CFLViolation, FloatingPointError):
            rejected += 1
            dt *= 0.5
            low_cfl_steps = 0
            if dt < 1/1024:
                _save_checkpoint(checkpoint,states=states,times=times,rhs=rhs_history,
                    rhs_times=rhs_times,dt=dt,step=step,mode=model.potential_solver.mode,
                    switch_time=switch_time,low_cfl_steps=low_cfl_steps)
                archive.finalize(completed=False)
                _atomic_json(status_path,{"state":"paused_minimum_dt","time_s":times[0],"dt_s":dt})
                raise RuntimeError("optimized run reached minimum dt=1/1024 s")
            continue

        if previous_mode != model.potential_solver.mode and switch_time is None:
            switch_time = times[0]
        if len(rhs_history) < 2:
            rhs_history.insert(0,rhs_current); rhs_times.insert(0,times[0])
        updated = np.maximum(updated,0.0)
        new_time = times[0]+step_dt
        states.insert(0,updated); times.insert(0,new_time)
        del states[3:]; del times[3:]
        accepted += 1; step += 1
        minimum_dt=min(minimum_dt,step_dt); maximum_dt=max(maximum_dt,step_dt)
        final_cfl=diag.cfl_rate*step_dt

        if math.isclose(new_time,round(new_time),abs_tol=1e-9):
            archive.capture(new_time,updated)
        if final_cfl < 0.08 and dt < config.dt_s:
            low_cfl_steps += 1
        else:
            low_cfl_steps = 0
        if low_cfl_steps >= 40 and 2*dt <= config.dt_s:
            dt *= 2; low_cfl_steps = 0

        current_second = int(math.floor(new_time+1e-9))
        if current_second > last_logged_second:
            adaptive = model.potential_solver.last_adaptive_diagnostics
            record = {
                "time_s":new_time,"dt_s":step_dt,"cfl":final_cfl,
                "potential_relative_residual":diag.potential.relative_residual,
                "maximum_speed_m_s":max(diag.maximum_speed_x,diag.maximum_speed_y),
                "minimum_density_m3":minimum,
                "mass_relative_change":float(np.sum(updated)/mass_reference-1),
                "potential_mode":adaptive.mode,"maximum_peclet":adaptive.maximum_peclet,
                "frame_count":archive.frame_count,
            }
            with diagnostics_path.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record)+"\n")
            last_logged_second=current_second
        if new_time >= next_checkpoint-1e-9 or math.isclose(new_time,config.t_end_s,abs_tol=1e-9):
            archive.flush()
            _save_checkpoint(checkpoint,states=states,times=times,rhs=rhs_history,
                rhs_times=rhs_times,dt=dt,step=step,mode=model.potential_solver.mode,
                switch_time=switch_time,low_cfl_steps=low_cfl_steps)
            _atomic_json(status_path,{
                "state":"running","time_s":new_time,"dt_s":dt,
                "cfl":final_cfl,"potential_mode":model.potential_solver.mode,
                "operator_switch_time_s":switch_time,"frame_count":archive.frame_count,
            })
            while next_checkpoint <= new_time+1e-9: next_checkpoint += 100

    archive_report=archive.finalize(completed=True)
    adaptive=model.potential_solver.last_adaptive_diagnostics
    summary=ProductionRunSummary(
        start_time,times[0],accepted,rejected,evaluations,time.perf_counter()-start_wall,
        minimum_dt,maximum_dt,dt,final_cfl,adaptive.mode,adaptive.maximum_peclet,
        float(np.sum(states[0])/mass_reference-1),float(np.min(states[0])),
    )
    snapshot=output/"simulations"/"density_optimized_alpha_0p4_t5011.npz"
    np.savez_compressed(snapshot,density=states[0],time_s=times[0],pdm_alpha=.4,
        potential_operator="adaptive_point_centered_to_exponential_conservative",
        potential_boundary_y="equal_extrapolation",operator_switch_time_s=switch_time,
        time_integration="variable_step_true_ab3",x_km=config.x_coordinates_m/1e3-125,
        y_km=config.y_coordinates_m/1e3)
    report={"summary":asdict(summary),"operator_switch_time_s":switch_time,
            "archive":archive_report,"snapshot":str(snapshot.resolve())}
    _atomic_json(output/"optimized_run_report.json",report)
    _atomic_json(status_path,{"state":"completed","time_s":times[0],"dt_s":dt,
        "potential_mode":adaptive.mode,"operator_switch_time_s":switch_time,
        "frame_count":archive.frame_count})
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/figure3.json")
    parser.add_argument("--output",default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.config,args.output,resume=not args.no_resume),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
