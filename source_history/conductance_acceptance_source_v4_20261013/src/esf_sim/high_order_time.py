"""True non-uniform third-order Adams-Bashforth time integration."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from ._model_impl import CFLViolation, _check_cfl
from .config import SimulationConfig
from .production import (
    ProductionESFModel, ProductionRunSummary, _load_history, _save,
)


def variable_ab3_weights(rhs_times, target_time):
    """Integrate the quadratic Lagrange basis from newest time to target."""
    if len(rhs_times) < 3:
        raise ValueError("three RHS times are required")
    t0 = float(rhs_times[0])
    scale = max(abs(target_time-t0), *(abs(float(t)-t0) for t in rhs_times[1:]))
    nodes = np.asarray([(float(t)-t0)/scale for t in rhs_times[:3]])
    upper = (float(target_time)-t0)/scale
    weights = []
    for i in range(3):
        polynomial = np.poly1d([1.0])
        denominator = 1.0
        for j in range(3):
            if i != j:
                polynomial *= np.poly1d([1.0, -nodes[j]])
                denominator *= nodes[i]-nodes[j]
        integral = np.polyint(polynomial/denominator)
        weights.append(scale*float(integral(upper)-integral(0.0)))
    return np.asarray(weights)


def _startup_step(model, density, dt, cfl_limit):
    rhs0, d0 = model.evaluate_rhs(density)
    _check_cfl(d0.cfl_rate, dt, cfl_limit)
    u1 = density+dt*rhs0
    rhs1, d1 = model.evaluate_rhs(u1)
    _check_cfl(d1.cfl_rate, dt, cfl_limit)
    u2 = 0.75*density+0.25*(u1+dt*rhs1)
    rhs2, d2 = model.evaluate_rhs(u2)
    _check_cfl(d2.cfl_rate, dt, cfl_limit)
    result = (density+2*(u2+dt*rhs2))/3
    return result, max((d0,d1,d2), key=lambda d:d.cfl_rate), rhs0, d0


def advance_checkpoint_ab3(
    config: SimulationConfig, source, destination, *, duration_s,
    maximum_dt_s=None, minimum_dt_s=1e-4, switch_peclet=0.75,
    cfl_safety=0.85,
):
    """Advance with one elliptic solve per accepted AB3 step after startup."""
    states, times, stored_dt = _load_history(Path(source))
    # Stored density histories do not carry RHS values, so only the current
    # state is trusted; two SSPRK3 steps seed third-order RHS history.
    states, times = states[:1], times[:1]
    start = times[0]
    target = start+duration_s
    max_dt = config.dt_s if maximum_dt_s is None else maximum_dt_s
    dt = min(max_dt, max(stored_dt, 0.25*max_dt))
    model = ProductionESFModel(config, switch_peclet=switch_peclet)
    rhs_history, rhs_times, rhs_diags = [], [], []
    mass0 = float(np.sum(states[0]))
    accepted = rejected = evaluations = 0
    min_used, max_used, final_cfl = math.inf, 0.0, 0.0
    wall_start = time.perf_counter()

    while times[0] < target-1e-12:
        step_dt = min(dt, target-times[0])
        current = states[0]
        try:
            if len(rhs_history) < 2:
                updated, diag, rhs_current, current_diag = _startup_step(
                    model, current, step_dt, config.cfl_limit
                )
                evaluations += 3
            else:
                if not rhs_times or rhs_times[0] != times[0]:
                    rhs_current, current_diag = model.evaluate_rhs(current)
                    evaluations += 1
                    rhs_history.insert(0, rhs_current)
                    rhs_times.insert(0, times[0])
                    rhs_diags.insert(0, current_diag)
                    del rhs_history[3:]; del rhs_times[3:]; del rhs_diags[3:]
                _check_cfl(current_diag.cfl_rate, step_dt, config.cfl_limit)
                weights = variable_ab3_weights(rhs_times, times[0]+step_dt)
                updated = current+sum(w*f for w,f in zip(weights, rhs_history))
                diag = max(rhs_diags, key=lambda d:d.cfl_rate)
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

        if len(rhs_history) < 2:
            rhs_history.insert(0, rhs_current)
            rhs_times.insert(0, times[0])
            rhs_diags.insert(0, current_diag)
        updated = np.maximum(updated, 0.0)
        new_time = times[0]+step_dt
        states.insert(0, updated); times.insert(0, new_time)
        del states[3:]; del times[3:]
        accepted += 1
        min_used, max_used = min(min_used,step_dt), max(max_used,step_dt)
        final_cfl = diag.cfl_rate*step_dt
        controlling_rate = max(d.cfl_rate for d in rhs_diags)
        dt = min(max_dt, 1.25*step_dt,
                 cfl_safety*config.cfl_limit/max(controlling_rate,np.finfo(float).tiny))
        dt = max(dt, minimum_dt_s)

    adaptive = model.potential_solver.last_adaptive_diagnostics
    summary = ProductionRunSummary(
        start, times[0], accepted, rejected, evaluations,
        time.perf_counter()-wall_start, min_used, max_used, dt, final_cfl,
        adaptive.mode, adaptive.maximum_peclet,
        float(np.sum(states[0])/mass0-1), float(np.min(states[0])),
    )
    _save(Path(destination), states, times, dt, summary)
    report = Path(destination).with_suffix(".json")
    report.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary


__all__ = ["advance_checkpoint_ab3", "variable_ab3_weights"]
