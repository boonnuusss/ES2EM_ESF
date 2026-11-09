"""Reproducible late-EPB performance/stability benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ._model_impl import ESFModel
from .config import SimulationConfig
from .optimized_numerics_v2 import advection_rhs as optimized_advection
from .optimized_potential import (
    CachedExponentialPotentialSolver, CachedPointCenteredPotentialSolver,
)
from .stabilized_potential import point_centered_peclet_diagnostics


def _median_time(callable_, repeats):
    callable_()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        values.append(time.perf_counter()-started)
    return float(np.median(values))


def benchmark(config, checkpoint, repeats=10):
    with np.load(checkpoint) as saved:
        density = np.asarray(saved["density"], dtype=np.float64)
        state_time = float(saved["time_s"])
        old_dt = float(saved["dt_s"])
    coefficient = np.maximum(density, config.density_floor_m3)*(
        config.collision_nu0_s*np.exp(
            -config.y_coordinates_m/config.collision_frequency_scale_height_m
        )
    )[:, None]
    peclet = point_centered_peclet_diagnostics(
        coefficient, config.dx_m, config.dy_m
    )
    records = {}
    potentials = {}
    for name, operator, solver_type in (
        ("legacy_point_centered", "point_centered_equation_2_21_log_coefficients", None),
        ("cached_point_centered", None, CachedPointCenteredPotentialSolver),
        ("legacy_exponential", "exponential_conservative", None),
        ("cached_exponential", None, CachedExponentialPotentialSolver),
    ):
        if solver_type is None:
            model = ESFModel(config.with_updates(potential_operator=operator))
            solver = model.potential_solver
        else:
            solver = solver_type(config)
        potential, diagnostics = solver.solve(density)
        potentials[name] = potential
        elapsed = _median_time(lambda: solver.solve(density), repeats)
        padded = np.pad(potential, ((1, 1), (0, 0)), mode="edge")
        u = -(padded[2:]-padded[:-2])/(2*config.dy_m)
        v = (np.roll(potential,-1,axis=1)-np.roll(potential,1,axis=1))/(2*config.dx_m)
        rate = float(np.max(np.abs(u)/config.dx_m+np.abs(v)/config.dy_m))
        records[name] = {
            "potential_solve_median_s": elapsed,
            "relative_residual": diagnostics.relative_residual,
            "maximum_speed_m_s": float(max(np.max(np.abs(u)), np.max(np.abs(v)))),
            "cfl_rate_s-1": rate,
            "safe_dt_at_configured_cfl_s": config.cfl_limit/rate,
        }
    exp = potentials["legacy_exponential"]
    records["cached_exponential"]["relative_potential_error_vs_legacy"] = float(
        np.linalg.norm(potentials["cached_exponential"]-exp)/np.linalg.norm(exp)
    )
    potential = potentials["cached_exponential"]
    padded = np.pad(potential, ((1, 1), (0, 0)), mode="edge")
    u = -(padded[2:]-padded[:-2])/(2*config.dy_m)
    v = (np.roll(potential,-1,axis=1)-np.roll(potential,1,axis=1))/(2*config.dx_m)
    from .numerics import advection_rhs as reference_advection
    kwargs = dict(dx=config.dx_m, dy=config.dy_m, scheme=config.scheme,
                  pdm_alpha=config.pdm_alpha, periodic_y=False)
    reference = reference_advection(density, u, v, **kwargs)
    optimized = optimized_advection(density, u, v, **kwargs)
    old_flux_time = _median_time(lambda: reference_advection(density,u,v,**kwargs), repeats)
    new_flux_time = _median_time(lambda: optimized_advection(density,u,v,**kwargs), repeats)
    report = {
        "checkpoint": str(checkpoint), "state_time_s": state_time,
        "legacy_checkpoint_dt_s": old_dt, "grid": list(density.shape),
        "point_centered_peclet": peclet, "potential_solvers": records,
        "advection": {
            "legacy_median_s": old_flux_time, "optimized_median_s": new_flux_time,
            "speedup": old_flux_time/new_flux_time,
            "relative_difference": float(np.linalg.norm(reference-optimized)/np.linalg.norm(reference)),
        },
    }
    old = records["legacy_point_centered"]
    new = records["cached_exponential"]
    report["projected_rhs_speedup"] = (
        old["potential_solve_median_s"]+old_flux_time
    )/(new["potential_solve_median_s"]+new_flux_time)
    report["projected_physical_time_speedup"] = report["projected_rhs_speedup"]*(
        new["safe_dt_at_configured_cfl_s"]/old["safe_dt_at_configured_cfl_s"]
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    report = benchmark(SimulationConfig.from_json(args.config), Path(args.checkpoint), args.repeats)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
