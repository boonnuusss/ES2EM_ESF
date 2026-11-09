"""Numerical verification tests exposed by the validate command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.sparse.linalg import spsolve

from .numerics import advection_rhs
from .potential import build_elliptic_matrix


def _ssprk3(
    state: np.ndarray,
    dt: float,
    rhs: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    first = state + dt * rhs(state)
    second = 0.75 * state + 0.25 * (first + dt * rhs(first))
    return (state + 2.0 * (second + dt * rhs(second))) / 3.0


def slotted_cylinder_initial(ncells: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the modified Zalesak slotted cylinder used in the paper."""

    coordinates = np.linspace(0.0, 1.0, ncells)
    x, y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(x - 0.5, y - 0.75)
    inside_cylinder = radius <= 0.15
    slot = (np.abs(x - 0.5) <= 0.03) & (y <= 0.85) & (y >= 0.75)
    density = np.full_like(x, 1.0e-10)
    density[inside_cylinder & ~slot] = 1.0
    return density, x, y


def _transition_width(profile: np.ndarray, coordinate: np.ndarray) -> int:
    candidates = np.where((profile > 0.1) & (profile < 0.9) & (coordinate < 0.48))[0]
    if not candidates.size:
        return 0
    groups = np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1)
    return int(max(group.size for group in groups))


def circular_advection_test(
    output_directory: str | Path,
    *,
    ncells: int = 256,
    t_end: float = 1.0,
    cfl: float = 0.25,
) -> dict[str, object]:
    """Rotate the slotted cylinder clockwise through one full revolution."""

    initial, x, y = slotted_cylinder_initial(ncells)
    dx = 1.0 / (ncells - 1)
    omega = 2.0 * np.pi
    velocity_x = omega * (y - 0.5)
    velocity_y = -omega * (x - 0.5)
    dt = cfl / float(
        np.max(np.abs(velocity_x) / dx + np.abs(velocity_y) / dx)
    )
    nsteps = int(np.ceil(t_end / dt))
    dt = t_end / nsteps
    results: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {"initial": initial, "x": x, "y": y}

    for scheme in ("first_order", "van_leer", "pdm7"):
        def rhs(value: np.ndarray) -> np.ndarray:
            return advection_rhs(
                value,
                velocity_x,
                velocity_y,
                dx=dx,
                dy=dx,
                scheme=scheme,
                pdm_alpha=0.2,
                periodic_y=True,
            )

        history = [initial.copy()]
        for _ in range(nsteps):
            if len(history) < 3:
                updated = _ssprk3(history[0], dt, rhs)
            else:
                predictor = (
                    15.0 / 8.0 * history[0]
                    - 5.0 / 4.0 * history[1]
                    + 3.0 / 8.0 * history[2]
                )
                updated = history[0] + dt * rhs(predictor)
            history.insert(0, updated)
            del history[3:]
        final = history[0]
        profile_row = int(np.argmin(np.abs(y[:, 0] - 0.82)))
        width = _transition_width(final[profile_row], x[profile_row])
        results[scheme] = {
            "minimum": float(np.min(final)),
            "maximum": float(np.max(final)),
            "mass_relative_change": float(np.sum(final) / np.sum(initial) - 1.0),
            "discontinuity_width_cells": width,
            "nonnegative": bool(np.min(final) >= -1.0e-12),
        }
        arrays[scheme] = final

    widths = [
        results[scheme]["discontinuity_width_cells"]
        for scheme in ("first_order", "van_leer", "pdm7")
    ]
    results["criteria"] = {
        "all_tvd_schemes_nonnegative": bool(
            all(
                results[scheme]["nonnegative"]
                for scheme in ("first_order", "van_leer", "pdm7")
            )
        ),
        "width_ordering_first_gt_van_leer_gt_pdm7": bool(
            widths[0] > widths[1] > widths[2]
        ),
        "van_leer_near_22_cells": bool(12 <= widths[1] <= 32),
        "pdm7_near_5_cells": bool(2 <= widths[2] <= 10),
    }
    results["grid"] = {"ncells": ncells, "nsteps": nsteps, "dt": dt, "t_end": t_end}
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "circular_advection_validation.npz", **arrays)
    return results


def elliptic_manufactured_solution_test() -> dict[str, object]:
    """Verify second-order convergence of the conservative five-point operator."""

    errors = []
    grid_sizes = (24, 48, 96)
    for nx in grid_sizes:
        ny = nx + 1
        dx = 1.0 / nx
        dy = 1.0 / (ny - 1)
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        xx, yy = np.meshgrid(x, y)
        exact = np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy)
        rhs_full = 5.0 * np.pi**2 * exact
        matrix, scale = build_elliptic_matrix(np.ones_like(exact), dx, dy)
        solution = spsolve(matrix, (rhs_full[1:-1] / scale).ravel())
        numerical = np.zeros_like(exact)
        numerical[1:-1] = solution.reshape(ny - 2, nx)
        error = float(np.sqrt(np.mean((numerical - exact) ** 2)))
        errors.append(error)
    rates = [
        float(np.log(errors[index] / errors[index + 1]) / np.log(2.0))
        for index in range(len(errors) - 1)
    ]
    return {
        "grid_sizes": list(grid_sizes),
        "l2_errors": errors,
        "observed_orders": rates,
        "second_order_pass": bool(min(rates) >= 1.85),
    }


def temporal_accuracy_test() -> dict[str, object]:
    """Verify third-order convergence of SSPRK3-started AB3 predictor-corrector."""

    errors = []
    time_steps = (0.1, 0.05, 0.025, 0.0125)
    for dt in time_steps:
        history = [np.asarray([1.0])]
        nsteps = int(round(1.0 / dt))
        rhs = lambda value: -value
        for _ in range(nsteps):
            if len(history) < 3:
                updated = _ssprk3(history[0], dt, rhs)
            else:
                predictor = (
                    15.0 / 8.0 * history[0]
                    - 5.0 / 4.0 * history[1]
                    + 3.0 / 8.0 * history[2]
                )
                updated = history[0] + dt * rhs(predictor)
            history.insert(0, updated)
            del history[3:]
        errors.append(abs(float(history[0][0]) - float(np.exp(-1.0))))
    rates = [
        float(np.log(errors[index] / errors[index + 1]) / np.log(2.0))
        for index in range(len(errors) - 1)
    ]
    return {
        "time_steps": list(time_steps),
        "absolute_errors": errors,
        "observed_orders": rates,
        "third_order_pass": bool(min(rates[-2:]) >= 2.7),
    }


def run_validation_suite(output_directory: str | Path) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "circular_advection": circular_advection_test(output),
        "elliptic_manufactured_solution": elliptic_manufactured_solution_test(),
        "temporal_accuracy": temporal_accuracy_test(),
    }
    report["all_pass"] = bool(
        all(report["circular_advection"]["criteria"].values())
        and report["elliptic_manufactured_solution"]["second_order_pass"]
        and report["temporal_accuracy"]["third_order_pass"]
    )
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
