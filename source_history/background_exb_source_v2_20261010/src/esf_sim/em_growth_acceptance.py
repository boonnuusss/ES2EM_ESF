"""Read-only scientific acceptance gates for reduced-EM ESF growth runs.

The module deliberately does not import :mod:`esf_sim.em_growth`: it only
reads the archived report, diagnostics and state arrays.  It is therefore safe
to run beside (or after) a production experiment without constructing a model
or changing a checkpoint.

The accepted production experiment is the three-track comparison

``local ES -> column ES -> column reduced-EM``.

The 60, 300 and 3150 s stages answer different questions.  The first is a
closure/energy sanity check, the second is an early-growth check, and the last
requires a resolved, growing depleted bubble.  The thresholds below are kept
in one public dictionary so that every JSON report archives the exact policy
used to grade the run.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


FORMAT = "esf_em_growth_acceptance_v2"
TRACKS = ("local_es", "column_es", "column_em")


THRESHOLDS: dict[str, Any] = {
    "density_cfl": {"warning": 0.24, "failure": 0.2500001},
    "coupled_cfl": {"warning": 0.38, "failure": 0.4000001},
    "relative_residual": {"warning": 1.0e-10, "failure": 1.0e-8},
    "power_pairing_defect": {"warning": 1.0e-10, "failure": 1.0e-8},
    "net_surface_current_ratio": {
        "warning": 1.0e-12,
        "failure": 1.0e-10,
    },
    "periodic_endpoint_relative_mismatch": {
        "warning": 1.0e-10,
        "failure": 1.0e-6,
    },
    "line_energy_budget": {
        # Upwind propagation is dissipative, so a positive work-minus-energy
        # remainder is expected.  Only net numerical energy creation is a
        # hard failure; excessive positive dissipation is reported as a
        # resolution warning.
        "maximum_creation_fraction": 5.0e-3,
        "dissipation_warning_fraction": 5.0e-2,
    },
    "fac_to_gravity_ratio": {
        "preferred": [0.05, 0.70],
        "failure": [0.02, 0.85],
    },
    "minimum_dt_s": {
        "60": {"warning": 0.25, "failure": 0.125},
        "300": {"warning": 0.125, "failure": 0.03125},
        "3150": {"warning": 0.0625, "failure": 0.015625},
    },
    "early_growth": {
        "late_window_s": 100.0,
        "minimum_column_em_rate_s_inv": -2.0e-4,
        "relative_l2_failure": 1.0e-4,
        "relative_l2_preferred": 1.0e-3,
    },
    "mature_growth": {
        "minimum_positive_window_s": 300.0,
        "minimum_rate_s_inv": 1.0e-4,
        "preferred_rate_s_inv": 2.0e-4,
        "minimum_final_to_minimum_amplitude": 2.0,
        "minimum_em_to_column_positive_rate_ratio": 0.3,
        "relative_l2_failure": 5.0e-3,
        "relative_l2_preferred": 1.0e-2,
    },
    "bubble": {
        "depletion_threshold": 0.25,
        "minimum_top_km": 430.0,
        "preferred_top_km": 450.0,
        "minimum_span_km": 80.0,
        "preferred_span_km": 100.0,
        "minimum_area_km2": 500.0,
        "minimum_maximum_depletion": 0.50,
        "minimum_two_cell_width_layers": 10,
    },
}


def _json_ready(value: Any) -> Any:
    """Convert numpy values and non-finite floats into strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        records.append(record)
    return records


def _configuration_hash_from_report(report: dict[str, Any]) -> str:
    """Recompute the trajectory hash from the archived configuration.

    This intentionally mirrors :func:`esf_sim.em_growth._configuration_hash`
    without importing the simulation runner.  Merely comparing the same hash
    string in the report, status and checkpoint would not detect a report
    whose physical configuration was edited after the run.
    """

    li_payload = dict(report.get("li_config", {}))
    em_payload = dict(report.get("em_config", {}))
    if not li_payload or not em_payload:
        raise ValueError("report lacks Li/EM configuration needed to verify hash")
    if em_payload.get("comparison_tracks") == "legacy_pair":
        em_payload.pop("comparison_tracks", None)
    if em_payload.get("right_boundary_model") == "fixed_voltage":
        em_payload.pop("right_boundary_model", None)
        em_payload.pop("right_boundary_conductance_s", None)
    if em_payload.get("parallel_averaging_profile") == "constant":
        for key in (
            "parallel_averaging_profile",
            "dipole_earth_radius_km",
            "dipole_column_lower_altitude_km",
            "dipole_quadrature_order",
        ):
            em_payload.pop(key, None)
    else:
        em_payload["parallel_column_reduction_version"] = (
            "two_moment_surface_fac_column_continuity_v2"
        )
    for key in (
        "t_end_s",
        "growth_fit_start_s",
        "diagnostics_interval_steps",
        "checkpoint_interval_s",
        "output_dir",
        "experiment_label",
    ):
        em_payload.pop(key, None)
    payload = json.dumps(
        {"li": li_payload, "em_trajectory": em_payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_values(records: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for record in records:
        if key not in record or record[key] is None:
            continue
        try:
            values.append(float(record[key]))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _extreme(
    records: Iterable[dict[str, Any]], key: str, operation
) -> float | None:
    values = _finite_values(records, key)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return float(operation(values))


def _linear_log_fit(time_s: np.ndarray, amplitude: np.ndarray) -> dict[str, Any]:
    valid = (
        np.isfinite(time_s)
        & np.isfinite(amplitude)
        & (amplitude > 0.0)
    )
    t = np.asarray(time_s[valid], dtype=np.float64)
    a = np.asarray(amplitude[valid], dtype=np.float64)
    if t.size < 2 or t[-1] <= t[0]:
        return {
            "sample_count": int(t.size),
            "start_s": float(t[0]) if t.size else None,
            "end_s": float(t[-1]) if t.size else None,
            "duration_s": float(t[-1] - t[0]) if t.size else 0.0,
            "rate_s_inv": None,
            "r_squared": None,
        }
    logarithm = np.log(a)
    slope, intercept = np.polyfit(t, logarithm, 1)
    predicted = slope * t + intercept
    residual = float(np.sum((logarithm - predicted) ** 2))
    total = float(np.sum((logarithm - np.mean(logarithm)) ** 2))
    r_squared = (
        1.0
        if total == 0.0 and residual == 0.0
        else (float(1.0 - residual / total) if total > 0.0 else None)
    )
    return {
        "sample_count": int(t.size),
        "start_s": float(t[0]),
        "end_s": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "rate_s_inv": float(slope),
        "r_squared": r_squared,
    }


def _positive_growth_window(
    time_s: np.ndarray, amplitude: np.ndarray
) -> dict[str, Any]:
    """Find the longest strictly positive log-growth run after global A_min."""

    time = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(amplitude, dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(values) & (values > 0.0)
    time = time[valid]
    values = values[valid]
    if time.size == 0:
        return {
            "minimum_time_s": None,
            "minimum_amplitude_m3": None,
            "final_to_minimum_amplitude": None,
            "window": _linear_log_fit(time, values),
        }
    if np.any(np.diff(time) <= 0.0):
        order = np.argsort(time, kind="stable")
        time = time[order]
        values = values[order]
        unique = np.concatenate(([True], np.diff(time) > 0.0))
        time = time[unique]
        values = values[unique]

    minimum_index = int(np.argmin(values))
    minimum = float(values[minimum_index])
    best_start = minimum_index
    best_end = minimum_index
    current_start: int | None = None
    # A positive edge i connects samples i and i+1.  Search only after A_min.
    for edge in range(minimum_index, max(minimum_index, time.size - 1)):
        positive = values[edge + 1] > values[edge]
        if positive and current_start is None:
            current_start = edge
        if (not positive or edge == time.size - 2) and current_start is not None:
            end = edge + 1 if positive else edge
            if time[end] - time[current_start] > time[best_end] - time[best_start]:
                best_start, best_end = current_start, end
            current_start = None
    window = _linear_log_fit(
        time[best_start : best_end + 1], values[best_start : best_end + 1]
    )
    return {
        "minimum_time_s": float(time[minimum_index]),
        "minimum_amplitude_m3": minimum,
        "final_amplitude_m3": float(values[-1]),
        "final_to_minimum_amplitude": float(values[-1] / minimum),
        "window": window,
    }


def _time_close(first: float, second: float) -> bool:
    scale = max(1.0, abs(float(first)), abs(float(second)))
    return bool(abs(float(first) - float(second)) <= 1.0e-10 * scale)


def _load_completed_history(
    root: Path,
    *,
    report: dict[str, Any],
    records: list[dict[str, Any]],
    final_time_s: float,
    final_density: dict[str, np.ndarray],
    final_line_state: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Load schema-v3 history and prove all terminal artifacts agree.

    A completed runner always writes ``checkpoint.npz`` before the final
    arrays, report and completed status.  Silently falling back to a sparse
    diagnostic history can therefore hide a stale/corrupt checkpoint or a
    mixture of artifacts from different trajectories; acceptance rejects
    that condition instead.
    """

    checkpoint_path = root / "checkpoint.npz"
    status_path = root / "status.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    status = _read_json(status_path)
    report_hash = str(report.get("configuration_hash", ""))
    if not report_hash:
        raise ValueError("em_growth_report.json lacks configuration_hash")
    expected_hash = _configuration_hash_from_report(report)
    if report_hash != expected_hash:
        raise ValueError("report configuration hash does not match archived configuration")
    if status.get("status") != "completed":
        raise ValueError("status.json does not mark the run completed")
    if str(status.get("configuration_hash", "")) != report_hash:
        raise ValueError("status/report configuration hashes differ")

    required = {
        "schema_version",
        "time_s",
        "step",
        "configuration_hash",
        "comparison_tracks",
        "baseline_density_m3",
        "column_control_density_m3",
        "coupled_density_m3",
        "coupled_w_plus_v",
        "coupled_w_minus_v",
        "times_s",
        "baseline_amplitudes_m3",
        "column_control_amplitudes_m3",
        "coupled_amplitudes_m3",
    }
    with np.load(checkpoint_path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"checkpoint.npz lacks schema-v3 fields: {missing}")
        if int(data["schema_version"]) != 3:
            raise ValueError("completed three-track checkpoint must use schema 3")
        if str(data["comparison_tracks"]) != "local_column_em":
            raise ValueError("checkpoint comparison_tracks marker is invalid")
        if str(data["configuration_hash"]) != report_hash:
            raise ValueError("checkpoint/report configuration hashes differ")
        checkpoint_time = float(data["time_s"])
        checkpoint_step = int(data["step"])
        if not _time_close(checkpoint_time, final_time_s):
            raise ValueError("checkpoint and final-state times differ")
        density_keys = {
            "local_es": "baseline_density_m3",
            "column_es": "column_control_density_m3",
            "column_em": "coupled_density_m3",
        }
        for track, key in density_keys.items():
            archived = np.asarray(data[key], dtype=np.float64)
            if not np.array_equal(archived, final_density[track]):
                raise ValueError(f"checkpoint/final density differs for {track}")
        for name in ("coupled_w_plus_v", "coupled_w_minus_v"):
            archived = np.asarray(data[name], dtype=np.float64)
            if not np.array_equal(archived, final_line_state[name]):
                raise ValueError(f"checkpoint/final line state differs for {name}")
        history_time = np.asarray(data["times_s"], dtype=np.float64)
        amplitudes = {
            "local_es": np.asarray(
                data["baseline_amplitudes_m3"], dtype=np.float64
            ),
            "column_es": np.asarray(
                data["column_control_amplitudes_m3"], dtype=np.float64
            ),
            "column_em": np.asarray(
                data["coupled_amplitudes_m3"], dtype=np.float64
            ),
        }

    if history_time.ndim != 1 or history_time.size < 2:
        raise ValueError("checkpoint growth history must be a nontrivial vector")
    if not np.all(np.isfinite(history_time)) or np.any(np.diff(history_time) <= 0.0):
        raise ValueError("checkpoint growth times must be finite and increasing")
    if not _time_close(history_time[0], 0.0) or not _time_close(
        history_time[-1], final_time_s
    ):
        raise ValueError("checkpoint growth history does not span 0 to final time")
    for track, values in amplitudes.items():
        if values.shape != history_time.shape:
            raise ValueError(f"checkpoint amplitude length differs for {track}")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"checkpoint amplitudes must be positive for {track}")

    report_time = float(report.get("completed_time_s", np.nan))
    report_step = int(report.get("steps", -1))
    status_time = float(status.get("time_s", np.nan))
    status_step = int(status.get("step", -1))
    if not (
        _time_close(report_time, final_time_s)
        and _time_close(status_time, final_time_s)
    ):
        raise ValueError("report/status/final-state times differ")
    if report_step != checkpoint_step or status_step != checkpoint_step:
        raise ValueError("report/status/checkpoint step counts differ")

    diagnostic_steps = np.asarray(
        [int(record.get("step", -1)) for record in records], dtype=np.int64
    )
    diagnostic_times = _finite_values(records, "time_s")
    if diagnostic_times.size != len(records):
        raise ValueError("every diagnostic record must contain a finite time")
    if (
        not np.all(np.isfinite(diagnostic_times))
        or
        np.any(diagnostic_steps <= 0)
        or np.any(np.diff(diagnostic_steps) <= 0)
        or np.any(np.diff(diagnostic_times) <= 0.0)
        or diagnostic_steps[-1] != checkpoint_step
        or not _time_close(diagnostic_times[-1], final_time_s)
    ):
        raise ValueError(
            "diagnostic steps/times are invalid or do not reach the terminal state"
        )

    return history_time, amplitudes, {
        "checkpoint": str(checkpoint_path),
        "status": str(status_path),
        "configuration_hash": report_hash,
        "checkpoint_schema_version": 3,
        "final_state_schema_version": 2,
        "time_s": final_time_s,
        "step": checkpoint_step,
        "history_sample_count": int(history_time.size),
        "terminal_arrays_equal": True,
    }


def _grid_metadata(
    report: dict[str, Any], shape: tuple[int, int]
) -> dict[str, Any]:
    li = report.get("li_config", {})
    ny, nx = shape
    dx = float(
        li.get(
            "dx_km",
            (float(li["x_max_km"]) - float(li["x_min_km"])) / max(nx - 1, 1),
        )
    )
    dy = float(
        li.get(
            "dy_km",
            (float(li["y_max_km"]) - float(li["y_min_km"])) / max(ny - 1, 1),
        )
    )
    if int(li.get("nx", nx)) != nx or int(li.get("ny", ny)) != ny:
        raise ValueError("final density shape disagrees with archived Li grid")
    span_in_steps = (
        (float(li["x_max_km"]) - float(li["x_min_km"])) / dx
        if dx > 0.0
        else float("nan")
    )
    duplicate = bool(
        nx > 1
        and np.isfinite(span_in_steps)
        and abs(span_in_steps - (nx - 1)) <= 1.0e-8 * max(nx - 1, 1)
    )
    layout = report.get("em_config", {}).get("periodic_x_layout", "legacy_stored_cells")
    if layout == "unique_periodic_cells":
        if not np.isfinite(dx) or dx <= 0.0:
            raise ValueError("unique periodic grid needs positive finite dx")
        if not np.isclose(span_in_steps, nx - 1, rtol=0.0, atol=1.0e-8):
            raise ValueError("unique periodic stored-coordinate bounds disagree with nx and dx")
        periods = nx * dx / float(li["perturbation_wavelength_km"])
        if periods < 1.0 or not np.isclose(periods, round(periods), rtol=0.0, atol=1.0e-10):
            raise ValueError("unique periodic grid seed does not fit nx * dx")
        duplicate = False
    elif layout != "legacy_stored_cells":
        raise ValueError("unsupported archived periodic_x_layout")
    y = float(li["y_min_km"]) + np.arange(ny, dtype=np.float64) * dy
    return {
        "nx_archived": nx,
        "ny": ny,
        "nx_unique": nx - 1 if duplicate else nx,
        "dx_km": dx,
        "dy_km": dy,
        "y_km": y,
        "periodic_endpoint_duplicate_removed": duplicate,
        "periodic_x_layout": layout,
        "computational_period_x_km": nx * dx,
    }


def _unique_x(array: np.ndarray, grid: dict[str, Any]) -> np.ndarray:
    return (
        np.asarray(array[:, :-1], dtype=np.float64)
        if grid["periodic_endpoint_duplicate_removed"]
        else np.asarray(array, dtype=np.float64)
    )


def _periodic_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components with periodic connectivity in x."""

    selected = np.asarray(mask, dtype=bool)
    ny, nx = selected.shape
    visited = np.zeros_like(selected)
    components: list[np.ndarray] = []
    for row in range(ny):
        for column in range(nx):
            if not selected[row, column] or visited[row, column]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, column)])
            visited[row, column] = True
            cells: list[tuple[int, int]] = []
            while queue:
                j, i = queue.popleft()
                cells.append((j, i))
                for jj, ii in (
                    (j - 1, i),
                    (j + 1, i),
                    (j, (i - 1) % nx),
                    (j, (i + 1) % nx),
                ):
                    if 0 <= jj < ny and selected[jj, ii] and not visited[jj, ii]:
                        visited[jj, ii] = True
                        queue.append((jj, ii))
            components.append(np.asarray(cells, dtype=np.int64))
    return components


def _maximum_periodic_run(mask: np.ndarray) -> int:
    """Return the longest contiguous true run on one periodic row."""

    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 1:
        raise ValueError("a periodic width row must be one-dimensional")
    count = int(selected.size)
    if count == 0 or not np.any(selected):
        return 0
    if np.all(selected):
        return count
    longest = 0
    current = 0
    # Starting immediately after a false cell makes a single pass over the
    # circle sufficient and prevents an end/start run from being split.
    start = int(np.flatnonzero(~selected)[0])
    for offset in range(1, count + 1):
        index = (start + offset) % count
        if selected[index]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def bubble_morphology(
    density_m3: np.ndarray,
    *,
    y_km: np.ndarray,
    dx_km: float,
    dy_km: float,
    threshold: float = 0.25,
) -> dict[str, Any]:
    """Measure the largest periodic connected depletion in one density state."""

    density = np.asarray(density_m3, dtype=np.float64)
    zonal_mean = np.mean(density, axis=1, keepdims=True)
    depletion = 1.0 - density / np.maximum(zonal_mean, np.finfo(float).tiny)
    components = _periodic_components(depletion >= threshold)
    if not components:
        return {
            "exists": False,
            "depletion_threshold": float(threshold),
            "component_count": 0,
            "cell_count": 0,
            "bottom_km": None,
            "top_km": None,
            "vertical_span_km": 0.0,
            "area_km2": 0.0,
            "maximum_depletion": float(max(0.0, np.max(depletion))),
            "maximum_width_cells": 0,
            "maximum_width_km": 0.0,
            "width_layer_count": 0,
            "layers_at_least_two_cells_wide": 0,
        }
    # Area is the primary unambiguous selector.  Top/span break exact ties.
    component = max(
        components,
        key=lambda cells: (
            cells.shape[0],
            int(np.max(cells[:, 0]) - np.min(cells[:, 0])),
            int(np.max(cells[:, 0])),
        ),
    )
    rows = component[:, 0]
    columns = component[:, 1]
    unique_rows = np.unique(rows)
    component_mask = np.zeros_like(depletion, dtype=bool)
    component_mask[rows, columns] = True
    widths = np.asarray(
        [_maximum_periodic_run(component_mask[row]) for row in unique_rows],
        dtype=np.int64,
    )
    bottom = float(y_km[int(np.min(rows))])
    top = float(y_km[int(np.max(rows))])
    return {
        "exists": True,
        "depletion_threshold": float(threshold),
        "component_count": len(components),
        "cell_count": int(component.shape[0]),
        "bottom_km": bottom,
        "top_km": top,
        "vertical_span_km": float(top - bottom),
        "area_km2": float(component.shape[0] * dx_km * dy_km),
        "maximum_depletion": float(np.max(depletion[rows, columns])),
        "maximum_width_cells": int(np.max(widths)),
        "maximum_width_km": float(np.max(widths) * dx_km),
        "width_layer_count": int(unique_rows.size),
        "layers_at_least_two_cells_wide": int(np.count_nonzero(widths >= 2)),
    }


def _stage(report: dict[str, Any], final_time_s: float) -> tuple[str, float]:
    requested = float(report.get("em_config", {}).get("t_end_s", final_time_s))
    if requested <= 90.0:
        return "60", 60.0
    if requested <= 600.0:
        return "300", 300.0
    return "3150", 3150.0


class _GateBook:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        status: str,
        value: Any,
        criterion: str,
        message: str,
    ) -> None:
        if status not in {"pass", "warning", "fail"}:
            raise ValueError(status)
        self.checks.append(
            {
                "name": name,
                "status": status,
                "value": _json_ready(value),
                "criterion": criterion,
                "message": message,
            }
        )

    def required(self, name: str, condition: bool, value: Any, criterion: str) -> None:
        self.add(
            name,
            "pass" if condition else "fail",
            value,
            criterion,
            "criterion satisfied" if condition else "required criterion failed",
        )

    def upper(
        self,
        name: str,
        value: float | None,
        *,
        warning: float,
        failure: float,
    ) -> None:
        if value is None or not np.isfinite(value):
            self.add(name, "fail", value, f"finite and <= {failure:g}", "metric missing")
        elif value > failure:
            self.add(name, "fail", value, f"<= {failure:g}", "failure limit exceeded")
        elif value > warning:
            self.add(name, "warning", value, f"preferred <= {warning:g}", "near limit")
        else:
            self.add(name, "pass", value, f"<= {warning:g}", "criterion satisfied")

    def lower(
        self,
        name: str,
        value: float | None,
        *,
        failure: float,
        preferred: float | None = None,
    ) -> None:
        if value is None or not np.isfinite(value):
            self.add(name, "fail", value, f"finite and >= {failure:g}", "metric missing")
        elif value < failure:
            self.add(name, "fail", value, f">= {failure:g}", "below failure limit")
        elif preferred is not None and value < preferred:
            self.add(name, "warning", value, f"preferred >= {preferred:g}", "below preferred target")
        else:
            target = preferred if preferred is not None else failure
            self.add(name, "pass", value, f">= {target:g}", "criterion satisfied")


def _maximum_absolute(records: list[dict[str, Any]], key: str) -> float | None:
    return _extreme(records, key, lambda values: np.max(np.abs(values)))


def _relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(first)), np.finfo(float).tiny)
    return float(np.linalg.norm(second - first) / denominator)


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    """NumPy-1.26/2.x compatible trapezoidal integral."""

    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(coordinates, dtype=np.float64)
    if y.size < 2 or y.shape != x.shape:
        return 0.0
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _per_record_fac_balance(
    records: list[dict[str, Any]], *, computational_area_m2: float
) -> dict[str, Any]:
    """Normalize FAC imbalance at each time rather than mixing extrema.

    Current archives contain the net integral and the pointwise maximum, not
    ``integral(abs(J))``.  The resulting ``|net|/(max|J| A)`` is therefore a
    conservative compatibility proxy (a lower bound on the usual relative
    L1 imbalance), and is labelled as such.  A future direct relative metric
    can be consumed without changing the report schema.
    """

    ratios: list[float] = []
    methods: set[str] = set()
    for record in records:
        direct = record.get("relative_net_fac_forcing_integral")
        if direct is not None:
            ratio = abs(float(direct))
            method = "direct_relative_integral"
            if not np.isfinite(ratio):
                return {
                    "evaluable": False,
                    "reason": "direct relative FAC imbalance is non-finite",
                    "normalization": method,
                    "record_count": len(ratios),
                    "maximum_ratio": None,
                }
        else:
            method = "max_abs_times_computational_area_proxy"
            if (
                "net_fac_forcing_integral" not in record
                or "maximum_abs_fac_forcing" not in record
            ):
                return {
                    "evaluable": False,
                    "reason": "FAC net integral or pointwise maximum is absent",
                    "normalization": method,
                    "record_count": len(ratios),
                    "maximum_ratio": None,
                }
            net = abs(float(record["net_fac_forcing_integral"]))
            maximum = abs(float(record["maximum_abs_fac_forcing"]))
            if not np.isfinite(net) or not np.isfinite(maximum):
                return {
                    "evaluable": False,
                    "reason": "FAC balance inputs are non-finite",
                    "normalization": method,
                    "record_count": len(ratios),
                    "maximum_ratio": None,
                }
            denominator = maximum * computational_area_m2
            ratio = (
                0.0
                if net == 0.0 and denominator == 0.0
                else net / max(denominator, np.finfo(float).tiny)
            )
        methods.add(method)
        ratios.append(float(ratio))
    normalization = (
        next(iter(methods))
        if len(methods) == 1
        else ("mixed_per_record" if methods else None)
    )
    return {
        "evaluable": bool(ratios),
        "reason": None if ratios else "no diagnostic records",
        "normalization": normalization,
        "normalizations_used": sorted(methods),
        "record_count": len(ratios),
        "maximum_ratio": float(np.max(ratios)) if ratios else None,
    }


def _retained_modal_eigenvalues_m2(
    *,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    max_x_wavenumber: int,
    max_y_wavenumber: int,
    include_sine_modes: bool,
) -> np.ndarray:
    """Reconstruct the archived low-wavenumber modal ordering/eigenvalues."""

    values: list[float] = []
    for ky in range(int(max_y_wavenumber) + 1):
        lambda_y = (
            4.0 * np.sin(0.5 * np.pi * ky / int(ny)) ** 2 / float(dy_m) ** 2
        )
        for kx in range(int(max_x_wavenumber) + 1):
            lambda_x = (
                4.0 * np.sin(np.pi * kx / int(nx)) ** 2 / float(dx_m) ** 2
            )
            eigenvalue = float(lambda_x + lambda_y)
            if kx != 0 or ky != 0:
                values.append(eigenvalue)
            if (
                include_sine_modes
                and kx > 0
                and not (int(nx) % 2 == 0 and kx == int(nx) // 2)
            ):
                values.append(eigenvalue)
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0 or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("retained modal eigenvalues are invalid")
    return result


def _line_energy_budget(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    final_line_state: dict[str, np.ndarray],
    final_time_s: float,
) -> dict[str, Any]:
    """Evaluate the physical characteristic-line energy and sampled port work.

    The characteristic energy itself is exact for the archived cell-centred
    state.  Existing diagnostics store one representative SSPRK-stage power
    per accepted macro step rather than the RK-weighted stage integral, so the
    port-work quadrature is explicitly identified as sampled/trapezoidal.  It
    can reject net energy creation, but it is not advertised as an exact
    discrete SSPRK energy identity.
    """

    em = report.get("em_config", {})
    li = report.get("li_config", {})
    if em.get("initial_line_state") != "zero" or float(
        em.get("initial_characteristic_seed_v", np.nan)
    ) != 0.0:
        return {
            "evaluable": False,
            "reason": "initial line energy is not archived for a nonzero line state",
            "verification_level": "not_evaluable",
        }
    required_record_keys = (
        "time_s",
        "boundary_power_modal_w",
        "remote_boundary_power_modal_w",
    )
    if not records or any(
        any(key not in record for key in required_record_keys)
        for record in records
    ):
        return {
            "evaluable": False,
            "reason": "sampled ionosphere/remote port powers are incomplete",
            "verification_level": "not_evaluable",
        }
    try:
        nx = int(li["nx"])
        ny = int(li["ny"])
        dx_m = float(li["dx_km"]) * 1000.0
        dy_m = float(li["dy_km"]) * 1000.0
        cells = int(em["field_line_cells"])
        length_m = float(em["field_line_length_km"]) * 1000.0
        speed_m_s = float(em["alfven_speed_km_s"]) * 1000.0
        hemispheres = int(em["hemisphere_count"])
        eigenvalues = _retained_modal_eigenvalues_m2(
            nx=nx,
            ny=ny,
            dx_m=dx_m,
            dy_m=dy_m,
            max_x_wavenumber=int(em["max_x_wavenumber"]),
            max_y_wavenumber=int(em["max_y_wavenumber"]),
            include_sine_modes=bool(em["include_sine_modes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "evaluable": False,
            "reason": f"line/basis metadata is incomplete: {error}",
            "verification_level": "not_evaluable",
        }
    plus = np.asarray(final_line_state["coupled_w_plus_v"], dtype=np.float64)
    minus = np.asarray(final_line_state["coupled_w_minus_v"], dtype=np.float64)
    if (
        plus.shape != (cells, eigenvalues.size)
        or minus.shape != plus.shape
        or not np.all(np.isfinite(plus))
        or not np.all(np.isfinite(minus))
    ):
        return {
            "evaluable": False,
            "reason": "final characteristic arrays disagree with line/basis metadata",
            "verification_level": "not_evaluable",
        }
    mu0 = 4.0e-7 * np.pi
    domain_area_m2 = nx * ny * dx_m * dy_m
    spacing_m = length_m / cells
    coefficient = (
        hemispheres
        * domain_area_m2
        * spacing_m
        / (4.0 * mu0 * speed_m_s**2)
    )
    final_energy = float(
        coefficient
        * np.sum((plus**2 + minus**2) * eigenvalues[None, :])
    )
    sample_time = np.asarray([record["time_s"] for record in records], dtype=np.float64)
    left_power = np.asarray(
        [record["boundary_power_modal_w"] for record in records], dtype=np.float64
    )
    remote_power = np.asarray(
        [record["remote_boundary_power_modal_w"] for record in records],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(sample_time))
        or not np.all(np.isfinite(left_power))
        or not np.all(np.isfinite(remote_power))
        or np.any(np.diff(sample_time) <= 0.0)
        or not _time_close(sample_time[-1], final_time_s)
    ):
        return {
            "evaluable": False,
            "reason": "sampled port powers are non-finite, unordered, or do not reach final time",
            "verification_level": "not_evaluable",
        }
    if sample_time[0] > 0.0:
        sample_time = np.concatenate(([0.0], sample_time))
        left_power = np.concatenate(([0.0], left_power))
        remote_power = np.concatenate(([0.0], remote_power))
    net_boundary_work = _trapezoid(left_power - remote_power, sample_time)
    remote_absorbed = _trapezoid(remote_power, sample_time)
    initial_energy = 0.0
    energy_change = final_energy - initial_energy
    work_minus_energy = net_boundary_work - energy_change
    scale = max(
        abs(net_boundary_work),
        abs(energy_change),
        abs(remote_absorbed),
        np.finfo(float).tiny,
    )
    return {
        "evaluable": True,
        "reason": None,
        "verification_level": "physical_energy_with_sampled_port_quadrature",
        "characteristic_energy_formula_exact": True,
        "discrete_ssprk_stage_power_integral_archived": False,
        "port_work_quadrature": "trapezoidal_representative_stage_samples",
        "initial_line_energy_j": initial_energy,
        "final_line_energy_j": final_energy,
        "line_energy_change_j": energy_change,
        "net_boundary_work_j": net_boundary_work,
        "remote_absorbed_energy_j": remote_absorbed,
        "work_minus_energy_j": work_minus_energy,
        "relative_work_minus_energy": float(work_minus_energy / scale),
        "numerical_energy_creation_fraction": float(
            max(0.0, -work_minus_energy) / scale
        ),
        "numerical_dissipation_fraction": float(
            max(0.0, work_minus_energy) / scale
        ),
        "sample_count_including_t0": int(sample_time.size),
    }


def evaluate_run(run_dir: str | Path) -> dict[str, Any]:
    """Evaluate one completed three-track run without modifying it."""

    root = Path(run_dir).resolve()
    report_path = root / "em_growth_report.json"
    diagnostics_path = root / "diagnostics.jsonl"
    states_path = root / "final_states.npz"
    for path in (report_path, diagnostics_path, states_path):
        if not path.exists():
            raise FileNotFoundError(path)
    report = _read_json(report_path)
    records = _read_jsonl(diagnostics_path)
    if not records:
        raise ValueError("diagnostics.jsonl is empty")
    with np.load(states_path, allow_pickle=False) as states:
        required = {
            "schema_version",
            "baseline_density_m3",
            "column_control_density_m3",
            "coupled_density_m3",
            "coupled_w_plus_v",
            "coupled_w_minus_v",
            "time_s",
        }
        missing = sorted(required - set(states.files))
        if missing:
            raise ValueError(f"final_states.npz lacks three-track arrays: {missing}")
        if int(states["schema_version"]) != 2:
            raise ValueError("completed three-track final state must use schema 2")
        density = {
            "local_es": np.asarray(states["baseline_density_m3"], dtype=np.float64),
            "column_es": np.asarray(
                states["column_control_density_m3"], dtype=np.float64
            ),
            "column_em": np.asarray(states["coupled_density_m3"], dtype=np.float64),
        }
        final_line_state = {
            "coupled_w_plus_v": np.asarray(
                states["coupled_w_plus_v"], dtype=np.float64
            ),
            "coupled_w_minus_v": np.asarray(
                states["coupled_w_minus_v"], dtype=np.float64
            ),
        }
        final_time = float(states["time_s"])
    shapes = {array.shape for array in density.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("three final density arrays must share one two-dimensional shape")

    grid = _grid_metadata(report, next(iter(shapes)))
    unique_density = {track: _unique_x(array, grid) for track, array in density.items()}
    endpoint_mismatch: dict[str, float | None] = {}
    for track, array in density.items():
        if grid["periodic_endpoint_duplicate_removed"]:
            scale = max(
                float(np.linalg.norm(array[:, 0])),
                float(np.linalg.norm(array[:, -1])),
                np.finfo(float).tiny,
            )
            endpoint_mismatch[track] = float(
                np.linalg.norm(array[:, -1] - array[:, 0]) / scale
            )
        else:
            endpoint_mismatch[track] = None

    history_time, amplitudes, archive_consistency = _load_completed_history(
        root,
        report=report,
        records=records,
        final_time_s=final_time,
        final_density=density,
        final_line_state=final_line_state,
    )
    history_source = "checkpoint_schema_v3"
    fit_start = float(report.get("em_config", {}).get("growth_fit_start_s", 0.0))
    growth: dict[str, Any] = {}
    for track in TRACKS:
        selected = history_time >= fit_start
        positive = _positive_growth_window(history_time, amplitudes[track])
        late_start = max(float(history_time[-1]) - 100.0, float(history_time[0]))
        late = history_time >= late_start
        growth[track] = {
            "archived_fit_window": _linear_log_fit(
                history_time[selected], amplitudes[track][selected]
            ),
            "late_100s_fit": _linear_log_fit(
                history_time[late], amplitudes[track][late]
            ),
            "post_minimum_positive_growth": positive,
        }

    bubble = {
        track: bubble_morphology(
            unique_density[track],
            y_km=grid["y_km"],
            dx_km=grid["dx_km"],
            dy_km=grid["dy_km"],
            threshold=THRESHOLDS["bubble"]["depletion_threshold"],
        )
        for track in TRACKS
    }
    relative_l2 = {
        "column_es_vs_local_es": _relative_l2(
            unique_density["local_es"], unique_density["column_es"]
        ),
        "column_em_vs_column_es": _relative_l2(
            unique_density["column_es"], unique_density["column_em"]
        ),
        "column_em_vs_local_es": _relative_l2(
            unique_density["local_es"], unique_density["column_em"]
        ),
    }

    times = _finite_values(records, "time_s")
    remote_power = _finite_values(records, "remote_boundary_power_modal_w")
    boundary_power = _finite_values(records, "boundary_power_modal_w")
    if remote_power.size == times.size and times.size >= 2:
        remote_energy = _trapezoid(remote_power, times)
    else:
        remote_energy = None
    if boundary_power.size == times.size and times.size >= 2:
        port_energy_scale = _trapezoid(np.abs(boundary_power), times)
    else:
        port_energy_scale = None
    absorption_fraction = (
        float(remote_energy / port_energy_scale)
        if remote_energy is not None
        and port_energy_scale is not None
        and port_energy_scale > np.finfo(float).tiny
        else None
    )

    fac_ratio_values = _finite_values(records, "fac_to_gravity_ratio")
    activation = _finite_values(records, "activation_fraction")
    if activation.size == fac_ratio_values.size and activation.size:
        active_fac = fac_ratio_values[activation >= 0.99]
    else:
        ramp = float(report.get("em_config", {}).get("feedback_ramp_time_s", 0.0))
        active_fac = (
            fac_ratio_values[times >= ramp]
            if times.size == fac_ratio_values.size
            else fac_ratio_values
        )
    active_fac = active_fac[np.isfinite(active_fac)]
    fac_ratio_median = float(np.median(active_fac)) if active_fac.size else None
    fac_ratio_maximum = float(np.max(active_fac)) if active_fac.size else None

    max_fac = _extreme(records, "maximum_abs_fac_forcing", np.max)
    max_net_fac = _maximum_absolute(records, "net_fac_forcing_integral")
    computational_area_m2 = float(
        grid["nx_unique"]
        * grid["ny"]
        * grid["dx_km"]
        * grid["dy_km"]
        * 1.0e6
    )
    fac_balance = _per_record_fac_balance(
        records, computational_area_m2=computational_area_m2
    )
    net_surface_ratio = fac_balance["maximum_ratio"]

    numerical = {
        "minimum_dt_s": _extreme(records, "dt_s", np.min),
        "maximum_baseline_cfl": _extreme(records, "baseline_cfl", np.max),
        "maximum_column_control_cfl": _extreme(
            records, "column_control_cfl", np.max
        ),
        "maximum_coupled_cfl": _extreme(records, "coupled_cfl", np.max),
        "maximum_baseline_potential_relative_residual": _extreme(
            records, "baseline_potential_relative_residual", np.max
        ),
        "maximum_column_control_potential_relative_residual": _extreme(
            records, "column_control_potential_relative_residual", np.max
        ),
        "maximum_coupled_potential_relative_residual": _extreme(
            records, "coupled_potential_relative_residual", np.max
        ),
        "maximum_coupling_relative_residual": _extreme(
            records, "coupling_relative_residual", np.max
        ),
        "minimum_densities_m3": {
            track: float(np.min(array)) for track, array in density.items()
        },
        "all_final_density_values_finite": all(
            bool(np.all(np.isfinite(array))) for array in density.values()
        ),
        "dt_reductions": int(report.get("dt_reductions", 0)),
    }
    conservation = {
        "maximum_absolute_local_es_content_relative_change": _maximum_absolute(
            records, "baseline_local_density_content_relative_change"
        ),
        "maximum_absolute_column_es_content_relative_change": _maximum_absolute(
            records, "column_control_conserved_density_content_relative_change"
        ),
        "maximum_absolute_column_em_content_relative_change": _maximum_absolute(
            records, "coupled_conserved_density_content_relative_change"
        ),
        "balance_status": "not_evaluable_without_vertical_boundary_flux_history",
        "is_acceptance_gate": False,
        "reason": (
            "periodic_y=False archives allow physical top/bottom advective "
            "flux; raw domain-content drift is not a conservation residual"
        ),
    }
    fac_and_power = {
        "fac_to_gravity_ratio_active_median": fac_ratio_median,
        "fac_to_gravity_ratio_active_maximum": fac_ratio_maximum,
        "maximum_abs_surface_fac_a_m2": max_fac,
        "maximum_abs_net_surface_current_a": max_net_fac,
        "net_surface_current_to_max_domain_current_ratio": net_surface_ratio,
        "surface_fac_balance": fac_balance,
        "maximum_relative_power_pairing_defect": _extreme(
            records, "relative_power_pairing_defect", np.max
        ),
        "remote_absorbed_energy_j": remote_energy,
        "integrated_absolute_ionosphere_port_power_j": port_energy_scale,
        "remote_absorption_fraction_of_absolute_port_energy": absorption_fraction,
        "minimum_remote_power_w": _extreme(
            records, "remote_boundary_power_modal_w", np.min
        ),
        "maximum_abs_remote_reflection_coefficient": _maximum_absolute(
            records, "remote_boundary_reflection_coefficient"
        ),
    }
    line_energy = _line_energy_budget(
        report, records, final_line_state, final_time
    )
    fac_and_power["line_energy_budget"] = line_energy

    stage, stage_target = _stage(report, final_time)
    gates = _GateBook()
    em = report.get("em_config", {})
    scope = report.get("scope", {})
    li = report.get("li_config", {})
    gates.required(
        "three_track_schema",
        report.get("format") == "esf_dimensional_em_growth_v2"
        and em.get("comparison_tracks") == "local_column_em",
        {"format": report.get("format"), "tracks": em.get("comparison_tracks")},
        "v2 and comparison_tracks=local_column_em",
    )
    gates.required(
        "physical_parallel_column",
        scope.get("parallel_averaging_profile") == "dipole_chapman_flux_tube"
        and em.get("parallel_averaging_length_km") is None,
        {
            "profile": scope.get("parallel_averaging_profile"),
            "constant_length_km": em.get("parallel_averaging_length_km"),
        },
        "dipole_chapman_flux_tube with no constant H_parallel",
    )
    gates.required(
        "surface_fac_representation",
        scope.get("fac_forcing_representation") == "surface_current",
        scope.get("fac_forcing_representation"),
        "surface_current",
    )
    gates.required(
        "matched_remote_boundary",
        scope.get("remote_boundary_model") == "matched"
        and abs(float(scope.get("remote_boundary_reflection_coefficient", np.inf)))
        <= 1.0e-12,
        {
            "model": scope.get("remote_boundary_model"),
            "reflection": scope.get("remote_boundary_reflection_coefficient"),
        },
        "matched and |R| <= 1e-12",
    )
    gates.required(
        "unit_feedback_gain",
        float(em.get("feedback_gain", np.nan)) == 1.0,
        em.get("feedback_gain"),
        "feedback_gain = 1 (not a weakened diagnostic coupling)",
    )
    gates.required(
        "deterministic_density_initialization",
        float(li.get("relative_noise", 0.0)) == 0.0
        and float(em.get("initial_characteristic_seed_v", 0.0)) == 0.0,
        {
            "relative_noise": li.get("relative_noise", 0.0),
            "line_seed_v": em.get("initial_characteristic_seed_v", 0.0),
        },
        "no random density noise and no explicit line seed",
    )
    requested_end = float(em.get("t_end_s", final_time))
    gates.required(
        "completed_requested_time",
        _time_close(final_time, requested_end),
        {"final_time_s": final_time, "requested_time_s": requested_end},
        "final_time equals requested_time",
    )
    endpoint_policy = THRESHOLDS["periodic_endpoint_relative_mismatch"]
    if grid["periodic_endpoint_duplicate_removed"]:
        for track in TRACKS:
            gates.upper(
                f"periodic_endpoint_match_{track}",
                endpoint_mismatch[track],
                warning=endpoint_policy["warning"],
                failure=endpoint_policy["failure"],
            )

    cfl_policy = THRESHOLDS["density_cfl"]
    gates.upper(
        "local_es_cfl",
        numerical["maximum_baseline_cfl"],
        warning=cfl_policy["warning"],
        failure=cfl_policy["failure"],
    )
    gates.upper(
        "column_es_cfl",
        numerical["maximum_column_control_cfl"],
        warning=cfl_policy["warning"],
        failure=cfl_policy["failure"],
    )
    gates.upper(
        "column_em_combined_cfl",
        numerical["maximum_coupled_cfl"],
        warning=THRESHOLDS["coupled_cfl"]["warning"],
        failure=THRESHOLDS["coupled_cfl"]["failure"],
    )
    for key in (
        "maximum_baseline_potential_relative_residual",
        "maximum_column_control_potential_relative_residual",
        "maximum_coupled_potential_relative_residual",
        "maximum_coupling_relative_residual",
    ):
        gates.upper(
            key,
            numerical[key],
            warning=THRESHOLDS["relative_residual"]["warning"],
            failure=THRESHOLDS["relative_residual"]["failure"],
        )
    gates.required(
        "positive_finite_density",
        numerical["all_final_density_values_finite"]
        and all(value > 0.0 for value in numerical["minimum_densities_m3"].values()),
        numerical["minimum_densities_m3"],
        "all finite and strictly positive",
    )
    if fac_balance["evaluable"]:
        gates.upper(
            "net_surface_current_balance_proxy",
            net_surface_ratio,
            warning=THRESHOLDS["net_surface_current_ratio"]["warning"],
            failure=THRESHOLDS["net_surface_current_ratio"]["failure"],
        )
    else:
        gates.add(
            "net_surface_current_balance_proxy",
            "fail",
            fac_balance,
            "finite per-record normalized FAC balance",
            "metric missing",
        )
    gates.upper(
        "modal_grid_power_pairing",
        fac_and_power["maximum_relative_power_pairing_defect"],
        warning=THRESHOLDS["power_pairing_defect"]["warning"],
        failure=THRESHOLDS["power_pairing_defect"]["failure"],
    )
    minimum_remote_power = fac_and_power["minimum_remote_power_w"]
    maximum_remote_power = _maximum_absolute(
        records, "remote_boundary_power_modal_w"
    )
    maximum_boundary_power = _maximum_absolute(records, "boundary_power_modal_w")
    remote_power_tolerance = 1.0e-10 * max(
        maximum_remote_power or 0.0,
        maximum_boundary_power or 0.0,
        1.0,
    )
    remote_energy_tolerance = 1.0e-10 * max(port_energy_scale or 0.0, 1.0)
    gates.required(
        "passive_remote_absorption",
        remote_energy is not None
        and minimum_remote_power is not None
        and minimum_remote_power >= -remote_power_tolerance
        and remote_energy >= -remote_energy_tolerance,
        {
            "absorbed_energy_j": remote_energy,
            "minimum_power_w": minimum_remote_power,
            "negative_power_tolerance_w": remote_power_tolerance,
            "negative_energy_tolerance_j": remote_energy_tolerance,
        },
        "matched-terminal power is nonnegative record-by-record and in integral",
    )
    if not line_energy.get("evaluable", False):
        gates.add(
            "line_energy_budget",
            "fail",
            line_energy,
            "physical characteristic energy and sampled two-port work evaluable",
            "line energy budget is not evaluable",
        )
    else:
        energy_policy = THRESHOLDS["line_energy_budget"]
        creation = float(line_energy["numerical_energy_creation_fraction"])
        dissipation = float(line_energy["numerical_dissipation_fraction"])
        if creation > energy_policy["maximum_creation_fraction"]:
            gates.add(
                "line_energy_budget",
                "fail",
                line_energy,
                (
                    "numerical creation fraction <= "
                    f"{energy_policy['maximum_creation_fraction']:g}"
                ),
                "sampled line budget indicates net energy creation",
            )
        elif dissipation > energy_policy["dissipation_warning_fraction"]:
            gates.add(
                "line_energy_budget",
                "warning",
                line_energy,
                (
                    "preferred numerical dissipation fraction <= "
                    f"{energy_policy['dissipation_warning_fraction']:g}"
                ),
                "line propagation is materially dissipative",
            )
        else:
            gates.add(
                "line_energy_budget",
                "pass",
                line_energy,
                "no net energy creation within sampled-power tolerance",
                "criterion satisfied",
            )
    fac_limits = THRESHOLDS["fac_to_gravity_ratio"]
    if fac_ratio_median is None:
        gates.add(
            "fac_to_gravity_ratio",
            "fail",
            None,
            "active median in [0.02, 0.85]",
            "metric missing",
        )
    elif not fac_limits["failure"][0] <= fac_ratio_median <= fac_limits["failure"][1]:
        gates.add(
            "fac_to_gravity_ratio",
            "fail",
            fac_ratio_median,
            "active median in [0.02, 0.85]",
            "coupling is negligible or FAC dominates gravity",
        )
    elif not fac_limits["preferred"][0] <= fac_ratio_median <= fac_limits["preferred"][1]:
        gates.add(
            "fac_to_gravity_ratio",
            "warning",
            fac_ratio_median,
            "preferred active median in [0.05, 0.70]",
            "outside preferred physical sensitivity range",
        )
    else:
        gates.add(
            "fac_to_gravity_ratio",
            "pass",
            fac_ratio_median,
            "active median in [0.05, 0.70]",
            "criterion satisfied",
        )

    gates.add(
        "column_content_balance",
        "warning",
        conservation,
        "archive top/bottom flux and close the discrete column-content budget",
        "raw content drift is diagnostic only for an open vertical boundary",
    )
    dt_policy = THRESHOLDS["minimum_dt_s"][stage]
    minimum_dt = numerical["minimum_dt_s"]
    if minimum_dt is None:
        gates.add("minimum_dt", "fail", None, "finite", "metric missing")
    elif minimum_dt < dt_policy["failure"]:
        gates.add(
            "minimum_dt",
            "fail",
            minimum_dt,
            f">= {dt_policy['failure']:g} s",
            "step collapse indicates ineffective progress",
        )
    elif minimum_dt < dt_policy["warning"]:
        gates.add(
            "minimum_dt",
            "warning",
            minimum_dt,
            f"preferred >= {dt_policy['warning']:g} s",
            "adaptive stepping is materially slower",
        )
    else:
        gates.add(
            "minimum_dt",
            "pass",
            minimum_dt,
            f">= {dt_policy['warning']:g} s",
            "criterion satisfied",
        )
    if stage == "60":
        gates.required(
            "no_60s_dt_reduction",
            numerical["dt_reductions"] == 0,
            numerical["dt_reductions"],
            "zero macro-step reductions",
        )

    em_l2 = relative_l2["column_em_vs_column_es"]
    if stage == "300":
        early = THRESHOLDS["early_growth"]
        gates.lower(
            "early_column_em_growth_rate",
            growth["column_em"]["late_100s_fit"]["rate_s_inv"],
            failure=early["minimum_column_em_rate_s_inv"],
        )
        gates.lower(
            "early_em_column_relative_l2",
            em_l2,
            failure=early["relative_l2_failure"],
            preferred=early["relative_l2_preferred"],
        )
    elif stage == "3150":
        mature = THRESHOLDS["mature_growth"]
        em_positive = growth["column_em"]["post_minimum_positive_growth"]
        window = em_positive["window"]
        gates.lower(
            "mature_positive_growth_window_duration",
            window["duration_s"],
            failure=mature["minimum_positive_window_s"],
        )
        gates.lower(
            "mature_column_em_growth_rate",
            window["rate_s_inv"],
            failure=mature["minimum_rate_s_inv"],
            preferred=mature["preferred_rate_s_inv"],
        )
        gates.lower(
            "mature_final_to_minimum_amplitude",
            em_positive["final_to_minimum_amplitude"],
            failure=mature["minimum_final_to_minimum_amplitude"],
        )
        gates.lower(
            "mature_em_column_relative_l2",
            em_l2,
            failure=mature["relative_l2_failure"],
            preferred=mature["relative_l2_preferred"],
        )
        column_rate = growth["column_es"]["post_minimum_positive_growth"][
            "window"
        ]["rate_s_inv"]
        em_rate = window["rate_s_inv"]
        rate_ratio = (
            float(em_rate / column_rate)
            if em_rate is not None and column_rate is not None and column_rate > 0.0
            else None
        )
        gates.lower(
            "mature_em_to_column_positive_rate_ratio",
            rate_ratio,
            failure=mature["minimum_em_to_column_positive_rate_ratio"],
        )
        morphology = bubble["column_em"]
        limits = THRESHOLDS["bubble"]
        gates.required(
            "mature_bubble_exists",
            morphology["exists"],
            morphology["component_count"],
            "at least one periodic D >= 0.25 component",
        )
        gates.lower(
            "mature_bubble_top",
            morphology["top_km"],
            failure=limits["minimum_top_km"],
            preferred=limits["preferred_top_km"],
        )
        gates.lower(
            "mature_bubble_vertical_span",
            morphology["vertical_span_km"],
            failure=limits["minimum_span_km"],
            preferred=limits["preferred_span_km"],
        )
        gates.lower(
            "mature_bubble_area",
            morphology["area_km2"],
            failure=limits["minimum_area_km2"],
        )
        gates.lower(
            "mature_bubble_maximum_depletion",
            morphology["maximum_depletion"],
            failure=limits["minimum_maximum_depletion"],
        )
        gates.lower(
            "mature_bubble_resolved_width_layers",
            morphology["layers_at_least_two_cells_wide"],
            failure=limits["minimum_two_cell_width_layers"],
        )

    failures = [item["message"] + f" [{item['name']}]" for item in gates.checks if item["status"] == "fail"]
    warnings = [item["message"] + f" [{item['name']}]" for item in gates.checks if item["status"] == "warning"]
    result = {
        "format": FORMAT,
        "run_dir": str(root),
        "stage": {
            "name": stage,
            "nominal_target_s": stage_target,
            "requested_end_s": requested_end,
            "completed_time_s": final_time,
        },
        "policy": THRESHOLDS,
        "inputs": {
            "report": str(report_path),
            "diagnostics": str(diagnostics_path),
            "final_states": str(states_path),
            "history_source": history_source,
            "diagnostic_records": len(records),
        },
        "grid": {
            key: value for key, value in grid.items() if key != "y_km"
        }
        | {"periodic_endpoint_relative_mismatch": endpoint_mismatch},
        "numerical_health": numerical,
        "column_particle_budget": conservation,
        "fac_and_power": fac_and_power,
        "growth": growth,
        "relative_l2": relative_l2,
        "morphology": bubble,
        "checks": gates.checks,
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "interpretation": (
            "EPB growth passed with unit feedback and a finite EM-column response; "
            "this is not a weak-coupling recovery."
            if not failures and stage == "3150"
            else (
                "stage gates passed" if not failures else "one or more required gates failed"
            )
        ),
    }
    return _json_ready(result)


def write_acceptance_report(
    run_dir: str | Path, output: str | Path | None = None
) -> dict[str, Any]:
    result = evaluate_run(run_dir)
    destination = (
        Path(output)
        if output is not None
        else Path(run_dir) / "em_growth_acceptance.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade a completed three-track reduced-EM ESF growth run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON destination (default: RUN_DIR/em_growth_acceptance.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = write_acceptance_report(arguments.run_dir, arguments.output)
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"em-growth acceptance failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["pass"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "FORMAT",
    "THRESHOLDS",
    "bubble_morphology",
    "evaluate_run",
    "main",
    "write_acceptance_report",
]
