"""Auditable post-processing for paired electrostatic/reduced-EM growth runs.

The numerical runner deliberately owns no plotting state.  This module reads
its archived arrays, selects the latest accepted density pair, reconstructs the
growth history, and writes a compact scientific comparison figure plus machine-
readable metrics.  It does not require the Li et al. reference data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .bidirectional_em import density_mode_amplitude


DISPLAY_SCALE_M3 = 1.0e11  # 10^5 cm^-3


@dataclass(frozen=True)
class DensityPair:
    """One accepted electrostatic/reduced-EM density pair."""

    baseline_m3: np.ndarray
    coupled_m3: np.ndarray
    time_s: float
    path: Path
    kind: str
    column_control_m3: np.ndarray | None = None


@dataclass(frozen=True)
class GrowthHistory:
    """Horizontal-structure amplitudes from an archived trajectory."""

    time_s: np.ndarray
    baseline_amplitude_m3: np.ndarray
    coupled_amplitude_m3: np.ndarray
    path: Path
    kind: str
    column_control_amplitude_m3: np.ndarray | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _scalar(payload: np.lib.npyio.NpzFile, key: str) -> float:
    if key not in payload.files:
        raise KeyError(f"{payload.zip.filename} does not contain {key!r}")
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"{key!r} in {payload.zip.filename} is not scalar")
    return float(value.reshape(()))


def _density_pair_from_npz(path: Path, kind: str) -> DensityPair:
    with np.load(path, allow_pickle=False) as payload:
        baseline = np.asarray(payload["baseline_density_m3"], dtype=np.float64)
        coupled = np.asarray(payload["coupled_density_m3"], dtype=np.float64)
        column = (
            np.asarray(payload["column_control_density_m3"], dtype=np.float64)
            if "column_control_density_m3" in payload.files
            else None
        )
        time_s = _scalar(payload, "time_s")
    if baseline.ndim != 2 or coupled.shape != baseline.shape:
        raise ValueError(f"{path} does not contain a conformable 2-D density pair")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(coupled)):
        raise ValueError(f"{path} contains non-finite density")
    if np.min(baseline) <= 0.0 or np.min(coupled) <= 0.0:
        raise ValueError(f"{path} contains non-positive accepted density")
    if column is not None:
        if column.shape != baseline.shape or not np.all(np.isfinite(column)):
            raise ValueError(f"{path} contains invalid column-control density")
        if np.min(column) <= 0.0:
            raise ValueError(f"{path} contains non-positive accepted density")
    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError(f"{path} contains an invalid time")
    return DensityPair(baseline, coupled, time_s, path.resolve(), kind, column)


def _snapshot_paths(root: Path) -> list[Path]:
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return []
    # Production snapshots use a deterministic integer archive index
    # (``snapshot_000123.npz``).  Early calibration outputs encoded the time in
    # the filename (``snapshot_t0123.0.npz``); retain read compatibility while
    # excluding any atomic-write temporary files.
    paths = {path for path in snapshots.glob("snapshot_t*.npz") if path.is_file()}
    paths.update(
        path
        for path in snapshots.glob("snapshot_*.npz")
        if path.is_file() and path.stem.removeprefix("snapshot_").isdigit()
    )
    return sorted(paths)


def load_latest_density_pair(root: str | Path) -> DensityPair:
    """Load the latest accepted pair from snapshots, final state, or checkpoint.

    Time, rather than filename order, decides which source is latest.  Equal-time
    ties prefer ``final_states.npz``, then an explicit snapshot, then the rolling
    checkpoint.
    """

    root = Path(root)
    candidates: list[DensityPair] = []
    final = root / "final_states.npz"
    checkpoint = root / "checkpoint.npz"
    if final.exists():
        candidates.append(_density_pair_from_npz(final, "final_states"))
    for path in _snapshot_paths(root):
        candidates.append(_density_pair_from_npz(path, "snapshot"))
    if checkpoint.exists():
        candidates.append(_density_pair_from_npz(checkpoint, "checkpoint"))
    if not candidates:
        raise FileNotFoundError(
            f"no density pair found below {root}; expected final_states.npz, "
            "checkpoint.npz, or snapshots/snapshot_*.npz"
        )
    priority = {"checkpoint": 0, "snapshot": 1, "final_states": 2}
    return max(candidates, key=lambda item: (item.time_s, priority[item.kind]))


def _validate_history(
    time_s: Iterable[float],
    baseline: Iterable[float],
    coupled: Iterable[float],
    *,
    path: Path,
    kind: str,
    column_control: Iterable[float] | None = None,
) -> GrowthHistory:
    time = np.asarray(tuple(time_s), dtype=np.float64)
    baseline_array = np.asarray(tuple(baseline), dtype=np.float64)
    coupled_array = np.asarray(tuple(coupled), dtype=np.float64)
    column_array = (
        np.asarray(tuple(column_control), dtype=np.float64)
        if column_control is not None
        else None
    )
    if time.ndim != 1 or time.size == 0:
        raise ValueError(f"{path} contains no growth history")
    if baseline_array.shape != time.shape or coupled_array.shape != time.shape:
        raise ValueError(f"growth arrays in {path} have inconsistent lengths")
    if column_array is not None and column_array.shape != time.shape:
        raise ValueError(f"growth arrays in {path} have inconsistent lengths")
    if not (
        np.all(np.isfinite(time))
        and np.all(np.isfinite(baseline_array))
        and np.all(np.isfinite(coupled_array))
    ):
        raise ValueError(f"growth history in {path} contains non-finite values")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"growth times in {path} are not strictly increasing")
    if np.any(baseline_array <= 0.0) or np.any(coupled_array <= 0.0):
        raise ValueError(f"growth amplitudes in {path} must be positive")
    if column_array is not None and (
        not np.all(np.isfinite(column_array)) or np.any(column_array <= 0.0)
    ):
        raise ValueError(f"growth amplitudes in {path} must be positive")
    return GrowthHistory(
        time,
        baseline_array,
        coupled_array,
        path.resolve(),
        kind,
        column_array,
    )


def _history_from_checkpoint(path: Path) -> GrowthHistory:
    with np.load(path, allow_pickle=False) as payload:
        return _validate_history(
            np.asarray(payload["times_s"], dtype=np.float64),
            np.asarray(payload["baseline_amplitudes_m3"], dtype=np.float64),
            np.asarray(payload["coupled_amplitudes_m3"], dtype=np.float64),
            path=path,
            kind="checkpoint",
            column_control=(
                np.asarray(
                    payload["column_control_amplitudes_m3"], dtype=np.float64
                )
                if "column_control_amplitudes_m3" in payload.files
                else None
            ),
        )


def _history_from_diagnostics(path: Path) -> GrowthHistory:
    records = _read_jsonl(path)
    return _validate_history(
        (record["time_s"] for record in records),
        (record["baseline_amplitude_m3"] for record in records),
        (record["coupled_amplitude_m3"] for record in records),
        path=path,
        kind="diagnostics",
        column_control=(
            (record["column_control_amplitude_m3"] for record in records)
            if records and all(
                "column_control_amplitude_m3" in record for record in records
            )
            else None
        ),
    )


def _history_from_snapshots(root: Path) -> GrowthHistory:
    pairs = [_density_pair_from_npz(path, "snapshot") for path in _snapshot_paths(root)]
    pairs.sort(key=lambda item: item.time_s)
    return _validate_history(
        (pair.time_s for pair in pairs),
        (density_mode_amplitude(pair.baseline_m3) for pair in pairs),
        (density_mode_amplitude(pair.coupled_m3) for pair in pairs),
        path=(root / "snapshots"),
        kind="snapshots",
        column_control=(
            (
                density_mode_amplitude(pair.column_control_m3)
                for pair in pairs
            )
            if pairs and all(pair.column_control_m3 is not None for pair in pairs)
            else None
        ),
    )


def load_growth_history(root: str | Path) -> GrowthHistory:
    """Load the densest authoritative growth history available."""

    root = Path(root)
    checkpoint = root / "checkpoint.npz"
    diagnostics = root / "diagnostics.jsonl"
    errors: list[str] = []
    if checkpoint.exists():
        try:
            return _history_from_checkpoint(checkpoint)
        except (KeyError, ValueError) as error:
            errors.append(f"checkpoint: {error}")
    if diagnostics.exists():
        try:
            return _history_from_diagnostics(diagnostics)
        except (KeyError, ValueError) as error:
            errors.append(f"diagnostics: {error}")
    if _snapshot_paths(root):
        try:
            return _history_from_snapshots(root)
        except (KeyError, ValueError) as error:
            errors.append(f"snapshots: {error}")
    detail = "; ".join(errors) if errors else "no history files are present"
    raise ValueError(f"cannot reconstruct growth history below {root}: {detail}")


def _coordinates(report: dict[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    li = report.get("li_config", {})
    ny, nx = shape
    required = ("x_min_km", "x_max_km", "y_min_km", "y_max_km")
    if not all(key in li for key in required):
        raise ValueError("em_growth_report.json lacks Li grid extents")
    if int(li.get("nx", nx)) != nx or int(li.get("ny", ny)) != ny:
        raise ValueError("density shape is inconsistent with the archived Li grid")
    x_native = np.linspace(float(li["x_min_km"]), float(li["x_max_km"]), nx)
    if report.get("em_config", {}).get("periodic_x_layout") == "unique_periodic_cells":
        dx = float(li["dx_km"])
        if not np.isclose(float(li["x_max_km"]) - float(li["x_min_km"]), (nx - 1) * dx):
            raise ValueError("unique periodic stored-coordinate bounds disagree with grid")
        x_native = float(li["x_min_km"]) + np.arange(nx) * dx
        x_km = x_native - (float(li["x_min_km"]) + 0.5 * nx * dx)
    else:
        x_km = x_native - 0.5 * (x_native[0] + x_native[-1])
    y_km = np.linspace(float(li["y_min_km"]), float(li["y_max_km"]), ny)
    return x_km, y_km


def _fit_log_growth(time: np.ndarray, amplitude: np.ndarray, start_s: float) -> dict[str, Any]:
    selected = (time >= start_s) & (amplitude > 0.0)
    count = int(np.count_nonzero(selected))
    if count < 2:
        return {
            "fit_start_s": float(start_s),
            "sample_count": count,
            "rate_s_inv": None,
            "doubling_time_s": None,
            "r_squared": None,
        }
    fit_time = time[selected]
    logarithm = np.log(amplitude[selected])
    slope, intercept = np.polyfit(fit_time, logarithm, 1)
    predicted = slope * fit_time + intercept
    residual_sum = float(np.sum((logarithm - predicted) ** 2))
    total_sum = float(np.sum((logarithm - np.mean(logarithm)) ** 2))
    r_squared = 1.0 if total_sum == 0.0 and residual_sum == 0.0 else (
        float(1.0 - residual_sum / total_sum) if total_sum > 0.0 else None
    )
    doubling = float(np.log(2.0) / slope) if slope > 0.0 else None
    return {
        "fit_start_s": float(start_s),
        "sample_count": count,
        "rate_s_inv": float(slope),
        "doubling_time_s": doubling,
        "r_squared": r_squared,
    }


def _instantaneous_rate(time: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    if time.size < 2:
        return np.full(time.shape, np.nan, dtype=np.float64)
    edge_order = 2 if time.size >= 3 else 1
    return np.gradient(np.log(amplitude), time, edge_order=edge_order)


def _diagnostic_health(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "record_count": 0}
    records = _read_jsonl(path)
    if not records:
        return {"available": True, "record_count": 0, "sha256": _sha256(path)}

    def finite_values(key: str) -> np.ndarray:
        return np.asarray(
            [record[key] for record in records if key in record], dtype=np.float64
        )

    def optional_extreme(key: str, operation) -> float | None:
        values = finite_values(key)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return None
        return float(operation(values))

    return {
        "available": True,
        "record_count": len(records),
        "sha256": _sha256(path),
        "first_time_s": optional_extreme("time_s", np.min),
        "last_time_s": optional_extreme("time_s", np.max),
        "minimum_dt_s": optional_extreme("dt_s", np.min),
        "maximum_dt_s": optional_extreme("dt_s", np.max),
        "maximum_baseline_cfl": optional_extreme("baseline_cfl", np.max),
        "maximum_coupled_cfl": optional_extreme("coupled_cfl", np.max),
        "maximum_coupling_relative_residual": optional_extreme(
            "coupling_relative_residual", np.max
        ),
        "minimum_baseline_density_m3": optional_extreme(
            "minimum_baseline_density_m3", np.min
        ),
        "minimum_coupled_density_m3": optional_extreme(
            "minimum_coupled_density_m3", np.min
        ),
        "maximum_absolute_net_d_j_parallel_ds_a_m": (
            optional_extreme("net_d_j_parallel_ds_a_m", lambda value: np.max(np.abs(value)))
        ),
        "maximum_relative_power_pairing_defect": optional_extreme(
            "relative_power_pairing_defect", np.max
        ),
    }


def _density_metrics(pair: DensityPair) -> dict[str, float]:
    baseline = pair.baseline_m3
    coupled = pair.coupled_m3
    difference = coupled - baseline
    baseline_norm = max(float(np.linalg.norm(baseline)), np.finfo(float).tiny)
    baseline_std = float(np.std(baseline))
    coupled_std = float(np.std(coupled))
    correlation = float(np.corrcoef(baseline.ravel(), coupled.ravel())[0, 1])
    if not np.isfinite(correlation):
        correlation = 1.0 if np.array_equal(baseline, coupled) else 0.0
    metrics = {
        "baseline_minimum_m3": float(np.min(baseline)),
        "baseline_maximum_m3": float(np.max(baseline)),
        "coupled_minimum_m3": float(np.min(coupled)),
        "coupled_maximum_m3": float(np.max(coupled)),
        "baseline_horizontal_structure_amplitude_m3": density_mode_amplitude(baseline),
        "coupled_horizontal_structure_amplitude_m3": density_mode_amplitude(coupled),
        "mean_difference_m3": float(np.mean(difference)),
        "rms_difference_m3": float(np.sqrt(np.mean(difference * difference))),
        "p99_absolute_difference_m3": float(np.percentile(np.abs(difference), 99.0)),
        "maximum_absolute_difference_m3": float(np.max(np.abs(difference))),
        "relative_l2_difference": float(np.linalg.norm(difference) / baseline_norm),
        "spatial_correlation": correlation,
        "baseline_spatial_standard_deviation_m3": baseline_std,
        "coupled_spatial_standard_deviation_m3": coupled_std,
    }
    if pair.column_control_m3 is not None:
        column = pair.column_control_m3

        def effect(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
            delta = second - first
            norm = max(float(np.linalg.norm(first)), np.finfo(float).tiny)
            correlation = float(
                np.corrcoef(first.ravel(), second.ravel())[0, 1]
            )
            if not np.isfinite(correlation):
                correlation = 1.0 if np.array_equal(first, second) else 0.0
            return {
                "mean_difference_m3": float(np.mean(delta)),
                "rms_difference_m3": float(
                    np.sqrt(np.mean(delta * delta))
                ),
                "maximum_absolute_difference_m3": float(
                    np.max(np.abs(delta))
                ),
                "relative_l2_difference": float(
                    np.linalg.norm(delta) / norm
                ),
                "spatial_correlation": correlation,
            }

        metrics.update(
            {
                "column_control_minimum_m3": float(np.min(column)),
                "column_control_maximum_m3": float(np.max(column)),
                "column_control_horizontal_structure_amplitude_m3": (
                    density_mode_amplitude(column)
                ),
                "separated_effects": {
                    "geometry_column_es_minus_local_es": effect(
                        baseline, column
                    ),
                    "electromagnetic_column_em_minus_column_es": effect(
                        column, coupled
                    ),
                    "total_column_em_minus_local_es": effect(
                        baseline, coupled
                    ),
                },
            }
        )
    return metrics


def _render_figure(
    pair: DensityPair,
    history: GrowthHistory,
    x_km: np.ndarray,
    y_km: np.ndarray,
    baseline_rate: np.ndarray,
    coupled_rate: np.ndarray,
    output: Path,
) -> dict[str, Path]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(10.2, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 6, height_ratios=(1.2, 0.8))
    density_axes = [figure.add_subplot(grid[0, start : start + 2]) for start in (0, 2, 4)]
    amplitude_axis = figure.add_subplot(grid[1, 0:3])
    rate_axis = figure.add_subplot(grid[1, 3:6])

    baseline_scaled = pair.baseline_m3 / DISPLAY_SCALE_M3
    coupled_scaled = pair.coupled_m3 / DISPLAY_SCALE_M3
    column_scaled = (
        pair.column_control_m3 / DISPLAY_SCALE_M3
        if pair.column_control_m3 is not None
        else None
    )
    difference_scaled = (pair.coupled_m3 - pair.baseline_m3) / DISPLAY_SCALE_M3
    density_min = float(min(np.min(baseline_scaled), np.min(coupled_scaled)))
    density_max = float(max(np.max(baseline_scaled), np.max(coupled_scaled)))
    if density_max <= density_min:
        density_max = density_min + 1.0
    difference_limit = float(np.max(np.abs(difference_scaled)))
    if difference_limit == 0.0:
        difference_limit = max(1.0e-12, 1.0e-6 * density_max)

    images = []
    if column_scaled is None:
        density_fields = (baseline_scaled, coupled_scaled)
        density_titles = ("(a) Electrostatic baseline", "(b) Reduced-EM")
        density_targets = density_axes[:2]
    else:
        density_fields = (baseline_scaled, column_scaled, coupled_scaled)
        density_titles = (
            "(a) Local electrostatic",
            "(b) Column electrostatic",
            "(c) Column reduced-EM",
        )
        density_targets = density_axes
    for axis, field, title in zip(
        density_targets,
        density_fields,
        density_titles,
    ):
        images.append(
            axis.pcolormesh(
                x_km,
                y_km,
                field,
                cmap="plasma",
                vmin=density_min,
                vmax=density_max,
                shading="nearest",
                rasterized=True,
            )
        )
        axis.set_title(title)
        axis.set_xlabel("East–West (km)")
    difference_image = None
    if column_scaled is None:
        difference_image = density_axes[2].pcolormesh(
            x_km,
            y_km,
            difference_scaled,
            cmap="RdBu_r",
            vmin=-difference_limit,
            vmax=difference_limit,
            shading="nearest",
            rasterized=True,
        )
        density_axes[2].set_title("(c) Reduced-EM − baseline")
        density_axes[2].set_xlabel("East–West (km)")
    density_axes[0].set_ylabel("Altitude (km)")
    for axis in density_axes[1:]:
        axis.tick_params(labelleft=False)
    density_colorbar = figure.colorbar(
        images[0], ax=density_targets, shrink=0.87, pad=0.02
    )
    density_colorbar.set_label(r"Density ($10^5$ cm$^{-3}$)")
    if difference_image is not None:
        difference_colorbar = figure.colorbar(
            difference_image, ax=density_axes[2], shrink=0.87, pad=0.02
        )
        difference_colorbar.set_label(r"$\Delta n$ ($10^5$ cm$^{-3}$)")

    amplitude_axis.semilogy(
        history.time_s,
        history.baseline_amplitude_m3,
        color="#444444",
        linewidth=1.2,
        label="Electrostatic baseline",
    )
    amplitude_axis.semilogy(
        history.time_s,
        history.coupled_amplitude_m3,
        color="#0072B2",
        linewidth=1.2,
        label="Reduced-EM",
    )
    if history.column_control_amplitude_m3 is not None:
        amplitude_axis.semilogy(
            history.time_s,
            history.column_control_amplitude_m3,
            color="#D55E00",
            linewidth=1.2,
            label="Column electrostatic",
        )
    amplitude_axis.set_title("(d) Horizontal-structure amplitude")
    amplitude_axis.set_xlabel("Time (s)")
    amplitude_axis.set_ylabel(r"RMS amplitude (m$^{-3}$)")
    amplitude_axis.legend(frameon=False, fontsize=7.2)
    amplitude_axis.grid(alpha=0.2, linewidth=0.5)

    rate_axis.plot(
        history.time_s,
        baseline_rate,
        color="#444444",
        linewidth=1.0,
        label="Electrostatic baseline",
    )
    rate_axis.plot(
        history.time_s,
        coupled_rate,
        color="#0072B2",
        linewidth=1.0,
        label="Reduced-EM",
    )
    if history.column_control_amplitude_m3 is not None:
        rate_axis.plot(
            history.time_s,
            _instantaneous_rate(
                history.time_s, history.column_control_amplitude_m3
            ),
            color="#D55E00",
            linewidth=1.0,
            label="Column electrostatic",
        )
    rate_axis.axhline(0.0, color="0.65", linewidth=0.6)
    rate_axis.set_title("(e) Instantaneous logarithmic growth rate")
    rate_axis.set_xlabel("Time (s)")
    rate_axis.set_ylabel(r"$d\ln A/dt$ (s$^{-1}$)")
    rate_axis.grid(alpha=0.2, linewidth=0.5)
    rate_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    figure.suptitle(f"Paired ESF growth at t = {pair.time_s:g} s", fontsize=10.0)

    paths = {
        "svg": output / "em_growth_comparison.svg",
        "pdf": output / "em_growth_comparison.pdf",
        "png": output / "em_growth_comparison.png",
    }
    figure.savefig(paths["svg"], metadata={"Creator": "esf_sim reduced-EM postprocessor"})
    figure.savefig(paths["pdf"], metadata={"Creator": "esf_sim reduced-EM postprocessor"})
    figure.savefig(paths["png"], dpi=300)
    plt.close(figure)
    return paths


def postprocess_em_growth(
    run_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
    growth_fit_start_s: float | None = None,
) -> dict[str, Any]:
    """Create a comparison plate and JSON metrics from one growth run."""

    root = Path(run_directory)
    report_path = root / "em_growth_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing archived run report: {report_path}")
    archived_report = _read_json(report_path)
    pair = load_latest_density_pair(root)
    history = load_growth_history(root)
    if (pair.column_control_m3 is None) != (
        history.column_control_amplitude_m3 is None
    ):
        raise ValueError(
            "selected density state and growth history disagree on three-track schema"
        )
    x_km, y_km = _coordinates(archived_report, pair.baseline_m3.shape)
    if history.time_s[-1] > pair.time_s + 1.0e-9:
        raise ValueError("growth history extends beyond the selected density pair")

    fit_start = growth_fit_start_s
    if fit_start is None:
        fit_start = float(
            archived_report.get("em_config", {}).get("growth_fit_start_s", 0.0)
        )
    if not np.isfinite(fit_start) or fit_start < 0.0 or fit_start > pair.time_s:
        raise ValueError("growth_fit_start_s must lie within the archived trajectory")

    baseline_rate = _instantaneous_rate(
        history.time_s, history.baseline_amplitude_m3
    )
    coupled_rate = _instantaneous_rate(
        history.time_s, history.coupled_amplitude_m3
    )
    column_control_rate = (
        _instantaneous_rate(
            history.time_s, history.column_control_amplitude_m3
        )
        if history.column_control_amplitude_m3 is not None
        else None
    )
    baseline_fit = _fit_log_growth(
        history.time_s, history.baseline_amplitude_m3, fit_start
    )
    coupled_fit = _fit_log_growth(
        history.time_s, history.coupled_amplitude_m3, fit_start
    )
    column_control_fit = (
        _fit_log_growth(
            history.time_s,
            history.column_control_amplitude_m3,
            fit_start,
        )
        if history.column_control_amplitude_m3 is not None
        else None
    )

    output = Path(output_directory) if output_directory is not None else root / "postprocess"
    output.mkdir(parents=True, exist_ok=True)
    figure_paths = _render_figure(
        pair,
        history,
        x_km,
        y_km,
        baseline_rate,
        coupled_rate,
        output,
    )
    source_path = output / "em_growth_source_data.npz"
    source_payload: dict[str, Any] = dict(
        x_km=x_km,
        y_km=y_km,
        final_time_s=np.float64(pair.time_s),
        baseline_density_m3=pair.baseline_m3,
        coupled_density_m3=pair.coupled_m3,
        density_difference_m3=pair.coupled_m3 - pair.baseline_m3,
        history_time_s=history.time_s,
        baseline_amplitude_m3=history.baseline_amplitude_m3,
        coupled_amplitude_m3=history.coupled_amplitude_m3,
        baseline_instantaneous_growth_rate_s_inv=baseline_rate,
        coupled_instantaneous_growth_rate_s_inv=coupled_rate,
    )
    if pair.column_control_m3 is not None:
        source_payload.update(
            {
                "column_control_density_m3": pair.column_control_m3,
                "geometry_density_effect_m3": (
                    pair.column_control_m3 - pair.baseline_m3
                ),
                "electromagnetic_density_effect_m3": (
                    pair.coupled_m3 - pair.column_control_m3
                ),
                "column_control_amplitude_m3": (
                    history.column_control_amplitude_m3
                ),
                "column_control_instantaneous_growth_rate_s_inv": (
                    column_control_rate
                ),
            }
        )
    np.savez_compressed(source_path, **source_payload)

    diagnostic_path = root / "diagnostics.jsonl"
    density_metrics = _density_metrics(pair)
    rate_difference = None
    if baseline_fit["rate_s_inv"] is not None and coupled_fit["rate_s_inv"] is not None:
        rate_difference = float(coupled_fit["rate_s_inv"] - baseline_fit["rate_s_inv"])
    growth_effects = None
    if (
        column_control_fit is not None
        and baseline_fit["rate_s_inv"] is not None
        and column_control_fit["rate_s_inv"] is not None
        and coupled_fit["rate_s_inv"] is not None
    ):
        growth_effects = {
            "geometry_column_es_minus_local_es_s_inv": float(
                column_control_fit["rate_s_inv"] - baseline_fit["rate_s_inv"]
            ),
            "electromagnetic_column_em_minus_column_es_s_inv": float(
                coupled_fit["rate_s_inv"] - column_control_fit["rate_s_inv"]
            ),
            "total_column_em_minus_local_es_s_inv": rate_difference,
        }
    completed_time = archived_report.get("completed_time_s")
    time_matches_report = completed_time is None or bool(
        np.isclose(float(completed_time), pair.time_s, rtol=0.0, atol=1.0e-9)
    )
    metrics: dict[str, Any] = {
        "format": (
            "esf_em_growth_postprocess_v2"
            if pair.column_control_m3 is not None
            else "esf_em_growth_postprocess_v1"
        ),
        "experiment_label": archived_report.get("experiment_label", root.name),
        "configuration_hash": archived_report.get("configuration_hash"),
        "scope": {
            "formal_reference_data_used": False,
            "paired_initial_condition_comparison": bool(
                archived_report.get("initial_density_array_equal", False)
            ),
            "growth_amplitude_definition": (
                "RMS of density after removing each altitude's horizontal mean"
            ),
            "instantaneous_growth_rate_definition": "finite-difference d(ln amplitude)/dt",
            "fit_definition": "ordinary least squares of ln(amplitude) versus time",
        },
        "selected_state": {
            "time_s": pair.time_s,
            "kind": pair.kind,
            "path": str(pair.path),
            "sha256": _sha256(pair.path),
            "shape": list(pair.baseline_m3.shape),
            "time_matches_archived_completed_time": time_matches_report,
        },
        "growth_history": {
            "kind": history.kind,
            "path": str(history.path),
            "sha256": _sha256(history.path) if history.path.is_file() else None,
            "sample_count": int(history.time_s.size),
            "first_time_s": float(history.time_s[0]),
            "last_time_s": float(history.time_s[-1]),
        },
        "density_comparison": density_metrics,
        "growth_fit": {
            "baseline": baseline_fit,
            "coupled": coupled_fit,
            "coupled_minus_baseline_rate_s_inv": rate_difference,
            **(
                {
                    "column_control": column_control_fit,
                    "separated_effects": growth_effects,
                }
                if column_control_fit is not None
                else {}
            ),
        },
        "numerical_health": _diagnostic_health(diagnostic_path),
        "input_report": {
            "path": str(report_path.resolve()),
            "sha256": _sha256(report_path),
            "archived_format": archived_report.get("format"),
            "scope": archived_report.get("scope", {}),
        },
        "outputs": {
            "figures": {key: str(path.resolve()) for key, path in figure_paths.items()},
            "source_data": str(source_path.resolve()),
            "metrics": str((output / "em_growth_metrics.json").resolve()),
        },
    }
    if not time_matches_report:
        raise ValueError(
            "latest accepted density time is inconsistent with completed_time_s in "
            "em_growth_report.json"
        )
    _atomic_json(output / "em_growth_metrics.json", metrics)
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an auditable ES-versus-reduced-EM growth comparison"
    )
    parser.add_argument("run_directory", help="completed em_growth output directory")
    parser.add_argument("--output", help="optional post-processing output directory")
    parser.add_argument("--fit-start", type=float, help="override growth-fit start time (s)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics = postprocess_em_growth(
        args.run_directory,
        output_directory=args.output,
        growth_fit_start_s=args.fit_start,
    )
    print(
        json.dumps(
            {
                "metrics": metrics["outputs"]["metrics"],
                "figures": metrics["outputs"]["figures"],
                "completed_time_s": metrics["selected_state"]["time_s"],
                "growth_rate_difference_s_inv": metrics["growth_fit"][
                    "coupled_minus_baseline_rate_s_inv"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DensityPair",
    "GrowthHistory",
    "load_growth_history",
    "load_latest_density_pair",
    "postprocess_em_growth",
]
