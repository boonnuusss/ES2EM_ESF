"""Failure-aware report for the point-centered equation (2.21) A/B test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .config import SimulationConfig
from .metrics import compare_with_reference, morphology_metrics


TARGET_TIME_S = 3150.0
ALPHA = 0.4
REFERENCE_VARIABLE = "n2_04_3150"


def _load_npz_snapshot(path: Path):
    with np.load(path) as saved:
        return (
            np.asarray(saved["density"], dtype=np.float64),
            np.asarray(saved["x_km"], dtype=np.float64),
            np.asarray(saved["y_km"], dtype=np.float64),
            float(saved["time_s"]),
        )


def _load_diagnostics(path: Path) -> list[dict[str, float]]:
    """Return the de-duplicated accepted-step envelope from all recorded paths."""

    unique: dict[tuple[int, float], dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (int(record["step"]), round(float(record["time_s"]), 9))
        unique[key] = record
    return sorted(unique.values(), key=lambda item: (float(item["time_s"]), int(item["step"])))

def load_data(base: SimulationConfig) -> dict[str, object]:
    output = base.output_directory
    a_path = output / "figure3" / "simulations" / "density_alpha_0p4.npz"
    run = output / "figure3_stages" / "centered_dirichlet" / "runs" / "alpha_0p4"
    checkpoint_path = run / "checkpoints" / "pdm7.npz"
    diagnostics_path = run / "simulations" / "pdm7_diagnostics.jsonl"
    a, x, y, time_a = _load_npz_snapshot(a_path)
    with np.load(checkpoint_path) as saved:
        b = np.asarray(saved["density"], dtype=np.float64)
        time_b = float(saved["time_s"])
        checkpoint_dt = float(saved["dt_s"])
        checkpoint_step = int(saved["step"])
    author_raw = np.asarray(loadmat(Path(base.reference_mat))[REFERENCE_VARIABLE], dtype=np.float64)
    author = author_raw[4:-4, 4:-4].T.copy()
    if a.shape != b.shape or a.shape != author.shape:
        raise ValueError("A, B checkpoint, and author arrays do not share a grid")
    return {
        "a": a,
        "b": b,
        "author": author,
        "x_km": x,
        "y_km": y,
        "time_a_s": time_a,
        "time_b_s": time_b,
        "checkpoint_dt_s": checkpoint_dt,
        "checkpoint_step": checkpoint_step,
        "diagnostics": _load_diagnostics(diagnostics_path),
    }


def numerical_health(records: list[dict[str, float]]) -> dict[str, float | int]:
    return {
        "accepted_records": len(records),
        "last_accepted_time_s": max(float(row["time_s"]) for row in records),
        "minimum_accepted_dt_s": min(float(row["dt_s"]) for row in records),
        "maximum_accepted_cfl": max(float(row["cfl"]) for row in records),
        "maximum_trial_speed_m_s": max(
            max(float(row["maximum_speed_x_m_s"]), float(row["maximum_speed_y_m_s"]))
            for row in records
        ),
        "maximum_accepted_potential_relative_residual": max(
            float(row["potential_relative_residual"]) for row in records
        ),
        "maximum_absolute_mass_relative_change": max(
            abs(float(row["mass_relative_change"])) for row in records
        ),
        "minimum_density_m3": min(float(row["minimum_density_m3"]) for row in records),
    }


def write_source_data(data: dict[str, object], output: Path) -> Path:
    records = data["diagnostics"]
    assert isinstance(records, list)
    path = output / "centered_five_point_ab_failure_source_data.npz"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        density_a_t3150_m3=data["a"],
        density_b_checkpoint_m3=data["b"],
        density_author_t3150_m3=data["author"],
        x_km=data["x_km"],
        y_km=data["y_km"],
        time_a_s=float(data["time_a_s"]),
        time_b_checkpoint_s=float(data["time_b_s"]),
        diagnostic_time_s=np.asarray([row["time_s"] for row in records], dtype=float),
        diagnostic_dt_s=np.asarray([row["dt_s"] for row in records], dtype=float),
        diagnostic_cfl=np.asarray([row["cfl"] for row in records], dtype=float),
        diagnostic_maximum_speed_m_s=np.asarray(
            [max(row["maximum_speed_x_m_s"], row["maximum_speed_y_m_s"]) for row in records],
            dtype=float,
        ),
        diagnostic_potential_relative_residual=np.asarray(
            [row["potential_relative_residual"] for row in records], dtype=float
        ),
        diagnostic_mass_relative_change=np.asarray(
            [row["mass_relative_change"] for row in records], dtype=float
        ),
    )
    return path


def write_report(data: dict[str, object], output: Path) -> dict[str, object]:
    x = np.asarray(data["x_km"], dtype=float)
    y = np.asarray(data["y_km"], dtype=float)
    metrics_a = morphology_metrics(np.asarray(data["a"]), x, y)
    metrics_b = morphology_metrics(np.asarray(data["b"]), x, y)
    metrics_author = morphology_metrics(np.asarray(data["author"]), x, y)
    records = data["diagnostics"]
    assert isinstance(records, list)
    report: dict[str, object] = {
        "experiment": "equation_2_21_point_centered_five_point_ab",
        "status": "b_failed_before_target_time",
        "target": {
            "grid_points": [201, 401],
            "pdm_alpha": ALPHA,
            "target_time_s": TARGET_TIME_S,
        },
        "variant_a": {
            "time_s": float(data["time_a_s"]),
            "operator": "conservative -div(a grad psi), harmonic face mean",
            "linear_solver": "CG plus PyAMG",
        },
        "variant_b": {
            "last_persisted_time_s": float(data["time_b_s"]),
            "last_persisted_step": int(data["checkpoint_step"]),
            "last_persisted_dt_s": float(data["checkpoint_dt_s"]),
            "operator": "point-centered five-point discretization of equation (2.21)",
            "linear_solver": "Intel MKL PARDISO",
            "potential_y_boundary": "psi=0, identical to A",
            "time_control": "CFL<=0.25; low-CFL recovery; maximum restored dt=0.015625 s after t=2579.98 s",
            "termination": "stopped after persistent timestep collapse and extreme trial velocities",
        },
        "numerical_health_b_accepted_step_envelope": numerical_health(records),
        "rejected_trial_potential_relative_residual": 4.299e-8,
        "execution_audit": (
            "Two timed-out wrapper calls left child simulations alive and caused interleaved raw JSONL records. "
            "Both processes were stopped. The reported numerical history is a de-duplicated scatter envelope "
            "of accepted steps, not a single path; raw interleaved records are retained for audit. "
            "The density panel uses one complete atomically written checkpoint."
        ),
        "metrics": {
            "a_t3150": metrics_a.to_dict(),
            "b_checkpoint_non_equitemporal": metrics_b.to_dict(),
            "author_t3150": metrics_author.to_dict(),
        },
        "valid_comparison_a_vs_author": compare_with_reference(metrics_a, metrics_author),
        "diagnostic_only_b_checkpoint_vs_author": compare_with_reference(metrics_b, metrics_author),
        "comparison_warning": (
            "B did not reach 3150 s. Its morphology metrics are non-equitemporal "
            "and must not be used to claim improvement over A."
        ),
        "conclusion": (
            "The literal point-centered five-point interpretation of equation (2.21) "
            "does not provide a viable calibration under the otherwise matched setup; "
            "it develops extreme velocity spikes and persistent CFL timestep collapse "
            "before the requested snapshot time."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "centered_five_point_ab_failure_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def create_figure(data: dict[str, object], output: Path) -> dict[str, Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))
    x = np.asarray(data["x_km"])
    y = np.asarray(data["y_km"])
    image_specs = (
        (axes[0, 0], np.asarray(data["a"]), "a  A: current operator, t=3150 s"),
        (
            axes[0, 1],
            np.asarray(data["b"]),
            f"b  B: point-centered, t={float(data['time_b_s']):.2f} s (stopped)",
        ),
        (axes[1, 0], np.asarray(data["author"]), "c  Author reference, t=3150 s"),
    )
    image = None
    for axis, density, title in image_specs:
        image = axis.pcolormesh(
            x,
            y,
            density / 1.0e11,
            cmap="plasma",
            vmin=0.0,
            vmax=11.0,
            shading="nearest",
            rasterized=True,
        )
        axis.set_title(title, loc="left", fontsize=8.0, fontweight="bold")
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
        axis.tick_params(top=True, right=True, length=3.5)
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[1, 0].set_ylabel("Altitude (km)")
    axes[1, 0].set_xlabel("East-West (km)")
    axes[0, 1].set_xlabel("East-West (km)")

    records = data["diagnostics"]
    assert isinstance(records, list)
    time = np.asarray([row["time_s"] for row in records], dtype=float)
    dt = np.asarray([row["dt_s"] for row in records], dtype=float)
    speed = np.asarray(
        [max(row["maximum_speed_x_m_s"], row["maximum_speed_y_m_s"]) for row in records],
        dtype=float,
    )
    history = axes[1, 1]
    speed_axis = history.twinx()
    history.scatter(time, dt, color="#2b6cb0", s=1.2, alpha=0.55, linewidths=0, label="Time step", rasterized=True)
    speed_axis.scatter(time, speed, color="#c53030", s=1.0, alpha=0.35, linewidths=0, label="Maximum speed", rasterized=True)
    history.set_yscale("log")
    speed_axis.set_yscale("log")
    history.set_xlim(0.0, TARGET_TIME_S)
    history.set_xlabel("Simulation time (s)")
    history.set_ylabel("Time step (s)", color="#2b6cb0")
    speed_axis.set_ylabel(r"Maximum speed (m s$^{-1}$)", color="#c53030")
    history.tick_params(axis="y", colors="#2b6cb0")
    speed_axis.tick_params(axis="y", colors="#c53030")
    history.axvline(float(data["time_b_s"]), color="0.25", linestyle="--", linewidth=0.8)
    history.set_title("d  B accepted-step envelope", loc="left", fontsize=8.0, fontweight="bold")
    history.grid(axis="y", color="0.88", linewidth=0.5)

    assert image is not None
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.09, top=0.95, wspace=0.34, hspace=0.20)
    colorbar_axis = fig.add_axes((0.91, 0.57, 0.018, 0.30))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"Density ($10^5$ cm$^{-3}$)")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        ext: output / f"centered_five_point_ab_failure.{ext}"
        for ext in ("svg", "pdf", "png", "tiff")
    }
    metadata = {"Creator": "esf_sim Python point-centered A/B diagnostic"}
    fig.savefig(paths["svg"], metadata=metadata)
    fig.savefig(paths["pdf"], metadata=metadata)
    fig.savefig(paths["png"], dpi=600)
    fig.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return paths


def reproduce(base: SimulationConfig) -> dict[str, object]:
    data = load_data(base)
    output = base.output_directory / "figure3_stages" / "centered_five_point_ab"
    source = write_source_data(data, output)
    report = write_report(data, output)
    figures = create_figure(data, output)
    result = {
        "status": report["status"],
        "last_persisted_time_s": report["variant_b"]["last_persisted_time_s"],
        "figures": {key: str(value) for key, value in figures.items()},
        "source_data": str(source),
        "report": str(output / "centered_five_point_ab_failure_report.json"),
    }
    (output / "centered_five_point_ab_failure_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.centered_ab_failure")
    parser.add_argument("--config", default="configs/figure2.json")
    args = parser.parse_args()
    result = reproduce(SimulationConfig.from_json(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())