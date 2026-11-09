"""Isolated harmonic-versus-arithmetic face-average calibration for Figure 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from . import _model_impl
from .config import SimulationConfig
from .metrics import compare_with_reference, morphology_metrics
from .model import run_simulation
from .potential import PotentialSolver
from .reference import md5sum


STAGE_NAME = "arithmetic_mean_ab"
REFERENCE_VARIABLE = "n2_04_3150"
SNAPSHOT_TIME_S = 3150.0
PDM_ALPHA = 0.4
WIDTH_ALTITUDES_KM = (430.0, 440.0, 450.0, 460.0, 470.0)
WIDTH_THRESHOLD = 0.5


class ArithmeticMeanESFModel(_model_impl.ESFModel):
    """Production model with only the elliptic face average changed."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.potential_solver = PotentialSolver(
            dx=config.dx_m,
            dy=config.dy_m,
            gravity=config.gravity_m_s2,
            collision_frequency=self.collision_frequency,
            density_floor=config.density_floor_m3,
            rtol=config.potential_rtol,
            maxiter=config.potential_maxiter,
            refresh_steps=config.preconditioner_refresh_steps,
            iteration_limit=config.preconditioner_iteration_limit,
            face_average="arithmetic",
        )


def _snapshot_complete(path: Path, shape: tuple[int, int]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as saved:
            return bool(
                saved["density"].shape == shape
                and np.isclose(float(saved["time_s"]), SNAPSHOT_TIME_S)
            )
    except (OSError, KeyError, ValueError):
        return False


def _load_snapshot(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path) as saved:
        return (
            np.asarray(saved["density"], dtype=np.float64),
            np.asarray(saved["x_km"], dtype=np.float64),
            np.asarray(saved["y_km"], dtype=np.float64),
            float(saved["time_s"]),
        )


def _author_density(reference_mat: str | Path) -> np.ndarray:
    raw = np.asarray(loadmat(reference_mat)[REFERENCE_VARIABLE], dtype=np.float64)
    return raw[4:-4, 4:-4].T.copy()


def _field_agreement(simulated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    residual = np.asarray(simulated, dtype=float) - np.asarray(reference, dtype=float)
    reference_range = max(float(np.ptp(reference)), np.finfo(float).tiny)
    centered_sim = simulated.ravel() - float(np.mean(simulated))
    centered_ref = reference.ravel() - float(np.mean(reference))
    denominator = max(
        float(np.linalg.norm(centered_sim) * np.linalg.norm(centered_ref)),
        np.finfo(float).tiny,
    )
    return {
        "normalized_rmse_by_reference_range": float(
            np.sqrt(np.mean(residual**2)) / reference_range
        ),
        "pearson_spatial_correlation": float(
            np.dot(centered_sim, centered_ref) / denominator
        ),
    }


def _diagnostics_summary(path: str | Path) -> dict[str, float | int]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"diagnostics file is empty: {path}")
    return {
        "records": len(records),
        "last_time_s": float(records[-1]["time_s"]),
        "final_dt_s": float(records[-1]["dt_s"]),
        "maximum_cfl": float(max(row["cfl"] for row in records)),
        "maximum_speed_x_m_s": float(
            max(row["maximum_speed_x_m_s"] for row in records)
        ),
        "maximum_speed_y_m_s": float(
            max(row["maximum_speed_y_m_s"] for row in records)
        ),
        "maximum_potential_relative_residual": float(
            max(row["potential_relative_residual"] for row in records)
        ),
        "maximum_absolute_mass_relative_change": float(
            max(abs(row["mass_relative_change"]) for row in records)
        ),
        "minimum_density_m3": float(
            min(row["minimum_density_m3"] for row in records)
        ),
    }


def _linear_crossing(x0: float, x1: float, f0: float, f1: float, level: float) -> float:
    if f1 == f0:
        return 0.5 * (x0 + x1)
    weight = float(np.clip((level - f0) / (f1 - f0), 0.0, 1.0))
    return x0 + weight * (x1 - x0)


def _widest_segment_in_side(
    x: np.ndarray,
    normalized_row: np.ndarray,
    *,
    side: str,
    threshold: float,
) -> float:
    side_mask = x < -5.0 if side == "left" else x > 5.0
    indices = np.where(side_mask)[0]
    if indices.size == 0:
        return float("nan")
    below = normalized_row[indices] < threshold
    padded = np.concatenate(([False], below, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.where(transitions == 1)[0]
    stops = np.where(transitions == -1)[0] - 1
    widths: list[float] = []
    for local_start, local_stop in zip(starts, stops):
        start = int(indices[local_start])
        stop = int(indices[local_stop])
        if start > indices[0]:
            left_crossing = _linear_crossing(
                float(x[start - 1]), float(x[start]),
                float(normalized_row[start - 1]), float(normalized_row[start]),
                threshold,
            )
        else:
            left_crossing = float(x[start] - 0.5 * np.median(np.diff(x)))
        if stop < indices[-1]:
            right_crossing = _linear_crossing(
                float(x[stop]), float(x[stop + 1]),
                float(normalized_row[stop]), float(normalized_row[stop + 1]),
                threshold,
            )
        else:
            right_crossing = float(x[stop] + 0.5 * np.median(np.diff(x)))
        widths.append(max(0.0, right_crossing - left_crossing))
    return max(widths, default=float("nan"))


def arm_width_profile(
    density: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    *,
    threshold: float = WIDTH_THRESHOLD,
) -> dict[str, np.ndarray]:
    left = np.full(y_km.shape, np.nan, dtype=float)
    right = np.full(y_km.shape, np.nan, dtype=float)
    for row in range(y_km.size):
        background = float(np.percentile(density[row], 90.0))
        if background <= 0.0:
            continue
        normalized = density[row] / background
        left[row] = _widest_segment_in_side(
            x_km, normalized, side="left", threshold=threshold
        )
        right[row] = _widest_segment_in_side(
            x_km, normalized, side="right", threshold=threshold
        )
    stack = np.vstack((left, right))
    valid_count = np.sum(np.isfinite(stack), axis=0)
    total = np.nansum(stack, axis=0)
    mean = np.divide(
        total,
        valid_count,
        out=np.full_like(total, np.nan, dtype=float),
        where=valid_count > 0,
    )
    return {"left_km": left, "right_km": right, "mean_km": mean}


def _width_table(
    density: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
) -> dict[str, dict[str, float]]:
    profile = arm_width_profile(density, x_km, y_km)
    table: dict[str, dict[str, float]] = {}
    for altitude in WIDTH_ALTITUDES_KM:
        row = int(np.argmin(np.abs(y_km - altitude)))
        table[f"{float(y_km[row]):.0f}"] = {
            "left_km": float(profile["left_km"][row]),
            "right_km": float(profile["right_km"][row]),
            "mean_km": float(profile["mean_km"][row]),
        }
    return table


def prepare_b_snapshot(
    base: SimulationConfig,
    *,
    resume: bool = True,
) -> tuple[Path, SimulationConfig]:
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    run_root = stage_root / "runs" / "b_arithmetic_mean"
    stable = stage_root / "simulations" / "density_arithmetic_mean_alpha_0p4.npz"
    config = base.with_updates(
        scheme="pdm7",
        pdm_alpha=PDM_ALPHA,
        t_end_s=SNAPSHOT_TIME_S,
        output_dir=str(run_root),
    )
    config.validate()
    if _snapshot_complete(stable, (base.ny, base.nx)):
        return stable, config

    archive = config.to_dict()
    archive["ab_calibration"] = {
        "variant": "B",
        "changed_factor_only": {
            "potential_face_average_A": "harmonic",
            "potential_face_average_B": "arithmetic",
        },
        "unchanged_from_A": [
            "201_by_401_grid",
            "all_201_horizontal_values_active_and_periodic",
            "zero_vertical_potential_boundary",
            "equal_value_vertical_density_extrapolation",
            "CG_plus_PyAMG_and_tolerance",
            "PDM7_reconstruction_and_alpha_0.4",
            "AB3_predictor_corrector_with_SSPRK3_startup",
            "dt_and_CFL_policy",
            "initial_condition_and_physics",
        ],
    }
    config_path = stage_root / "configs" / "b_arithmetic_mean_alpha_0p4.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(archive, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    source = run_root / "simulations" / "density_pdm7.npz"
    if not _snapshot_complete(source, (base.ny, base.nx)):
        original_model = _model_impl.ESFModel
        _model_impl.ESFModel = ArithmeticMeanESFModel
        try:
            source = run_simulation(config, resume=resume)
        finally:
            _model_impl.ESFModel = original_model

    with np.load(source) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload["potential_face_average"] = np.array("arithmetic")
    stable.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stable, **payload)
    return stable, config


def write_report_and_source(
    base: SimulationConfig,
    a_path: str | Path,
    b_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    a, x, y, time_a = _load_snapshot(a_path)
    b, xb, yb, time_b = _load_snapshot(b_path)
    if not (np.array_equal(x, xb) and np.array_equal(y, yb)):
        raise ValueError("A and B grids differ")
    author = _author_density(base.reference_mat)
    metrics_a = morphology_metrics(a, x, y)
    metrics_b = morphology_metrics(b, x, y)
    metrics_author = morphology_metrics(author, x, y)
    widths = {
        "A_harmonic": _width_table(a, x, y),
        "B_arithmetic": _width_table(b, x, y),
        "author_reference": _width_table(author, x, y),
    }
    width_summary: dict[str, Any] = {}
    for altitude in map(lambda value: f"{value:.0f}", WIDTH_ALTITUDES_KM):
        a_width = widths["A_harmonic"][altitude]["mean_km"]
        b_width = widths["B_arithmetic"][altitude]["mean_km"]
        ref_width = widths["author_reference"][altitude]["mean_km"]
        width_summary[altitude] = {
            "B_minus_A_km": b_width - a_width,
            "B_over_A": b_width / max(a_width, np.finfo(float).tiny),
            "A_absolute_error_vs_author_km": abs(a_width - ref_width),
            "B_absolute_error_vs_author_km": abs(b_width - ref_width),
            "B_improves_width_error": bool(abs(b_width - ref_width) < abs(a_width - ref_width)),
        }
    report: dict[str, Any] = {
        "experiment": "isolated harmonic-versus-arithmetic potential-face A/B",
        "grid": [base.nx, base.ny],
        "pdm_alpha": PDM_ALPHA,
        "time_s": SNAPSHOT_TIME_S,
        "reference_variable": REFERENCE_VARIABLE,
        "reference_mat_md5": md5sum(base.reference_mat),
        "changed_factor_only": "face average in conservative five-point potential operator",
        "variants": {
            "A": "harmonic face average; production Figure 3 baseline",
            "B": "arithmetic face average; every other setting retained",
        },
        "times_verified_s": {"A": time_a, "B": time_b},
        "solver_diagnostics": {
            "A_harmonic": _diagnostics_summary(
                base.output_directory
                / "figure3"
                / "runs"
                / "alpha_0p4"
                / "simulations"
                / "pdm7_diagnostics.jsonl"
            ),
            "B_arithmetic": _diagnostics_summary(
                stage_root
                / "runs"
                / "b_arithmetic_mean"
                / "simulations"
                / "pdm7_diagnostics.jsonl"
            ),
        },
        "morphology_metrics": {
            "A_harmonic": metrics_a.to_dict(),
            "B_arithmetic": metrics_b.to_dict(),
            "author_reference": metrics_author.to_dict(),
        },
        "reference_comparison": {
            "A": compare_with_reference(metrics_a, metrics_author),
            "B": compare_with_reference(metrics_b, metrics_author),
        },
        "field_agreement_diagnostic_only": {
            "A": _field_agreement(a, author),
            "B": _field_agreement(b, author),
        },
        "main_arm_width_definition": {
            "threshold": "n / row_p90 < 0.5",
            "side_selection": "widest connected depleted segment on each side of |x|=5 km",
            "reported_value": "mean of left and right widths with linearly interpolated threshold crossings",
        },
        "main_arm_widths_km": widths,
        "width_change_summary": width_summary,
    }
    improvements = [value["B_improves_width_error"] for value in width_summary.values()]
    report["calibration_outcome"] = {
        "B_widens_majority_of_sampled_rows": bool(
            sum(value["B_minus_A_km"] > 0.0 for value in width_summary.values())
            >= 3
        ),
        "B_improves_width_error_majority": bool(sum(improvements) >= 3),
        "B_improves_bubble_top_error": bool(
            report["reference_comparison"]["B"]["bubble_top_absolute_error_km"]
            < report["reference_comparison"]["A"]["bubble_top_absolute_error_km"]
        ),
        "B_improves_spatial_correlation": bool(
            report["field_agreement_diagnostic_only"]["B"]["pearson_spatial_correlation"]
            > report["field_agreement_diagnostic_only"]["A"]["pearson_spatial_correlation"]
        ),
        "B_requires_smaller_final_dt": bool(
            report["solver_diagnostics"]["B_arithmetic"]["final_dt_s"]
            < report["solver_diagnostics"]["A_harmonic"]["final_dt_s"]
        ),
    }
    report_path = stage_root / "arithmetic_mean_ab_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    profiles = {
        label: arm_width_profile(field, x, y)["mean_km"]
        for label, field in (("A", a), ("B", b), ("author", author))
    }
    source_path = stage_root / "arithmetic_mean_ab_source_data.npz"
    np.savez_compressed(
        source_path,
        x_km=x,
        y_km=y,
        density_A_harmonic_m3=a,
        density_B_arithmetic_m3=b,
        density_author_m3=author,
        difference_B_minus_A_m3=b - a,
        arm_width_A_km=profiles["A"],
        arm_width_B_km=profiles["B"],
        arm_width_author_km=profiles["author"],
        width_threshold=WIDTH_THRESHOLD,
        time_s=SNAPSHOT_TIME_S,
        pdm_alpha=PDM_ALPHA,
    )
    return report_path, source_path, report


def create_ab_figure(
    base: SimulationConfig,
    a_path: str | Path,
    b_path: str | Path,
) -> dict[str, Path]:
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    a, x, y, _ = _load_snapshot(a_path)
    b, _, _, _ = _load_snapshot(b_path)
    author = _author_density(base.reference_mat)
    profiles = {
        label: arm_width_profile(field, x, y)["mean_km"]
        for label, field in (("A", a), ("B", b), ("author", author))
    }

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
            "legend.frameon": False,
        }
    )
    figure = plt.figure(figsize=(9.0, 6.8))
    grid = figure.add_gridspec(
        2, 3, left=0.07, right=0.92, bottom=0.09, top=0.95,
        wspace=0.24, hspace=0.28, height_ratios=(1.08, 0.92)
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    lower = [figure.add_subplot(grid[1, index]) for index in range(3)]
    density_image = None
    for axis, field, title in zip(
        axes,
        (a, b, author),
        ("a) A: harmonic mean", "b) B: arithmetic mean", "c) Author reference"),
    ):
        density_image = axis.pcolormesh(
            x, y, field / 1.0e11, shading="nearest", cmap="plasma",
            vmin=0.0, vmax=11.0, rasterized=True,
        )
        axis.set_title(title, fontsize=8.2, pad=3.0)
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
        axis.set_xlabel("East?West (km)")
    axes[0].set_ylabel("Altitude (km)")
    assert density_image is not None
    color_axis = figure.add_axes((0.935, 0.555, 0.018, 0.365))
    colorbar = figure.colorbar(density_image, cax=color_axis)
    colorbar.set_label(r"Density ($10^5$ cm$^{-3}$)")

    difference = (b - a) / 1.0e11
    difference_limit = max(float(np.percentile(np.abs(difference), 99.0)), 0.25)
    difference_image = lower[0].pcolormesh(
        x, y, difference, shading="nearest", cmap="RdBu_r",
        vmin=-difference_limit, vmax=difference_limit, rasterized=True,
    )
    lower[0].set_title("d) B ? A", fontsize=8.2, pad=3.0)
    lower[0].set_xlim(-125.0, 125.0)
    lower[0].set_ylim(250.0, 650.0)
    lower[0].set_xticks([-100, -50, 0, 50, 100])
    lower[0].set_yticks([250, 350, 450, 550, 650])
    lower[0].set_xlabel("East?West (km)")
    lower[0].set_ylabel("Altitude (km)")
    difference_axis = figure.add_axes((0.318, 0.125, 0.012, 0.255))
    difference_colorbar = figure.colorbar(difference_image, cax=difference_axis)
    difference_colorbar.set_label(r"B ? A ($10^5$ cm$^{-3}$)")

    row = int(np.argmin(np.abs(y - 460.0)))
    colors = {"A": "#4C78A8", "B": "#E45756", "author": "#333333"}
    labels = {"A": "A harmonic", "B": "B arithmetic", "author": "Author"}
    for label, field in (("A", a), ("B", b), ("author", author)):
        background = float(np.percentile(field[row], 90.0))
        lower[1].plot(
            x, field[row] / background, lw=1.25, color=colors[label], label=labels[label]
        )
    lower[1].axhline(WIDTH_THRESHOLD, color="#777777", lw=0.8, ls="--")
    lower[1].set_xlim(-125.0, 125.0)
    lower[1].set_ylim(0.0, 1.15)
    lower[1].set_xlabel("East?West (km)")
    lower[1].set_ylabel(r"$n/n_{90}$")
    lower[1].set_title("e) Cross-section at 460 km", fontsize=8.2, pad=3.0)
    lower[1].legend(loc="lower right", fontsize=6.6)

    altitude_mask = (y >= 420.0) & (y <= 480.0)
    for label in ("A", "B", "author"):
        lower[2].plot(
            profiles[label][altitude_mask], y[altitude_mask],
            lw=1.4, color=colors[label], label=labels[label]
        )
    lower[2].set_xlim(left=0.0)
    lower[2].set_ylim(420.0, 480.0)
    lower[2].set_xlabel("Mean main-arm width (km)")
    lower[2].set_ylabel("Altitude (km)")
    lower[2].set_title("f) Main-arm width", fontsize=8.2, pad=3.0)
    lower[2].legend(loc="best", fontsize=6.6)

    base_path = stage_root / "arithmetic_mean_ab_comparison"
    outputs = {
        extension: base_path.with_suffix(f".{extension}")
        for extension in ("svg", "pdf", "png", "tiff")
    }
    metadata = {"Creator": "esf_sim arithmetic-face A/B calibration"}
    figure.savefig(outputs["svg"], metadata=metadata)
    figure.savefig(outputs["pdf"], metadata=metadata)
    figure.savefig(outputs["png"], dpi=600)
    figure.savefig(outputs["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    return outputs


def run_ab(
    base: SimulationConfig,
    *,
    simulate: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    a_path = base.output_directory / "figure3" / "simulations" / "density_alpha_0p4.npz"
    if not _snapshot_complete(a_path, (base.ny, base.nx)):
        raise FileNotFoundError(f"complete harmonic baseline is missing: {a_path}")
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    b_path = stage_root / "simulations" / "density_arithmetic_mean_alpha_0p4.npz"
    if simulate:
        b_path, _ = prepare_b_snapshot(base, resume=resume)
    elif not _snapshot_complete(b_path, (base.ny, base.nx)):
        raise FileNotFoundError(f"complete arithmetic snapshot is missing: {b_path}")
    report_path, source_path, report = write_report_and_source(base, a_path, b_path)
    outputs = create_ab_figure(base, a_path, b_path)
    return {
        "A_snapshot": str(a_path),
        "B_snapshot": str(b_path),
        "report": str(report_path),
        "source_data": str(source_path),
        "figure_outputs": {key: str(value) for key, value in outputs.items()},
        "calibration_outcome": report["calibration_outcome"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.arithmetic_mean_ab")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_ab(
        SimulationConfig.from_json(args.config),
        simulate=not args.skip_simulation,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
