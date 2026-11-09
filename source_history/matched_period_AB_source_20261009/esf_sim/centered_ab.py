"""A/B calibration of the ESF potential discretization for Figure 3 alpha=0.4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .config import SimulationConfig
from . import experimental as experiment
from .metrics import compare_with_reference, morphology_metrics


ALPHA = 0.4
TIME_S = 3150.0
REFERENCE_VARIABLE = "n2_04_3150"


def _load_snapshot(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path) as saved:
        return (
            np.asarray(saved["density"], dtype=np.float64),
            np.asarray(saved["x_km"], dtype=np.float64),
            np.asarray(saved["y_km"], dtype=np.float64),
            float(saved["time_s"]),
        )


def _paths(base: SimulationConfig) -> dict[str, Path]:
    root = base.output_directory
    return {
        "a": root / "figure3" / "simulations" / "density_alpha_0p4.npz",
        "b": root / "figure3_stages" / "centered_dirichlet" / "simulations" / "density_alpha_0p4.npz",
        "output": root / "figure3_stages" / "centered_five_point_ab",
    }


def run_centered_case(base: SimulationConfig, *, resume: bool = True) -> Path:
    """Run B with recoverable CFL control and the point-centered solver."""

    conservative_time_base = base.with_updates(dt_s=0.015625)
    return experiment.run_stage_case(
        conservative_time_base,
        experiment.STAGES["centered_dirichlet"],
        ALPHA,
        resume=resume,
    )


def load_ab_data(base: SimulationConfig) -> dict[str, np.ndarray]:
    paths = _paths(base)
    a, x_a, y_a, t_a = _load_snapshot(paths["a"])
    b, x_b, y_b, t_b = _load_snapshot(paths["b"])
    if a.shape != (401, 201) or b.shape != (401, 201):
        raise ValueError("A and B must both be 401-by-201 arrays")
    if not np.isclose(t_a, TIME_S) or not np.isclose(t_b, TIME_S):
        raise ValueError("A and B must both be snapshots at t=3150 s")
    if not np.array_equal(x_a, x_b) or not np.array_equal(y_a, y_b):
        raise ValueError("A and B coordinates differ")

    raw = loadmat(Path(base.reference_mat))
    reference_with_ghosts = np.asarray(raw[REFERENCE_VARIABLE], dtype=np.float64)
    reference = reference_with_ghosts[4:-4, 4:-4].T.copy()
    if reference.shape != a.shape:
        raise ValueError("author reference shape does not match A/B")
    return {"a": a, "b": b, "reference": reference, "x_km": x_a, "y_km": y_a}


def write_source_data(data: Mapping[str, np.ndarray], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "centered_five_point_ab_source_data.npz"
    np.savez_compressed(
        path,
        density_a_current_m3=data["a"],
        density_b_centered_m3=data["b"],
        density_author_m3=data["reference"],
        difference_b_minus_a_m3=data["b"] - data["a"],
        x_km=data["x_km"],
        y_km=data["y_km"],
        alpha=ALPHA,
        time_s=TIME_S,
    )
    return path


def write_metrics(data: Mapping[str, np.ndarray], output: Path) -> dict[str, object]:
    x = data["x_km"]
    y = data["y_km"]
    metrics = {
        key: morphology_metrics(data[key], x, y)
        for key in ("a", "b", "reference")
    }
    comparisons = {
        "a_vs_author": compare_with_reference(metrics["a"], metrics["reference"]),
        "b_vs_author": compare_with_reference(metrics["b"], metrics["reference"]),
    }
    author_norm = max(float(np.linalg.norm(data["reference"])), np.finfo(float).tiny)
    pixel_errors = {
        "a_relative_l2": float(np.linalg.norm(data["a"] - data["reference"]) / author_norm),
        "b_relative_l2": float(np.linalg.norm(data["b"] - data["reference"]) / author_norm),
        "b_minus_a_relative_l2": float(np.linalg.norm(data["b"] - data["a"]) / max(float(np.linalg.norm(data["a"])), np.finfo(float).tiny)),
    }
    error_keys = (
        "bubble_top_absolute_error_km",
        "depleted_area_relative_error",
        "percentile_1_relative_error",
        "percentile_99_relative_error",
    )
    improvement = {
        key: float(comparisons["a_vs_author"][key] - comparisons["b_vs_author"][key])
        for key in error_keys
    }
    report: dict[str, object] = {
        "experiment": "point_centered_equation_2_21_ab_alpha_0p4",
        "controlled_settings": {
            "grid_points": [201, 401],
            "dx_km": 1.25,
            "dy_km": 1.0,
            "pdm_alpha": ALPHA,
            "snapshot_time_s": TIME_S,
            "potential_y_boundary": "psi=0 at y=250 and 650 km for both A and B",
            "density_y_boundary": "four equal-value ghost layers for both A and B",
            "advection": "PDM7 plus Rusanov for both A and B",
            "time_integration": "AB3 with SSPRK3 startup; B restores dt after sustained low CFL with dt capped at 0.015625 s",
        },
        "variant_a": {
            "operator": "conservative -div(a grad psi), harmonic face mean",
            "linear_solver": "warm-started CG plus PyAMG, rtol=1e-8",
        },
        "variant_b": {
            "operator": "point-centered five-point discretization of equation (2.21)",
            "linear_solver": "Intel MKL PARDISO, symbolic analysis reused, numerical factorization repeated",
        },
        "metrics": {key: value.to_dict() for key, value in metrics.items()},
        "comparison": comparisons,
        "pixel_errors": pixel_errors,
        "error_reduction_a_minus_b": improvement,
        "b_better_by_morphology_metric": {
            key: bool(improvement[key] > 0.0) for key in error_keys
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "centered_five_point_ab_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def create_figure(data: Mapping[str, np.ndarray], output: Path) -> dict[str, Path]:
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
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8), sharex=True, sharey=True)
    x = data["x_km"]
    y = data["y_km"]
    density_panels = (
        (axes[0, 0], data["a"], "a  Current harmonic-conservative"),
        (axes[0, 1], data["b"], "b  Point-centered Eq. (2.21)"),
        (axes[1, 0], data["reference"], "c  Author reference"),
    )
    image = None
    for axis, density, title in density_panels:
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
    difference = (data["b"] - data["a"]) / 1.0e11
    difference_limit = max(0.25, float(np.percentile(np.abs(difference), 99.5)))
    diff_image = axes[1, 1].pcolormesh(
        x,
        y,
        difference,
        cmap="RdBu_r",
        vmin=-difference_limit,
        vmax=difference_limit,
        shading="nearest",
        rasterized=True,
    )
    axes[1, 1].set_title("d  B - A", loc="left", fontsize=8.0, fontweight="bold")
    for row in range(2):
        for column in range(2):
            axis = axes[row, column]
            axis.set_xlim(-125.0, 125.0)
            axis.set_ylim(250.0, 650.0)
            axis.set_xticks([-100, -50, 0, 50, 100])
            axis.set_yticks([250, 350, 450, 550, 650])
            axis.tick_params(top=True, right=True, length=3.5)
            if row == 1:
                axis.set_xlabel("East-West (km)")
            if column == 0:
                axis.set_ylabel("Altitude (km)")
    assert image is not None
    density_bar = fig.colorbar(image, ax=axes[:, 0], fraction=0.035, pad=0.025)
    density_bar.set_label(r"Density ($10^5$ cm$^{-3}$)")
    diff_bar = fig.colorbar(diff_image, ax=axes[:, 1], fraction=0.035, pad=0.025)
    diff_bar.set_label(r"B - A ($10^5$ cm$^{-3}$)")
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.09, top=0.95, wspace=0.20, hspace=0.15)

    output.mkdir(parents=True, exist_ok=True)
    paths = {ext: output / f"centered_five_point_ab.{ext}" for ext in ("svg", "pdf", "png", "tiff")}
    metadata = {"Creator": "esf_sim Python A/B calibration"}
    fig.savefig(paths["svg"], metadata=metadata)
    fig.savefig(paths["pdf"], metadata=metadata)
    fig.savefig(paths["png"], dpi=600)
    fig.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return paths


def reproduce(base: SimulationConfig, *, simulate: bool = True, resume: bool = True) -> dict[str, object]:
    if simulate:
        run_centered_case(base, resume=resume)
    data = load_ab_data(base)
    output = _paths(base)["output"]
    source = write_source_data(data, output)
    report = write_metrics(data, output)
    figures = create_figure(data, output)
    return {
        "figures": {key: str(value) for key, value in figures.items()},
        "source_data": str(source),
        "metrics": str(output / "centered_five_point_ab_metrics.json"),
        "b_better_by_morphology_metric": report["b_better_by_morphology_metric"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.centered_ab")
    parser.add_argument("--config", default="configs/figure2.json")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig.from_json(args.config)
    result = reproduce(
        base,
        simulate=not args.skip_simulation,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())