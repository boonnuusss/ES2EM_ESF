"""Render the stopped arithmetic/extrapolation/author-index experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .config import SimulationConfig
from .experimental_combination import STAGE_NAME, expand_author_horizontal
from .metrics import morphology_metrics


def _load_snapshot(path: Path):
    with np.load(path) as saved:
        return np.asarray(saved["density"], dtype=np.float64)


def render_stopped_diagnostic(base: SimulationConfig) -> dict[str, object]:
    root = base.output_directory / "figure3_stages" / STAGE_NAME
    checkpoint_path = root / "runs" / "alpha_0p6" / "checkpoints" / "combo.npz"
    with np.load(checkpoint_path) as saved:
        stopped_active = np.asarray(saved["states"][0], dtype=np.float64)
        stopped_time = float(saved["time_s"])
        stopped_dt = float(saved["dt_s"])
        stopped_step = int(saved["step"])
    stopped = expand_author_horizontal(stopped_active)
    stopped_path = root / "simulations" / "density_alpha_0p6_stopped.npz"
    np.savez_compressed(
        stopped_path,
        density=stopped,
        active_density=stopped_active,
        x_km=base.x_coordinates_m / 1.0e3 - 125.0,
        y_km=base.y_coordinates_m / 1.0e3,
        time_s=stopped_time,
        dt_s=stopped_dt,
        step=stopped_step,
    )

    simulation_1900 = _load_snapshot(
        root / "simulations" / "density_alpha_0p6_t1900.npz"
    )
    simulation_2250 = _load_snapshot(
        root / "simulations" / "density_alpha_0p6_t2250.npz"
    )
    author = np.asarray(
        loadmat(Path(base.reference_mat))["n2_06_3050"][4:-4, 4:-4].T,
        dtype=np.float64,
    )
    x = base.x_coordinates_m / 1.0e3 - 125.0
    y = base.y_coordinates_m / 1.0e3

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    panels = [
        (simulation_1900, "Simulation, t = 1,900 s"),
        (simulation_2250, "Simulation, t = 2,250 s"),
        (stopped, f"Stopped simulation, t = {stopped_time:,.1f} s"),
        (author, "Author, t = 3,050 s"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 7.8), sharex=True, sharey=True)
    image = None
    for label, axis, (density, title) in zip("abcd", axes.ravel(), panels):
        image = axis.pcolormesh(
            x,
            y,
            density / 1.0e11,
            shading="auto",
            cmap="plasma",
            vmin=0.0,
            vmax=11.0,
            rasterized=True,
        )
        axis.text(
            0.035,
            0.965,
            f"{label})",
            transform=axis.transAxes,
            va="top",
            color="white",
            fontweight="bold",
            fontsize=9,
        )
        axis.set_title(title, fontsize=8)
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
    for axis in axes[1]:
        axis.set_xlabel("East-West (km)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Altitude (km)")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.035)
    colorbar.set_label(r"Plasma density ($10^5$ cm$^{-3}$)")
    fig.subplots_adjust(
        left=0.11, right=0.88, bottom=0.08, top=0.95, wspace=0.08, hspace=0.12
    )

    output_base = root / "combo_alpha06_stopped_diagnostic"
    outputs = {}
    for extension in ("svg", "pdf", "png", "tiff"):
        path = output_base.with_suffix(f".{extension}")
        options = {"bbox_inches": "tight"}
        if extension in {"png", "tiff"}:
            options["dpi"] = 600
        fig.savefig(path, **options)
        outputs[extension] = str(path)
    plt.close(fig)

    metric_inputs = {
        "simulation_t1900": simulation_1900,
        "simulation_t2250": simulation_2250,
        "simulation_stopped": stopped,
        "author_t3050": author,
    }
    report = {
        "stage": STAGE_NAME,
        "status": "stopped_after_cfl_step_collapse",
        "stopped_time_s": stopped_time,
        "stopped_dt_s": stopped_dt,
        "stopped_step": stopped_step,
        "horizontal_layout_is_inferred": True,
        "metrics": {
            key: morphology_metrics(value, x, y).to_dict()
            for key, value in metric_inputs.items()
        },
        "outputs": outputs,
    }
    (root / "combo_alpha06_stopped_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        root / "combo_alpha06_stopped_source_data.npz",
        x_km=x,
        y_km=y,
        **metric_inputs,
    )
    return report


if __name__ == "__main__":
    config = SimulationConfig.from_json("configs/figure2.json")
    print(json.dumps(render_stopped_diagnostic(config), indent=2))

