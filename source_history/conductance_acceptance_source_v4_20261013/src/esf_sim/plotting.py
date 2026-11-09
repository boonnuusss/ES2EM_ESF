"""Journal-style rendering of the independently simulated Figure 2."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCHEME_ORDER = ("first_order", "van_leer", "pdm7")
PANEL_TITLES = {
    "first_order": "(a) First-order upwind",
    "van_leer": "(b) Second-order van Leer",
    "pdm7": r"(c) Seventh-order PDM ($\alpha=0.2$)",
}


def _load_snapshot(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path) as saved:
        return (
            np.asarray(saved["density"], dtype=np.float64),
            np.asarray(saved["x_km"], dtype=np.float64),
            np.asarray(saved["y_km"], dtype=np.float64),
            float(saved["time_s"]),
        )


def create_figure2(
    snapshots: Mapping[str, str | Path],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Create editable vector and 600-dpi raster Figure 2 deliverables."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 4.55),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    image = None
    for axis, scheme in zip(axes, SCHEME_ORDER):
        density, x_km, y_km, time_s = _load_snapshot(snapshots[scheme])
        scaled_density = density / 1.0e11
        image = axis.pcolormesh(
            x_km,
            y_km,
            scaled_density,
            cmap="plasma",
            vmin=0.0,
            vmax=11.0,
            shading="nearest",
            rasterized=True,
        )
        axis.set_title(PANEL_TITLES[scheme], pad=5.0, fontsize=8.2)
        axis.set_xlabel("x (km)")
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
        axis.text(
            0.97,
            0.02,
            f"t = {time_s:.0f} s",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="white",
            fontsize=7.0,
        )
    axes[0].set_ylabel("Altitude (km)")
    figure.subplots_adjust(left=0.085, right=0.885, bottom=0.115, top=0.925, wspace=0.13)
    assert image is not None
    colorbar_axis = figure.add_axes((0.905, 0.115, 0.018, 0.81))
    colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=np.arange(0, 12, 1))
    colorbar.set_label(r"Electron density ($10^5$ cm$^{-3}$)", labelpad=6.0)

    paths = {
        "svg": output / "figure2_reproduction.svg",
        "pdf": output / "figure2_reproduction.pdf",
        "png": output / "figure2_reproduction.png",
        "tiff": output / "figure2_reproduction.tiff",
    }
    figure.savefig(paths["svg"], metadata={"Creator": "esf_sim independent Python model"})
    figure.savefig(paths["pdf"], metadata={"Creator": "esf_sim independent Python model"})
    figure.savefig(paths["png"], dpi=600)
    figure.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    return paths
