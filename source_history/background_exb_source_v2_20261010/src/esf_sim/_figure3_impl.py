"""Independent numerical reproduction of Li et al. (2021) Figure 3."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .config import (
    CONSERVATIVE_HARMONIC_OPERATOR,
    DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
    SimulationConfig,
)
from .metrics import compare_with_reference, morphology_metrics
from .model import run_simulation
from .reference import verify_reference_files


@dataclass(frozen=True)
class Figure3Case:
    key: str
    alpha: float
    time_s: float
    reference_variable: str
    panel_label: str


FIGURE3_CASES = (
    Figure3Case("alpha_0p2", 0.2, 3150.0, "n2_02_3150", "a)"),
    Figure3Case("alpha_0p4", 0.4, 3150.0, "n2_04_3150", "b)"),
    Figure3Case("alpha_0p6", 0.6, 3050.0, "n2_06_3050", "c)"),
    Figure3Case("alpha_0p8", 0.8, 2860.0, "n2_08_2860", "d)"),
)


def _load_snapshot(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path) as saved:
        return (
            np.asarray(saved["density"], dtype=np.float64),
            np.asarray(saved["x_km"], dtype=np.float64),
            np.asarray(saved["y_km"], dtype=np.float64),
            float(saved["time_s"]),
        )


def _snapshot_matches(
    path: Path, expected_time_s: float, expected_operator: str,
    expected_boundary: str = DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as saved:
            stored_operator = (
                str(saved["potential_operator"])
                if "potential_operator" in saved.files
                else CONSERVATIVE_HARMONIC_OPERATOR
            )
            stored_boundary = (
                str(saved["potential_boundary_y"])
                if "potential_boundary_y" in saved.files
                else DIRICHLET_ZERO_POTENTIAL_BOUNDARY
            )
            return bool(
                saved["density"].shape == (401, 201)
                and np.isclose(float(saved["time_s"]), expected_time_s)
                and stored_operator == expected_operator
                and stored_boundary == expected_boundary
            )
    except (OSError, KeyError, ValueError):
        return False


def _archive_config(config: SimulationConfig, path: Path, case: Figure3Case) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload["figure3_case"] = {
        "alpha": case.alpha,
        "snapshot_time_s": case.time_s,
        "reference_variable": case.reference_variable,
    }
    payload["implementation_note"] = (
        "The 650 km upper boundary follows the English paper, its 401-by-1-km "
        "grid, and the published vertical axis."
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_snapshot(source: Path, destination: Path, case: Figure3Case) -> Path:
    density, x_km, y_km, time_s = _load_snapshot(source)
    with np.load(source) as saved:
        potential_operator = (
            str(saved["potential_operator"])
            if "potential_operator" in saved.files
            else CONSERVATIVE_HARMONIC_OPERATOR
        )
        potential_boundary_y = (
            str(saved["potential_boundary_y"])
            if "potential_boundary_y" in saved.files
            else DIRICHLET_ZERO_POTENTIAL_BOUNDARY
        )
    if not np.isclose(time_s, case.time_s):
        raise ValueError(
            f"{source} is at t={time_s:g} s, expected t={case.time_s:g} s"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        density=density,
        x_km=x_km,
        y_km=y_km,
        time_s=time_s,
        scheme="pdm7",
        pdm_alpha=case.alpha,
        potential_operator=potential_operator,
        potential_boundary_y=potential_boundary_y,
    )
    return destination


def prepare_figure3_snapshots(
    base: SimulationConfig,
    *,
    simulate: bool = True,
    resume: bool = True,
) -> dict[str, Path]:
    """Run the four PDM cases and return stable Figure 3 snapshot paths."""

    output = base.output_directory / "figure3"
    snapshots: dict[str, Path] = {}
    for case in FIGURE3_CASES:
        stable = output / "simulations" / f"density_{case.key}.npz"
        if _snapshot_matches(
            stable, case.time_s, base.potential_operator, base.potential_boundary_y
        ):
            snapshots[case.key] = stable
            continue

        if case.alpha == 0.2:
            source = base.output_directory / "simulations" / "density_pdm7.npz"
            if not _snapshot_matches(
                source, case.time_s, base.potential_operator, base.potential_boundary_y
            ):
                run_directory = output / "runs" / case.key
                config = base.with_updates(
                    scheme="pdm7",
                    pdm_alpha=case.alpha,
                    t_end_s=case.time_s,
                    output_dir=str(run_directory),
                )
                _archive_config(config, output / "configs" / f"{case.key}.json", case)
                if simulate:
                    print(
                        f"[esf_sim] running Figure 3 alpha={case.alpha:g} "
                        f"to t={case.time_s:g} s",
                        flush=True,
                    )
                    source = run_simulation(config, resume=resume)
        else:
            run_directory = output / "runs" / case.key
            config = base.with_updates(
                scheme="pdm7",
                pdm_alpha=case.alpha,
                t_end_s=case.time_s,
                output_dir=str(run_directory),
            )
            _archive_config(config, output / "configs" / f"{case.key}.json", case)
            source = run_directory / "simulations" / "density_pdm7.npz"
            if simulate and not _snapshot_matches(
                source, case.time_s, base.potential_operator, base.potential_boundary_y
            ):
                print(
                    f"[esf_sim] running Figure 3 alpha={case.alpha:g} "
                    f"to t={case.time_s:g} s",
                    flush=True,
                )
                source = run_simulation(config, resume=resume)

        if not _snapshot_matches(
                source, case.time_s, base.potential_operator, base.potential_boundary_y
            ):
            raise FileNotFoundError(
                f"missing complete Figure 3 snapshot {source}; run without "
                "--skip-simulation first"
            )
        snapshots[case.key] = _normalize_snapshot(source, stable, case)
    return snapshots


def write_figure3_source_data(
    snapshots: Mapping[str, str | Path], output_path: str | Path
) -> Path:
    densities = []
    x_km = y_km = None
    times = []
    for case in FIGURE3_CASES:
        density, case_x, case_y, time_s = _load_snapshot(snapshots[case.key])
        if x_km is not None and (
            not np.array_equal(case_x, x_km) or not np.array_equal(case_y, y_km)
        ):
            raise ValueError("Figure 3 snapshots do not share a common grid")
        densities.append(density)
        x_km, y_km = case_x, case_y
        times.append(time_s)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        density_m3=np.stack(densities),
        x_km=x_km,
        y_km=y_km,
        pdm_alpha=np.array([case.alpha for case in FIGURE3_CASES]),
        time_s=np.array(times),
        panel=np.array([case.panel_label for case in FIGURE3_CASES]),
    )
    return path


def create_figure3(
    snapshots: Mapping[str, str | Path], output_directory: str | Path
) -> dict[str, Path]:
    """Render the paper-like 2-by-2 panel layout in vector and raster formats."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(6.2, 8.15), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.845,
        bottom=0.075,
        top=0.975,
        wspace=0.08,
        hspace=0.045,
    )
    axes = np.array(
        [
            [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
            [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
        ]
    )
    image = None
    for index, (axis, case) in enumerate(zip(axes.ravel(), FIGURE3_CASES)):
        density, x_km, y_km, _ = _load_snapshot(snapshots[case.key])
        image = axis.pcolormesh(
            x_km,
            y_km,
            density / 1.0e11,
            cmap="plasma",
            vmin=0.0,
            vmax=11.0,
            shading="nearest",
            rasterized=True,
        )
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
        axis.tick_params(top=True, right=True, length=4.0, labelsize=9.0)
        axis.text(
            0.05,
            0.96,
            rf"{case.panel_label}  $\alpha$ = {case.alpha:.1f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10.5,
            fontweight="bold",
        )
        row, column = divmod(index, 2)
        if row == 0:
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel("East-West (km)", fontsize=10.0)
        if column == 1:
            axis.tick_params(labelleft=False)
        else:
            axis.set_ylabel("Altitude (km)", fontsize=10.0)

    assert image is not None
    colorbar_axis = figure.add_axes((0.885, 0.245, 0.044, 0.535))
    colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=np.arange(0, 12, 1))
    colorbar.ax.tick_params(direction="out", length=3.0, labelsize=8.5)
    colorbar.ax.set_title(r"$10^5$/cm$^3$", fontsize=9.0, pad=8.0)

    paths = {
        "svg": output / "figure3_reproduction.svg",
        "pdf": output / "figure3_reproduction.pdf",
        "png": output / "figure3_reproduction.png",
        "tiff": output / "figure3_reproduction.tiff",
    }
    metadata = {"Creator": "esf_sim independent Python model"}
    figure.savefig(paths["svg"], metadata=metadata)
    figure.savefig(paths["pdf"], metadata=metadata)
    figure.savefig(paths["png"], dpi=600)
    figure.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    return paths


def _load_figure3_reference(
    path: str | Path,
) -> dict[str, tuple[np.ndarray, float, str]]:
    raw = loadmat(path)
    result = {}
    for case in FIGURE3_CASES:
        if case.reference_variable not in raw:
            raise KeyError(f"{case.reference_variable!r} is missing from {path}")
        with_ghosts = np.asarray(raw[case.reference_variable], dtype=np.float64)
        if with_ghosts.shape != (209, 409):
            raise ValueError(
                f"{case.reference_variable} has shape {with_ghosts.shape}; "
                "expected (209, 409)"
            )
        result[case.key] = (
            with_ghosts[4:-4, 4:-4].T.copy(),
            case.time_s,
            case.reference_variable,
        )
    return result


def write_figure3_metrics(
    snapshots: Mapping[str, str | Path],
    *,
    reference_directory: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    simulated = {}
    simulated_metrics = {}
    for case in FIGURE3_CASES:
        density, x_km, y_km, time_s = _load_snapshot(snapshots[case.key])
        metric = morphology_metrics(density, x_km, y_km)
        simulated_metrics[case.key] = metric
        simulated[case.key] = {
            "pdm_alpha": case.alpha,
            "time_s": time_s,
            "metrics": metric.to_dict(),
        }

    root = Path(reference_directory)
    references = _load_figure3_reference(root / "numDensity.mat")
    author_reference = {}
    comparisons = {}
    for case in FIGURE3_CASES:
        density, time_s, variable = references[case.key]
        metric = morphology_metrics(
            density,
            np.linspace(-125.0, 125.0, 201),
            np.linspace(250.0, 650.0, 401),
        )
        author_reference[case.key] = {
            "pdm_alpha": case.alpha,
            "time_s": time_s,
            "variable": variable,
            "metrics": metric.to_dict(),
        }
        comparisons[case.key] = compare_with_reference(
            simulated_metrics[case.key], metric
        )

    gradients = [
        simulated_metrics[case.key].edge_gradient_p99_m3_per_km
        for case in FIGURE3_CASES
    ]
    complexities = [
        simulated_metrics[case.key].interface_complexity for case in FIGURE3_CASES
    ]
    fine_scales = [
        simulated_metrics[case.key].fine_scale_laplacian_normalized
        for case in FIGURE3_CASES
    ]
    trend = {
        "edge_gradient_decreases_with_alpha": bool(
            gradients[0] > gradients[1] > gradients[2] > gradients[3]
        ),
        "interface_complexity_decreases_with_alpha": bool(
            complexities[0] > complexities[1] > complexities[2] > complexities[3]
        ),
        "fine_scale_content_decreases_with_alpha": bool(
            fine_scales[0] > fine_scales[1] > fine_scales[2] > fine_scales[3]
        ),
        "alpha_0p8_is_smoother_than_alpha_0p2": bool(
            gradients[3] < gradients[0]
            and complexities[3] < complexities[0]
            and fine_scales[3] < fine_scales[0]
        ),
    }
    all_reference = all(
        all(comparison["criteria"].values())
        for comparison in comparisons.values()
    )
    report: dict[str, object] = {
        "provenance": {
            "simulation": (
                "independent Python solves; author arrays are read only for final "
                "acceptance metrics"
            ),
            "reference_dataset_version": "ScienceDB V2",
            "reference_file_integrity": verify_reference_files(root),
            "snapshot_times_s": {
                case.key: case.time_s for case in FIGURE3_CASES
            },
        },
        "simulated": simulated,
        "author_reference": author_reference,
        "comparison": comparisons,
        "scientific_alpha_trend": trend,
        "all_reference_thresholds_pass": bool(all_reference),
        "all_scientific_alpha_trend_criteria_pass": bool(all(trend.values())),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def reproduce_figure3(
    base: SimulationConfig,
    *,
    simulate: bool = True,
    resume: bool = True,
) -> tuple[dict[str, Path], dict[str, object]]:
    output = base.output_directory / "figure3"
    snapshots = prepare_figure3_snapshots(base, simulate=simulate, resume=resume)
    write_figure3_source_data(snapshots, output / "figure3_source_data.npz")
    figures = create_figure3(snapshots, output)
    report = write_figure3_metrics(
        snapshots,
        reference_directory=Path(base.reference_mat).parent,
        output_path=output / "figure3_metrics.json",
    )
    shutil.copyfile(
        output / "figure3_metrics.json", output / "figure3_acceptance_report.json"
    )
    return figures, report
