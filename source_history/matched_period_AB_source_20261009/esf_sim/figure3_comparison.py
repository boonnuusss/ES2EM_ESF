"""Render a direct Figure 3 comparison between this model and the author data.

The comparison is intentionally separate from the simulation drivers.  It reads
the four completed snapshots, preserves the published colour range, and writes
both the plotted source arrays and quantitative agreement metrics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat

from .metrics import compare_with_reference, morphology_metrics
from .reference import verify_reference_files


@dataclass(frozen=True)
class ComparisonCase:
    key: str
    alpha: float
    time_s: float
    reference_variable: str
    simulation_relative_path: str
    expected_operator: str
    recovered: bool = False


CASES = (
    ComparisonCase(
        "alpha_0p2",
        0.2,
        3150.0,
        "n2_02_3150",
        (
            "alpha02_stability_calibration/exponential_conservative/"
            "simulations/density_pdm7.npz"
        ),
        "exponential_conservative",
        True,
    ),
    ComparisonCase(
        "alpha_0p4",
        0.4,
        3150.0,
        "n2_04_3150",
        "simulations/density_alpha_0p4.npz",
        "point_centered_equation_2_21_log_coefficients",
    ),
    ComparisonCase(
        "alpha_0p6",
        0.6,
        3050.0,
        "n2_06_3050",
        "simulations/density_alpha_0p6.npz",
        "point_centered_equation_2_21_log_coefficients",
    ),
    ComparisonCase(
        "alpha_0p8",
        0.8,
        2860.0,
        "n2_08_2860",
        "simulations/density_alpha_0p8.npz",
        "point_centered_equation_2_21_log_coefficients",
    ),
)

DEFAULT_ROOT = Path("output/best_baseline/figure3_v2_timeseries")
DEFAULT_REFERENCE = Path("data/reference/numDensity.mat")
DEFAULT_OUTPUT = DEFAULT_ROOT / "comparison"
DISPLAY_SCALE_M3 = 1.0e11  # 10^5 cm^-3
DISPLAY_MIN = 0.0
DISPLAY_MAX = 11.0


def _scalar(saved: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in saved.files:
        return default
    value = saved[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def load_inputs(
    root: str | Path = DEFAULT_ROOT,
    reference_path: str | Path = DEFAULT_REFERENCE,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Load and validate the four completed simulation/reference pairs."""

    root = Path(root)
    reference_path = Path(reference_path)
    raw_reference = loadmat(reference_path)
    simulations: dict[str, dict[str, Any]] = {}
    references: dict[str, np.ndarray] = {}
    common_x: np.ndarray | None = None
    common_y: np.ndarray | None = None

    for case in CASES:
        path = root / case.simulation_relative_path
        if not path.exists():
            raise FileNotFoundError(f"missing completed Figure 3 snapshot: {path}")
        with np.load(path, allow_pickle=False) as saved:
            density = np.asarray(saved["density"], dtype=np.float64)
            x_km = np.asarray(saved["x_km"], dtype=np.float64)
            y_km = np.asarray(saved["y_km"], dtype=np.float64)
            time_s = float(_scalar(saved, "time_s"))
            operator = str(_scalar(saved, "potential_operator", "unknown"))
            boundary = str(_scalar(saved, "potential_boundary_y", "unknown"))

        if density.shape != (401, 201):
            raise ValueError(f"{path} has density shape {density.shape}, expected (401, 201)")
        if x_km.shape != (201,) or y_km.shape != (401,):
            raise ValueError(f"{path} has an unexpected coordinate shape")
        if not np.isclose(time_s, case.time_s, atol=1.0e-9):
            raise ValueError(f"{path} stores t={time_s:g} s, expected {case.time_s:g} s")
        if operator != case.expected_operator:
            raise ValueError(
                f"{path} uses potential_operator={operator!r}, "
                f"expected {case.expected_operator!r}"
            )
        if boundary != "equal_extrapolation":
            raise ValueError(f"{path} uses potential_boundary_y={boundary!r}")
        if not np.all(np.isfinite(density)) or float(np.min(density)) <= 0.0:
            raise ValueError(f"{path} contains non-finite or non-positive density")
        if common_x is not None and (
            not np.array_equal(x_km, common_x) or not np.array_equal(y_km, common_y)
        ):
            raise ValueError("the four Figure 3 simulations do not share one grid")
        common_x, common_y = x_km, y_km

        if case.reference_variable not in raw_reference:
            raise KeyError(f"{case.reference_variable!r} is absent from {reference_path}")
        with_ghosts = np.asarray(raw_reference[case.reference_variable], dtype=np.float64)
        if with_ghosts.shape != (209, 409):
            raise ValueError(
                f"{case.reference_variable} has shape {with_ghosts.shape}, "
                "expected (209, 409)"
            )
        reference = with_ghosts[4:-4, 4:-4].T.copy()
        if reference.shape != density.shape:
            raise ValueError(f"trimmed {case.reference_variable} has shape {reference.shape}")

        simulations[case.key] = {
            "density": density,
            "x_km": x_km,
            "y_km": y_km,
            "time_s": time_s,
            "potential_operator": operator,
            "potential_boundary_y": boundary,
            "snapshot_path": str(path.resolve()),
        }
        references[case.key] = reference

    return simulations, references


def _field_agreement(simulated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    sim = np.asarray(simulated, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    difference = sim - ref
    reference_range = float(np.ptp(ref))
    return {
        "spatial_correlation": float(np.corrcoef(sim.ravel(), ref.ravel())[0, 1]),
        "rmse_m3": float(np.sqrt(np.mean(difference * difference))),
        "normalized_rmse_by_author_range": float(
            np.sqrt(np.mean(difference * difference))
            / max(reference_range, np.finfo(float).tiny)
        ),
        "relative_l2_error": float(
            np.linalg.norm(difference.ravel())
            / max(float(np.linalg.norm(ref.ravel())), np.finfo(float).tiny)
        ),
        "mean_bias_m3": float(np.mean(difference)),
        "maximum_absolute_difference_m3": float(np.max(np.abs(difference))),
    }


def write_metrics(
    simulations: dict[str, dict[str, Any]],
    references: dict[str, np.ndarray],
    *,
    reference_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write field and morphology agreement with full provenance."""

    cases: dict[str, Any] = {}
    simulated_complexity: list[float] = []
    author_complexity: list[float] = []
    simulated_edge_gradient: list[float] = []
    author_edge_gradient: list[float] = []

    for case in CASES:
        record = simulations[case.key]
        sim_density = record["density"]
        ref_density = references[case.key]
        x_km = record["x_km"]
        y_km = record["y_km"]
        sim_morphology = morphology_metrics(sim_density, x_km, y_km)
        ref_morphology = morphology_metrics(ref_density, x_km, y_km)
        acceptance = compare_with_reference(sim_morphology, ref_morphology)
        simulated_complexity.append(sim_morphology.interface_complexity)
        author_complexity.append(ref_morphology.interface_complexity)
        simulated_edge_gradient.append(sim_morphology.edge_gradient_p99_m3_per_km)
        author_edge_gradient.append(ref_morphology.edge_gradient_p99_m3_per_km)
        cases[case.key] = {
            "alpha": case.alpha,
            "time_s": case.time_s,
            "reference_variable": case.reference_variable,
            "simulation_snapshot": record["snapshot_path"],
            "potential_operator": record["potential_operator"],
            "potential_boundary_y": record["potential_boundary_y"],
            "recovered_trajectory": case.recovered,
            "field_agreement": _field_agreement(sim_density, ref_density),
            "simulation_morphology": sim_morphology.to_dict(),
            "author_morphology": ref_morphology.to_dict(),
            "acceptance": acceptance,
        }

    trend = {
        "simulation_interface_complexity_decreases_with_alpha": bool(
            all(a > b for a, b in zip(simulated_complexity, simulated_complexity[1:]))
        ),
        "author_interface_complexity_decreases_with_alpha": bool(
            all(a > b for a, b in zip(author_complexity, author_complexity[1:]))
        ),
        "simulation_edge_gradient_decreases_with_alpha": bool(
            all(a > b for a, b in zip(simulated_edge_gradient, simulated_edge_gradient[1:]))
        ),
        "author_edge_gradient_decreases_with_alpha": bool(
            all(a > b for a, b in zip(author_edge_gradient, author_edge_gradient[1:]))
        ),
    }
    correlations = [
        cases[case.key]["field_agreement"]["spatial_correlation"] for case in CASES
    ]
    report: dict[str, Any] = {
        "figure_contract": {
            "conclusion": (
                "The simulation reproduces the alpha-dependent transition from "
                "branched secondary structure to a smooth single depletion channel."
            ),
            "archetype": "two-row image comparison plate",
            "panel_map": {
                "top_row": "independent/recovered simulation",
                "bottom_row": "Li et al. (2021) author arrays",
                "columns": [case.key for case in CASES],
            },
            "fixed_display_range_1e5_cm3": [DISPLAY_MIN, DISPLAY_MAX],
        },
        "provenance": {
            "reference_path": str(Path(reference_path).resolve()),
            "reference_file_integrity": verify_reference_files(Path(reference_path).parent),
            "simulation_independence": (
                "Author arrays are used only in the bottom row and for comparison "
                "metrics; they are not inputs to any numerical solve."
            ),
            "alpha_0p2_scope": (
                "The density follows the formal v2 point-centred trajectory through "
                "t=2300 s, then uses the exponential conservative potential operator "
                "to remove the non-physical near-zero elliptic mode."
            ),
        },
        "cases": cases,
        "alpha_trend": trend,
        "summary": {
            "mean_spatial_correlation": float(np.mean(correlations)),
            "minimum_spatial_correlation": float(np.min(correlations)),
            "all_interface_complexity_trends_pass": bool(
                trend["simulation_interface_complexity_decreases_with_alpha"]
                and trend["author_interface_complexity_decreases_with_alpha"]
            ),
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_metrics_csv(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "case",
        "alpha",
        "time_s",
        "spatial_correlation",
        "normalized_rmse_by_author_range",
        "relative_l2_error",
        "bubble_top_simulation_km",
        "bubble_top_author_km",
        "bubble_top_error_km",
        "depleted_area_error_percent",
        "interface_complexity_simulation",
        "interface_complexity_author",
        "recovered_trajectory",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in CASES:
            item = report["cases"][case.key]
            field = item["field_agreement"]
            sim = item["simulation_morphology"]
            author = item["author_morphology"]
            acceptance = item["acceptance"]
            writer.writerow(
                {
                    "case": case.key,
                    "alpha": case.alpha,
                    "time_s": case.time_s,
                    "spatial_correlation": field["spatial_correlation"],
                    "normalized_rmse_by_author_range": field[
                        "normalized_rmse_by_author_range"
                    ],
                    "relative_l2_error": field["relative_l2_error"],
                    "bubble_top_simulation_km": sim["bubble_top_km"],
                    "bubble_top_author_km": author["bubble_top_km"],
                    "bubble_top_error_km": acceptance[
                        "bubble_top_absolute_error_km"
                    ],
                    "depleted_area_error_percent": 100.0
                    * acceptance["depleted_area_relative_error"],
                    "interface_complexity_simulation": sim[
                        "interface_complexity"
                    ],
                    "interface_complexity_author": author["interface_complexity"],
                    "recovered_trajectory": case.recovered,
                }
            )
    return path


def write_source_data(
    simulations: dict[str, dict[str, Any]],
    references: dict[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    first = simulations[CASES[0].key]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        simulation_density_m3=np.stack(
            [simulations[case.key]["density"] for case in CASES]
        ),
        author_density_m3=np.stack([references[case.key] for case in CASES]),
        x_km=first["x_km"],
        y_km=first["y_km"],
        pdm_alpha=np.array([case.alpha for case in CASES], dtype=np.float64),
        time_s=np.array([case.time_s for case in CASES], dtype=np.float64),
        case_key=np.array([case.key for case in CASES]),
        author_variable=np.array([case.reference_variable for case in CASES]),
        potential_operator=np.array(
            [simulations[case.key]["potential_operator"] for case in CASES]
        ),
        recovered_trajectory=np.array([case.recovered for case in CASES]),
        display_units="1e5 cm^-3",
        display_vmin=DISPLAY_MIN,
        display_vmax=DISPLAY_MAX,
    )
    return path


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def create_comparison_figure(
    simulations: dict[str, dict[str, Any]],
    references: dict[str, np.ndarray],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Create the fixed-scale 2-by-4 comparison plate."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _style()
    width_in = 183.0 / 25.4
    height_in = 139.0 / 25.4
    figure = plt.figure(figsize=(width_in, height_in), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        4,
        left=0.105,
        right=0.895,
        bottom=0.135,
        top=0.925,
        wspace=0.075,
        hspace=0.075,
    )
    axes = np.array(
        [[figure.add_subplot(grid[row, column]) for column in range(4)] for row in range(2)]
    )
    panel_labels = tuple("abcdefgh")
    image = None
    for column, case in enumerate(CASES):
        record = simulations[case.key]
        fields = (record["density"], references[case.key])
        for row, density in enumerate(fields):
            axis = axes[row, column]
            image = axis.pcolormesh(
                record["x_km"],
                record["y_km"],
                density / DISPLAY_SCALE_M3,
                cmap="plasma",
                vmin=DISPLAY_MIN,
                vmax=DISPLAY_MAX,
                shading="nearest",
                rasterized=True,
            )
            axis.set_xlim(-125.0, 125.0)
            axis.set_ylim(250.0, 650.0)
            axis.set_box_aspect(1.6)
            axis.set_xticks([-100.0, 0.0, 100.0])
            axis.set_yticks([250.0, 350.0, 450.0, 550.0, 650.0])
            axis.tick_params(
                top=True,
                right=True,
                length=3.0,
                pad=2.0,
                labelsize=7.0,
                labelbottom=row == 1,
                labelleft=column == 0,
            )
            panel_index = row * 4 + column
            axis.text(
                0.045,
                0.965,
                f"{panel_labels[panel_index]})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=8.2,
                fontweight="bold",
            )
            if row == 0:
                marker = "*" if case.recovered else ""
                axis.set_title(
                    rf"$\alpha={case.alpha:.1f}${marker}   $t={case.time_s:g}$ s",
                    fontsize=7.7,
                    pad=3.5,
                )
            if column == 0:
                axis.set_ylabel("Altitude (km)", fontsize=7.8, labelpad=3.0)

    assert image is not None
    figure.supxlabel("East-West (km)", x=0.50, y=0.077, fontsize=8.0)
    figure.text(
        0.018,
        0.715,
        "Simulation",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )
    figure.text(
        0.018,
        0.335,
        "Author data",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )
    colorbar_axis = figure.add_axes((0.915, 0.235, 0.022, 0.575))
    colorbar = figure.colorbar(
        image, cax=colorbar_axis, ticks=np.arange(0.0, 12.0, 1.0)
    )
    colorbar.ax.tick_params(direction="out", length=2.5, labelsize=6.8, pad=2.0)
    colorbar.ax.set_title(r"$10^5$ cm$^{-3}$", fontsize=7.2, pad=5.0)
    figure.text(
        0.105,
        0.018,
        (
            r"* $\alpha=0.2$: formal v2 trajectory to 2300 s; stabilized "
            "conservative potential operator thereafter."
        ),
        ha="left",
        va="bottom",
        fontsize=6.1,
        color="#333333",
    )

    stem = output / "figure3_simulation_vs_author"
    paths = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    metadata = {
        "Title": "Figure 3 simulation versus author data",
        "Creator": "esf_sim Python comparison workflow",
    }
    figure.savefig(paths["svg"], dpi=600, metadata=metadata)
    figure.savefig(paths["pdf"], dpi=600, metadata=metadata)
    figure.savefig(paths["png"], dpi=600)
    figure.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    return paths


def write_caption(output_path: str | Path) -> Path:
    caption = (
        "Figure 3 | Comparison of simulated and published plasma density for four "
        "PDM parameters. The upper row shows the present 201 x 401 simulations and "
        "the lower row shows the corresponding author arrays from Li et al. (2021). "
        "Columns show alpha = 0.2, 0.4, 0.6 and 0.8 at t = 3150, 3150, 3050 and "
        "2860 s, respectively. All panels use the same 0-11 x 10^5 cm^-3 colour "
        "range. The alpha = 0.2 density follows the formal v2 point-centred run "
        "through t = 2300 s and then uses the stabilized exponential-conservative "
        "potential operator; the other three panels use the formal v2 point-centred "
        "operator throughout. Author arrays are used only for comparison."
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")
    return path


def qa_outputs(paths: dict[str, Path], output_path: str | Path) -> dict[str, Any]:
    """Perform non-interactive raster/vector/PDF integrity checks."""

    with Image.open(paths["png"]) as image:
        png_size = image.size
        png_dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
        grayscale = np.asarray(image.convert("L").resize((360, 274)), dtype=np.uint8)
        png_unique_levels = int(np.unique(grayscale).size)
        png_nonwhite_fraction = float(np.mean(grayscale < 250))
    with Image.open(paths["tiff"]) as image:
        tiff_size = image.size
        tiff_dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))

    svg_root = ET.parse(paths["svg"]).getroot()
    svg_text_count = sum(1 for element in svg_root.iter() if element.tag.endswith("text"))
    svg_images = [element for element in svg_root.iter() if element.tag.endswith("image")]
    svg_image_count = len(svg_images)
    density_panel_aspect_ratios = [
        float(element.attrib["height"]) / float(element.attrib["width"])
        for element in svg_images[:8]
    ]

    pdf_page_count: int | None = None
    pdf_width_in: float | None = None
    pdf_height_in: float | None = None
    pdf_error: str | None = None
    try:
        from pypdf import PdfReader

        reader = PdfReader(paths["pdf"])
        pdf_page_count = len(reader.pages)
        page = reader.pages[0]
        pdf_width_in = float(page.mediabox.width) / 72.0
        pdf_height_in = float(page.mediabox.height) / 72.0
    except (ImportError, OSError, ValueError, IndexError) as error:
        pdf_error = f"{type(error).__name__}: {error}"

    expected_size = (int(183.0 / 25.4 * 600), int(139.0 / 25.4 * 600))
    checks = {
        "png_is_600_dpi_size": png_size == expected_size,
        "tiff_matches_png_size": tiff_size == png_size,
        "png_dpi_near_600": all(abs(value - 600.0) < 1.0 for value in png_dpi),
        "tiff_dpi_near_600": all(abs(value - 600.0) < 1.0 for value in tiff_dpi),
        "png_not_blank": png_unique_levels > 100 and png_nonwhite_fraction > 0.50,
        "svg_has_editable_text": svg_text_count >= 25,
        "svg_contains_density_panel_images": svg_image_count >= 8,
        "density_panels_preserve_physical_aspect": bool(
            len(density_panel_aspect_ratios) == 8
            and all(abs(value - 1.6) < 0.01 for value in density_panel_aspect_ratios)
        ),
        "pdf_is_single_page": pdf_page_count == 1,
        "pdf_size_matches_figure": bool(
            pdf_width_in is not None
            and pdf_height_in is not None
            and abs(pdf_width_in - 183.0 / 25.4) < 0.02
            and abs(pdf_height_in - 139.0 / 25.4) < 0.02
        ),
    }
    report = {
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "png": {
            "size_pixels": png_size,
            "dpi": png_dpi,
            "unique_grayscale_levels_in_preview": png_unique_levels,
            "nonwhite_fraction_in_preview": png_nonwhite_fraction,
        },
        "tiff": {"size_pixels": tiff_size, "dpi": tiff_dpi},
        "svg": {
            "editable_text_elements": svg_text_count,
            "image_elements": svg_image_count,
            "density_panel_height_width_ratios": density_panel_aspect_ratios,
        },
        "pdf": {
            "page_count": pdf_page_count,
            "width_in": pdf_width_in,
            "height_in": pdf_height_in,
            "inspection_error": pdf_error,
        },
    }
    path = Path(output_path)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def reproduce_comparison(
    *,
    root: str | Path = DEFAULT_ROOT,
    reference_path: str | Path = DEFAULT_REFERENCE,
    output_directory: str | Path = DEFAULT_OUTPUT,
) -> tuple[dict[str, Path], dict[str, Any]]:
    simulations, references = load_inputs(root, reference_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    source_data = write_source_data(
        simulations, references, output / "figure3_simulation_vs_author_source_data.npz"
    )
    metrics = write_metrics(
        simulations,
        references,
        reference_path=reference_path,
        output_path=output / "figure3_simulation_vs_author_metrics.json",
    )
    metrics_csv = write_metrics_csv(
        metrics, output / "figure3_simulation_vs_author_metrics.csv"
    )
    figures = create_comparison_figure(simulations, references, output)
    caption = write_caption(output / "figure3_simulation_vs_author_caption.txt")
    qa = qa_outputs(figures, output / "figure3_simulation_vs_author_qa.json")
    pdf_delivery = Path("output/pdf/figure3_simulation_vs_author.pdf")
    pdf_delivery.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figures["pdf"], pdf_delivery)
    manifest = {
        "figures": {key: str(path.resolve()) for key, path in figures.items()},
        "pdf_delivery": str(pdf_delivery.resolve()),
        "source_data": str(source_data.resolve()),
        "metrics_json": str(
            (output / "figure3_simulation_vs_author_metrics.json").resolve()
        ),
        "metrics_csv": str(metrics_csv.resolve()),
        "caption": str(caption.resolve()),
        "qa": qa,
    }
    (output / "figure3_simulation_vs_author_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not qa["all_checks_pass"]:
        failed = [key for key, value in qa["checks"].items() if not value]
        raise RuntimeError(f"Figure 3 comparison QA failed: {failed}")
    return figures, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    figures, metrics = reproduce_comparison(
        root=args.root,
        reference_path=args.reference,
        output_directory=args.output,
    )
    for kind, path in figures.items():
        print(f"[figure3-comparison] {kind}: {path}", flush=True)
    print(
        "[figure3-comparison] mean spatial correlation: "
        f"{metrics['summary']['mean_spatial_correlation']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
