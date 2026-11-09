"""Isolated A/B calibration of the paper's horizontal periodic indexing.

Variant A is the production baseline: all 201 stored columns are advanced as
periodic unknowns.  Variant B follows the exact copy relationships observed in
the authors' public ScienceDB arrays: the nominal 201-column interior storage
contains 198 active columns followed by three periodic continuation columns.
The public arrays add another four copied columns on each side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .config import SimulationConfig
from .metrics import compare_with_reference, morphology_metrics
from .model import run_simulation
from .reference import md5sum


ACTIVE_HORIZONTAL_CELLS = 198
DISPLAY_HORIZONTAL_CELLS = 201
INTERNAL_PERIODIC_CONTINUATION = 3
EXTERNAL_GHOST_CELLS = 4
REFERENCE_VARIABLE = "n2_04_3150"
SNAPSHOT_TIME_S = 3150.0
PDM_ALPHA = 0.4
STAGE_NAME = "periodic_index_ab"


def expand_author_interior(active: np.ndarray) -> np.ndarray:
    """Return ``198 active + 3 periodic continuation`` horizontal storage."""

    values = np.asarray(active, dtype=np.float64)
    if values.shape[-1] != ACTIVE_HORIZONTAL_CELLS:
        raise ValueError(
            f"active horizontal size must be {ACTIVE_HORIZONTAL_CELLS}, "
            f"got {values.shape[-1]}"
        )
    continuation = INTERNAL_PERIODIC_CONTINUATION
    return np.concatenate((values, values[..., :continuation]), axis=-1)


def add_four_external_ghosts(active: np.ndarray) -> np.ndarray:
    """Return the exact 209-column layout found in the authors' data."""

    values = np.asarray(active, dtype=np.float64)
    ghost = EXTERNAL_GHOST_CELLS
    continuation = INTERNAL_PERIODIC_CONTINUATION + ghost
    return np.concatenate(
        (values[..., -ghost:], values, values[..., :continuation]), axis=-1
    )


def active_horizontal_coordinates_m(base: SimulationConfig) -> np.ndarray:
    """Coordinates associated with interior indices 0 through 197."""

    start = base.x_min_km * 1.0e3
    return start + np.arange(ACTIVE_HORIZONTAL_CELLS) * base.dx_m


def author_index_evidence(reference_mat: str | Path) -> dict[str, Any]:
    """Verify the periodic copy indices directly in all four grid sizes."""

    raw = loadmat(reference_mat)
    cases = {
        "n1_04_3330": 101,
        "n2_04_3150": 201,
        "n3_04_3000": 301,
        "n4_04_2900": 401,
    }
    report: dict[str, Any] = {}
    for variable, nominal in cases.items():
        stored = np.asarray(raw[variable], dtype=np.float64)
        interior = stored[4:-4, 4:-4]
        active_stop = nominal - INTERNAL_PERIODIC_CONTINUATION
        left_expected = interior[active_stop - EXTERNAL_GHOST_CELLS : active_stop]
        right_expected = interior[
            INTERNAL_PERIODIC_CONTINUATION : INTERNAL_PERIODIC_CONTINUATION
            + EXTERNAL_GHOST_CELLS
        ]
        left_error = float(
            np.max(np.abs(stored[:EXTERNAL_GHOST_CELLS, 4:-4] - left_expected))
        )
        right_error = float(
            np.max(np.abs(stored[-EXTERNAL_GHOST_CELLS:, 4:-4] - right_expected))
        )
        active = interior[:active_stop]
        reconstructed = add_periodic_storage_general(active)
        scale = max(float(np.max(np.abs(interior))), np.finfo(float).tiny)
        continuation_error = float(
            np.max(
                np.abs(
                    interior[active_stop:]
                    - interior[:INTERNAL_PERIODIC_CONTINUATION]
                )
            )
        )
        report[variable] = {
            "stored_shape": list(stored.shape),
            "interior_shape": list(interior.shape),
            "active_horizontal_slice_python": [
                0,
                active_stop,
            ],
            "active_horizontal_count": int(active.shape[0]),
            "external_left_source_interior_slice_python": [
                active_stop - EXTERNAL_GHOST_CELLS,
                active_stop,
            ],
            "external_right_source_interior_slice_python": [
                INTERNAL_PERIODIC_CONTINUATION,
                INTERNAL_PERIODIC_CONTINUATION + EXTERNAL_GHOST_CELLS,
            ],
            "left_copy_max_abs_error_m3": left_error,
            "right_copy_max_abs_error_m3": right_error,
            "three_column_continuation_max_abs_error_m3": continuation_error,
            "three_column_continuation_max_relative_error": continuation_error / scale,
            "full_horizontal_reconstruction_max_abs_error_m3": float(
                np.max(np.abs(reconstructed - stored[:, 4:-4]))
            ),
            "external_copy_relations_exact": bool(
                left_error == 0.0 and right_error == 0.0
            ),
        }
    return report


def add_periodic_storage_general(active: np.ndarray) -> np.ndarray:
    """General form of the author's 3-inner/4-outer horizontal storage."""

    values = np.asarray(active, dtype=np.float64)
    h = INTERNAL_PERIODIC_CONTINUATION
    g = EXTERNAL_GHOST_CELLS
    return np.concatenate((values[-g:], values, values[: h + g]), axis=0)


def b_active_config(base: SimulationConfig, output: str | Path) -> SimulationConfig:
    """Build variant B while retaining every non-indexing numerical setting."""

    x_km = active_horizontal_coordinates_m(base) / 1.0e3
    return base.with_updates(
        scheme="pdm7",
        pdm_alpha=PDM_ALPHA,
        nx=ACTIVE_HORIZONTAL_CELLS,
        x_min_km=float(x_km[0]),
        x_max_km=float(x_km[-1]),
        t_end_s=SNAPSHOT_TIME_S,
        output_dir=str(output),
    )


def _snapshot_complete(path: Path, shape: tuple[int, int], time_s: float) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as saved:
            return bool(
                saved["density"].shape == shape
                and np.isclose(float(saved["time_s"]), time_s)
            )
    except (OSError, KeyError, ValueError):
        return False


def prepare_b_snapshot(
    base: SimulationConfig, *, resume: bool = True
) -> tuple[Path, SimulationConfig]:
    """Run the 198-active-column B case and normalize it to 201 columns."""

    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    run_root = stage_root / "runs" / "b_author_index"
    stable = stage_root / "simulations" / "density_b_author_index_alpha_0p4.npz"
    config = b_active_config(base, run_root)
    config.validate()
    if _snapshot_complete(stable, (base.ny, DISPLAY_HORIZONTAL_CELLS), SNAPSHOT_TIME_S):
        return stable, config

    archive = config.to_dict()
    archive["ab_calibration"] = {
        "variant": "B",
        "changed_factor": "horizontal_periodic_indexing_only",
        "nominal_interior_storage_columns": DISPLAY_HORIZONTAL_CELLS,
        "active_horizontal_slice_python": [0, 198],
        "active_horizontal_columns": ACTIVE_HORIZONTAL_CELLS,
        "internal_right_periodic_continuation_columns": INTERNAL_PERIODIC_CONTINUATION,
        "external_ghost_columns_each_side": EXTERNAL_GHOST_CELLS,
        "unchanged_from_baseline": [
            "physics",
            "harmonic_face_potential_operator",
            "zero_vertical_potential_boundary",
            "CG_plus_PyAMG",
            "PDM7_reconstruction_and_limiter",
            "AB3_predictor_corrector_with_SSPRK3_startup",
            "dt_and_CFL_policy",
            "vertical_grid_and_density_boundary",
        ],
    }
    config_path = stage_root / "configs" / "b_author_index_alpha_0p4.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(archive, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    source = run_root / "simulations" / "density_pdm7.npz"
    if not _snapshot_complete(source, (base.ny, ACTIVE_HORIZONTAL_CELLS), SNAPSHOT_TIME_S):
        source = run_simulation(config, resume=resume)
    with np.load(source) as saved:
        active_density = np.asarray(saved["density"], dtype=np.float64)
        y_km = np.asarray(saved["y_km"], dtype=np.float64)
        time_s = float(saved["time_s"])
        dt_s = float(saved["dt_s"])
        initial_mass = float(saved["initial_mass"])
        final_mass = float(saved["final_mass"])
    stable.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stable,
        density=expand_author_interior(active_density),
        active_density=active_density,
        storage_with_four_ghosts=add_four_external_ghosts(active_density),
        x_km=base.x_coordinates_m / 1.0e3 - 125.0,
        active_x_km=active_horizontal_coordinates_m(base) / 1.0e3 - 125.0,
        y_km=y_km,
        time_s=time_s,
        dt_s=dt_s,
        scheme="pdm7",
        pdm_alpha=PDM_ALPHA,
        initial_mass=initial_mass,
        final_mass=final_mass,
        periodic_index_variant="B_author_198_plus_3_continuation",
    )
    return stable, config


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
    ref_range = max(float(np.ptp(reference)), np.finfo(float).tiny)
    centered_a = simulated.ravel() - float(np.mean(simulated))
    centered_b = reference.ravel() - float(np.mean(reference))
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    return {
        "normalized_rmse_by_reference_range": float(
            np.sqrt(np.mean(residual**2)) / ref_range
        ),
        "mean_absolute_error_m3": float(np.mean(np.abs(residual))),
        "pearson_spatial_correlation": float(
            np.dot(centered_a, centered_b) / max(denominator, np.finfo(float).tiny)
        ),
    }


def write_report_and_source(
    base: SimulationConfig,
    a_path: str | Path,
    b_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    a, x, y, ta = _load_snapshot(a_path)
    b, xb, yb, tb = _load_snapshot(b_path)
    if not (np.array_equal(x, xb) and np.array_equal(y, yb)):
        raise ValueError("A and B normalized snapshots do not share the nominal grid")
    author = _author_density(base.reference_mat)
    metrics = {
        "A_current_201_active": morphology_metrics(a, x, y),
        "B_author_index_198_active": morphology_metrics(b, x, y),
        "author_reference": morphology_metrics(author, x, y),
    }
    evidence = author_index_evidence(base.reference_mat)
    report: dict[str, Any] = {
        "experiment": "isolated horizontal periodic-index A/B",
        "pdm_alpha": PDM_ALPHA,
        "time_s": SNAPSHOT_TIME_S,
        "reference_variable": REFERENCE_VARIABLE,
        "reference_mat_md5": md5sum(base.reference_mat),
        "index_conclusion": {
            "nominal_201_interior_layout": "indices 0:198 active + 198:201 periodic continuation",
            "active_python_slice": [0, 198],
            "external_four_ghost_sources": {
                "left": "interior[194:198]",
                "right": "interior[3:7]",
            },
            "supported_by_exact_zero_error_external_ghost_copies": bool(
                all(item["external_copy_relations_exact"] for item in evidence.values())
            ),
        },
        "author_array_evidence": evidence,
        "variants": {
            "A": "current production baseline; all 201 stored columns are active periodic unknowns",
            "B": "198 active periodic unknowns; normalize to 198 active + 3 right continuation columns",
        },
        "metrics": {key: value.to_dict() for key, value in metrics.items()},
        "reference_comparison": {
            "A": compare_with_reference(metrics["A_current_201_active"], metrics["author_reference"]),
            "B": compare_with_reference(metrics["B_author_index_198_active"], metrics["author_reference"]),
        },
        "field_agreement_diagnostic_only": {
            "A": _field_agreement(a, author),
            "B": _field_agreement(b, author),
            "note": "Nonlinear branch phases make pointwise metrics diagnostic, not acceptance criteria.",
        },
        "times_verified_s": {"A": ta, "B": tb},
    }
    report["calibration_outcome"] = {
        "B_improves_bubble_top_error": bool(
            report["reference_comparison"]["B"]["bubble_top_absolute_error_km"]
            < report["reference_comparison"]["A"]["bubble_top_absolute_error_km"]
        ),
        "B_improves_depleted_area_error": bool(
            report["reference_comparison"]["B"]["depleted_area_relative_error"]
            < report["reference_comparison"]["A"]["depleted_area_relative_error"]
        ),
        "B_improves_spatial_correlation": bool(
            report["field_agreement_diagnostic_only"]["B"]["pearson_spatial_correlation"]
            > report["field_agreement_diagnostic_only"]["A"]["pearson_spatial_correlation"]
        ),
    }
    report_path = stage_root / "periodic_index_ab_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    source_path = stage_root / "periodic_index_ab_source_data.npz"
    np.savez_compressed(
        source_path,
        x_km=x,
        y_km=y,
        density_A_current_201_active_m3=a,
        density_B_author_index_198_active_m3=b,
        density_author_m3=author,
        difference_B_minus_A_m3=b - a,
        time_s=SNAPSHOT_TIME_S,
        pdm_alpha=PDM_ALPHA,
    )
    return report_path, source_path, report


def create_ab_figure(
    base: SimulationConfig,
    a_path: str | Path,
    b_path: str | Path,
) -> dict[str, Path]:
    """Render A, B, author, and B-minus-A in a compact 2-by-2 plate."""

    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    a, x, y, _ = _load_snapshot(a_path)
    b, _, _, _ = _load_snapshot(b_path)
    author = _author_density(base.reference_mat)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(6.8, 7.6), sharex=True, sharey=True, constrained_layout=False
    )
    panels = (
        (a / 1.0e11, "a) A: current 201-active indexing", "plasma", 0.0, 11.0),
        (b / 1.0e11, "b) B: author 198 + 3 indexing", "plasma", 0.0, 11.0),
        (author / 1.0e11, "c) Author reference", "plasma", 0.0, 11.0),
    )
    density_image = None
    for axis, (field, title, cmap, vmin, vmax) in zip(axes.ravel()[:3], panels):
        density_image = axis.pcolormesh(
            x, y, field, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        axis.set_title(title, fontsize=8.3, pad=4.0)
    difference = (b - a) / 1.0e11
    difference_limit = max(float(np.percentile(np.abs(difference), 99.0)), 0.25)
    difference_image = axes[1, 1].pcolormesh(
        x,
        y,
        difference,
        shading="nearest",
        cmap="RdBu_r",
        vmin=-difference_limit,
        vmax=difference_limit,
        rasterized=True,
    )
    axes[1, 1].set_title("d) B − A (indexing effect)", fontsize=8.3, pad=4.0)
    for axis in axes.ravel():
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
    axes[1, 0].set_xlabel("East–West (km)")
    axes[1, 1].set_xlabel("East–West (km)")
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[1, 0].set_ylabel("Altitude (km)")
    figure.subplots_adjust(left=0.10, right=0.86, bottom=0.075, top=0.955, wspace=0.10, hspace=0.12)
    assert density_image is not None
    density_cax = figure.add_axes((0.885, 0.535, 0.024, 0.40))
    density_colorbar = figure.colorbar(density_image, cax=density_cax)
    density_colorbar.set_label(r"Density ($10^5$ cm$^{-3}$)")
    difference_cax = figure.add_axes((0.885, 0.085, 0.024, 0.36))
    difference_colorbar = figure.colorbar(difference_image, cax=difference_cax)
    difference_colorbar.set_label(r"B − A ($10^5$ cm$^{-3}$)")
    base_path = stage_root / "periodic_index_ab_comparison"
    outputs = {extension: base_path.with_suffix(f".{extension}") for extension in ("svg", "pdf", "png", "tiff")}
    figure.savefig(outputs["svg"], metadata={"Creator": "esf_sim periodic-index A/B"})
    figure.savefig(outputs["pdf"], metadata={"Creator": "esf_sim periodic-index A/B"})
    figure.savefig(outputs["png"], dpi=600)
    figure.savefig(outputs["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    return outputs


def run_ab(
    base: SimulationConfig, *, simulate: bool = True, resume: bool = True
) -> dict[str, Any]:
    """Execute or render the isolated periodic-index calibration."""

    a_path = base.output_directory / "figure3" / "simulations" / "density_alpha_0p4.npz"
    if not _snapshot_complete(a_path, (base.ny, DISPLAY_HORIZONTAL_CELLS), SNAPSHOT_TIME_S):
        raise FileNotFoundError(
            f"complete production A baseline is missing: {a_path}"
        )
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    b_path = stage_root / "simulations" / "density_b_author_index_alpha_0p4.npz"
    if simulate:
        b_path, _ = prepare_b_snapshot(base, resume=resume)
    elif not _snapshot_complete(b_path, (base.ny, DISPLAY_HORIZONTAL_CELLS), SNAPSHOT_TIME_S):
        raise FileNotFoundError(f"complete B snapshot is missing: {b_path}")
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
    parser = argparse.ArgumentParser(prog="python -m esf_sim.periodic_index_ab")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verify-indices-only", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig.from_json(args.config)
    if args.verify_indices_only:
        print(json.dumps(author_index_evidence(base.reference_mat), indent=2))
        return 0
    result = run_ab(
        base,
        simulate=not args.skip_simulation,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
