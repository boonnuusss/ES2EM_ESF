"""Generate machine-readable morphology and author-data acceptance reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .metrics import compare_with_reference, morphology_metrics
from .plotting import SCHEME_ORDER
from .reference import load_figure2_reference, verify_reference_files


def write_metrics_report(
    snapshots: Mapping[str, str | Path],
    *,
    reference_directory: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    x_km = np.linspace(-125.0, 125.0, 201)
    y_km = np.linspace(250.0, 650.0, 401)
    simulated: dict[str, object] = {}
    simulated_metrics = {}
    for scheme in SCHEME_ORDER:
        with np.load(snapshots[scheme]) as saved:
            density = np.asarray(saved["density"], dtype=np.float64)
            sx = np.asarray(saved["x_km"], dtype=np.float64)
            sy = np.asarray(saved["y_km"], dtype=np.float64)
            time_s = float(saved["time_s"])
        metric = morphology_metrics(density, sx, sy)
        simulated_metrics[scheme] = metric
        simulated[scheme] = {"time_s": time_s, "metrics": metric.to_dict()}

    root = Path(reference_directory)
    references = load_figure2_reference(root / "numDensity.mat")
    comparisons: dict[str, object] = {}
    reference_metrics: dict[str, object] = {}
    for scheme in SCHEME_ORDER:
        density, time_s, variable = references[scheme]
        metric = morphology_metrics(density, x_km, y_km)
        reference_metrics[scheme] = {
            "variable": variable,
            "time_s": time_s,
            "metrics": metric.to_dict(),
        }
        comparisons[scheme] = compare_with_reference(simulated_metrics[scheme], metric)

    gradients = [
        simulated_metrics[scheme].edge_gradient_p99_m3_per_km for scheme in SCHEME_ORDER
    ]
    complexities = [
        simulated_metrics[scheme].interface_complexity for scheme in SCHEME_ORDER
    ]
    fine_scales = [
        simulated_metrics[scheme].fine_scale_laplacian_normalized
        for scheme in SCHEME_ORDER
    ]
    hierarchy = {
        "edge_gradient_pdm7_gt_van_leer_gt_first_order": bool(
            gradients[2] > gradients[1] > gradients[0]
        ),
        "interface_complexity_pdm7_gt_van_leer_gt_first_order": bool(
            complexities[2] > complexities[1] > complexities[0]
        ),
        "fine_scale_content_pdm7_gt_van_leer_gt_first_order": bool(
            fine_scales[2] > fine_scales[1] > fine_scales[0]
        ),
    }
    morphology = {
        "first_order_single_smooth_channel": bool(
            simulated_metrics["first_order"].branch_peak_count <= 1
        ),
        "van_leer_has_bifurcation_or_secondary_structure": bool(
            simulated_metrics["van_leer"].branch_peak_count >= 2
            or complexities[1] > 1.2 * complexities[0]
        ),
        "pdm7_has_steeper_and_finer_structure_than_van_leer": bool(
            gradients[2] > gradients[1] and fine_scales[2] > fine_scales[1]
        ),
    }
    all_reference_criteria = all(
        all(item["criteria"].values()) for item in comparisons.values()
    )
    report: dict[str, object] = {
        "provenance": {
            "simulation": "independent Python solve; author arrays are not read by the solver",
            "reference_dataset_version": "ScienceDB V2",
            "reference_file_integrity": verify_reference_files(root),
            "figure2_reference_note": (
                "ScienceDB stores the first-order reference at 2800 s, van Leer at "
                "3150 s, and alpha=0.2 PDM7 at 3150 s. Independent outputs are all 3150 s."
            ),
            "morphology_metric_note": (
                "Boundary complexity uses every significant component of the strong "
                "depletion mask, not only the largest connected component."
            ),
        },
        "simulated": simulated,
        "author_reference": reference_metrics,
        "comparison": comparisons,
        "scientific_morphology": morphology,
        "hierarchy": hierarchy,
        "all_reference_thresholds_pass": bool(all_reference_criteria),
        "all_scientific_morphology_criteria_pass": bool(all(morphology.values())),
        "all_hierarchy_thresholds_pass": bool(all(hierarchy.values())),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
