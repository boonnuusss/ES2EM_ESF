"""Public Figure 3 workflow with morphology-aware trend acceptance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from . import _figure3_impl as _impl
from ._figure3_impl import (
    FIGURE3_CASES,
    Figure3Case,
    create_figure3,
    prepare_figure3_snapshots,
    write_figure3_source_data,
)


_write_figure3_metrics_impl = _impl.write_figure3_metrics


def write_figure3_metrics(
    snapshots: Mapping[str, str | Path],
    *,
    reference_directory: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Write acceptance metrics using robust low-alpha versus high-alpha groups."""

    report = _write_figure3_metrics_impl(
        snapshots,
        reference_directory=reference_directory,
        output_path=output_path,
    )
    simulated = report["simulated"]
    metrics = [simulated[case.key]["metrics"] for case in FIGURE3_CASES]
    gradients = [item["edge_gradient_p99_m3_per_km"] for item in metrics]
    complexities = [item["interface_complexity"] for item in metrics]
    fine_scales = [item["fine_scale_laplacian_normalized"] for item in metrics]

    strict_monotonic = {
        "edge_gradient_strictly_decreases": bool(
            gradients[0] > gradients[1] > gradients[2] > gradients[3]
        ),
        "interface_complexity_strictly_decreases": bool(
            complexities[0] > complexities[1] > complexities[2] > complexities[3]
        ),
        "fine_scale_content_strictly_decreases": bool(
            fine_scales[0] > fine_scales[1] > fine_scales[2] > fine_scales[3]
        ),
    }
    trend = {
        "low_alpha_group_has_steeper_edges_than_high_alpha_group": bool(
            min(gradients[:2]) > max(gradients[2:])
        ),
        "low_alpha_group_is_more_complex_than_high_alpha_group": bool(
            min(complexities[:2]) > max(complexities[2:])
        ),
        "low_alpha_group_has_more_fine_scale_content": bool(
            min(fine_scales[:2]) > max(fine_scales[2:])
        ),
        "alpha_0p8_is_smoother_than_alpha_0p2": bool(
            gradients[3] < gradients[0]
            and complexities[3] < complexities[0]
            and fine_scales[3] < fine_scales[0]
        ),
    }
    report["scientific_alpha_trend"] = trend
    report["strict_monotonic_diagnostic_not_an_acceptance_condition"] = (
        strict_monotonic
    )
    report["all_scientific_alpha_trend_criteria_pass"] = bool(all(trend.values()))
    Path(output_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def reproduce_figure3(*args, **kwargs):
    _impl.write_figure3_metrics = write_figure3_metrics
    return _impl.reproduce_figure3(*args, **kwargs)


__all__ = [
    "FIGURE3_CASES",
    "Figure3Case",
    "create_figure3",
    "prepare_figure3_snapshots",
    "reproduce_figure3",
    "write_figure3_metrics",
    "write_figure3_source_data",
]
