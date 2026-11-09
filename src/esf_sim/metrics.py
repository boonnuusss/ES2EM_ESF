"""Morphology and reference-acceptance metrics for ESF density snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MorphologyMetrics:
    percentile_1_m3: float
    percentile_99_m3: float
    edge_gradient_p99_m3_per_km: float
    gradient_total_variation_m3_per_km: float
    fine_scale_laplacian_normalized: float
    depleted_area_km2: float
    bubble_top_km: float
    boundary_length_km: float
    interface_complexity: float
    fine_structure_count: int
    branch_peak_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _depletion_mask(density: np.ndarray) -> np.ndarray:
    """Return strong depletion cores using a row-local 30% threshold."""

    horizontal_reference = np.percentile(density, 90.0, axis=1, keepdims=True)
    mask = density < 0.30 * np.maximum(horizontal_reference, np.finfo(float).tiny)
    active_rows = horizontal_reference[:, 0] > 0.03 * float(np.max(horizontal_reference))
    return mask & active_rows[:, None]


def _retain_significant_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if not count:
        return np.zeros_like(mask), 0
    sizes = np.asarray(
        ndimage.sum(mask, labels, index=np.arange(1, count + 1)),
        dtype=float,
    )
    retained = np.zeros_like(mask)
    keep_ids = np.where(sizes >= 4)[0] + 1
    for component_id in keep_ids:
        retained |= labels == component_id
    return retained, int(keep_ids.size)


def morphology_metrics(
    density: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
) -> MorphologyMetrics:
    n = np.asarray(density, dtype=np.float64)
    x = np.asarray(x_km, dtype=np.float64)
    y = np.asarray(y_km, dtype=np.float64)
    if n.shape != (y.size, x.size):
        raise ValueError("density shape must be (len(y_km), len(x_km))")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    gradient_y, gradient_x = np.gradient(n, dy, dx)
    gradient = np.hypot(gradient_x, gradient_y)
    retained, fine_structure_count = _retain_significant_components(
        _depletion_mask(n)
    )

    edge_horizontal = retained[:, 1:] != retained[:, :-1]
    edge_vertical = retained[1:, :] != retained[:-1, :]
    boundary_length = float(
        np.count_nonzero(edge_horizontal) * dy + np.count_nonzero(edge_vertical) * dx
    )
    area = float(np.count_nonzero(retained) * dx * dy)
    bubble_top = float(y[np.where(retained)[0].max()]) if np.any(retained) else float("nan")
    complexity = boundary_length / max(
        2.0 * np.sqrt(np.pi * area), np.finfo(float).tiny
    )

    branch_peak_count = 0
    if np.any(retained):
        rows = np.where(retained)[0]
        sample_rows = rows[rows >= int(np.median(rows))]
        profile = (
            np.mean(retained[sample_rows], axis=0)
            if sample_rows.size
            else retained[rows[-1]]
        )
        _, branch_peak_count = ndimage.label(profile > 0.08)

    normalized = n / 1.0e12
    laplacian_measure = float(np.mean(np.abs(ndimage.laplace(normalized))))
    return MorphologyMetrics(
        percentile_1_m3=float(np.percentile(n, 1.0)),
        percentile_99_m3=float(np.percentile(n, 99.0)),
        edge_gradient_p99_m3_per_km=float(np.percentile(gradient, 99.0)),
        gradient_total_variation_m3_per_km=float(np.mean(gradient)),
        fine_scale_laplacian_normalized=laplacian_measure,
        depleted_area_km2=area,
        bubble_top_km=bubble_top,
        boundary_length_km=boundary_length,
        interface_complexity=float(complexity),
        fine_structure_count=fine_structure_count,
        branch_peak_count=int(branch_peak_count),
    )


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).tiny)


def compare_with_reference(
    simulated: MorphologyMetrics,
    reference: MorphologyMetrics,
) -> dict[str, Any]:
    top_error_km = abs(simulated.bubble_top_km - reference.bubble_top_km)
    area_error = relative_error(simulated.depleted_area_km2, reference.depleted_area_km2)
    p1_error = relative_error(simulated.percentile_1_m3, reference.percentile_1_m3)
    p99_error = relative_error(simulated.percentile_99_m3, reference.percentile_99_m3)
    return {
        "bubble_top_absolute_error_km": top_error_km,
        "depleted_area_relative_error": area_error,
        "percentile_1_relative_error": p1_error,
        "percentile_99_relative_error": p99_error,
        "criteria": {
            "bubble_top_within_25_km": bool(top_error_km <= 25.0),
            "depleted_area_within_25_percent": bool(area_error <= 0.25),
            "percentile_1_within_10_percent": bool(p1_error <= 0.10),
            "percentile_99_within_10_percent": bool(p99_error <= 0.10),
        },
    }
