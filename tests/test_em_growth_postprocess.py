from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from esf_sim.em_growth_postprocess import (
    load_growth_history,
    load_latest_density_pair,
    postprocess_em_growth,
)


matplotlib.use("Agg")


def _densities(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(0.8, 1.2, 4)[:, None]
    x = np.cos(np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False))[None, :]
    baseline = 1.0e11 * y * (1.0 + 0.05 * x)
    coupled = baseline + scale * 1.0e8 * y * x
    return baseline, coupled


def _write_report(root: Path, *, completed_time_s: float = 2.0) -> None:
    payload = {
        "format": "esf_dimensional_em_growth_v1",
        "experiment_label": "synthetic_postprocess_test",
        "configuration_hash": "abc123",
        "completed_time_s": completed_time_s,
        "initial_density_array_equal": True,
        "li_config": {
            "nx": 5,
            "ny": 4,
            "x_min_km": 0.0,
            "x_max_km": 10.0,
            "y_min_km": 250.0,
            "y_max_km": 256.0,
        },
        "em_config": {"growth_fit_start_s": 0.0},
        "scope": {"two_way_nonlinear_density_feedback": True},
    }
    (root / "em_growth_report.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_postprocessor_writes_auditable_figure_metrics_and_source_data(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    baseline, coupled = _densities()
    np.savez_compressed(
        root / "final_states.npz",
        baseline_density_m3=baseline,
        coupled_density_m3=coupled,
        time_s=np.float64(2.0),
    )
    np.savez_compressed(
        root / "checkpoint.npz",
        baseline_density_m3=baseline,
        coupled_density_m3=coupled,
        time_s=np.float64(2.0),
        times_s=np.array([0.0, 1.0, 2.0]),
        baseline_amplitudes_m3=np.array([1.0, 2.0, 4.0]),
        coupled_amplitudes_m3=np.array([1.0, 3.0, 9.0]),
    )
    records = [
        {
            "step": 1,
            "time_s": 1.0,
            "dt_s": 1.0,
            "baseline_amplitude_m3": 2.0,
            "coupled_amplitude_m3": 3.0,
            "baseline_cfl": 0.1,
            "coupled_cfl": 0.2,
            "coupling_relative_residual": 1.0e-10,
            "minimum_baseline_density_m3": float(np.min(baseline)),
            "minimum_coupled_density_m3": float(np.min(coupled)),
        },
        {
            "step": 2,
            "time_s": 2.0,
            "dt_s": 1.0,
            "baseline_amplitude_m3": 4.0,
            "coupled_amplitude_m3": 9.0,
            "baseline_cfl": 0.15,
            "coupled_cfl": 0.25,
            "coupling_relative_residual": 2.0e-10,
            "minimum_baseline_density_m3": float(np.min(baseline)),
            "minimum_coupled_density_m3": float(np.min(coupled)),
        },
    ]
    (root / "diagnostics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    _write_report(root)

    metrics = postprocess_em_growth(root)

    output = root / "postprocess"
    for name in (
        "em_growth_comparison.svg",
        "em_growth_comparison.pdf",
        "em_growth_comparison.png",
        "em_growth_metrics.json",
        "em_growth_source_data.npz",
    ):
        assert (output / name).stat().st_size > 0
    assert "<text" in (output / "em_growth_comparison.svg").read_text(
        encoding="utf-8"
    )
    assert metrics["selected_state"]["kind"] == "final_states"
    assert metrics["selected_state"]["time_matches_archived_completed_time"]
    assert metrics["growth_history"]["kind"] == "checkpoint"
    assert metrics["growth_fit"]["baseline"]["rate_s_inv"] == pytest.approx(
        np.log(2.0)
    )
    assert metrics["growth_fit"]["coupled"]["rate_s_inv"] == pytest.approx(
        np.log(3.0)
    )
    assert metrics["numerical_health"]["maximum_coupled_cfl"] == 0.25
    assert metrics["density_comparison"]["relative_l2_difference"] > 0.0
    parsed = json.loads((output / "em_growth_metrics.json").read_text(encoding="utf-8"))
    assert parsed["format"] == "esf_em_growth_postprocess_v1"
    with np.load(output / "em_growth_source_data.npz", allow_pickle=False) as source:
        np.testing.assert_array_equal(source["coupled_density_m3"], coupled)
        assert source["history_time_s"].shape == (3,)


def test_latest_snapshot_and_diagnostics_fallback_are_supported(tmp_path):
    root = tmp_path / "snapshot_run"
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True)
    baseline, coupled = _densities()
    for index, (time_s, scale) in enumerate(((0.0, 0.0), (1.0, 2.0))):
        _, snapshot_coupled = _densities(scale)
        np.savez_compressed(
            snapshots / f"snapshot_{index:06d}.npz",
            time_s=np.float64(time_s),
            baseline_density_m3=baseline,
            coupled_density_m3=snapshot_coupled,
        )
    records = [
        {
            "time_s": 0.0,
            "baseline_amplitude_m3": 1.0,
            "coupled_amplitude_m3": 1.0,
        },
        {
            "time_s": 1.0,
            "baseline_amplitude_m3": 2.0,
            "coupled_amplitude_m3": 2.5,
        },
    ]
    (root / "diagnostics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    _write_report(root, completed_time_s=1.0)

    pair = load_latest_density_pair(root)
    history = load_growth_history(root)
    assert pair.kind == "snapshot"
    assert pair.time_s == 1.0
    assert history.kind == "diagnostics"
    assert history.time_s.tolist() == [0.0, 1.0]
    assert history.coupled_amplitude_m3.tolist() == [1.0, 2.5]

    metrics = postprocess_em_growth(root)
    assert metrics["selected_state"]["kind"] == "snapshot"
    assert metrics["growth_history"]["kind"] == "diagnostics"


def test_three_track_postprocess_separates_geometry_and_em_effects(tmp_path):
    root = tmp_path / "three_track"
    root.mkdir()
    baseline, coupled = _densities()
    column = baseline + 0.4 * (coupled - baseline)
    np.savez_compressed(
        root / "final_states.npz",
        schema_version=np.int64(2),
        baseline_density_m3=baseline,
        column_control_density_m3=column,
        coupled_density_m3=coupled,
        time_s=np.float64(2.0),
    )
    np.savez_compressed(
        root / "checkpoint.npz",
        schema_version=np.int64(3),
        baseline_density_m3=baseline,
        column_control_density_m3=column,
        coupled_density_m3=coupled,
        time_s=np.float64(2.0),
        times_s=np.array([0.0, 1.0, 2.0]),
        baseline_amplitudes_m3=np.array([1.0, 2.0, 4.0]),
        column_control_amplitudes_m3=np.array([1.0, 2.5, 6.25]),
        coupled_amplitudes_m3=np.array([1.0, 3.0, 9.0]),
    )
    _write_report(root)

    pair = load_latest_density_pair(root)
    history = load_growth_history(root)
    np.testing.assert_array_equal(pair.column_control_m3, column)
    np.testing.assert_array_equal(
        history.column_control_amplitude_m3, [1.0, 2.5, 6.25]
    )

    metrics = postprocess_em_growth(root)
    assert metrics["format"] == "esf_em_growth_postprocess_v2"
    effects = metrics["density_comparison"]["separated_effects"]
    assert effects["geometry_column_es_minus_local_es"][
        "relative_l2_difference"
    ] > 0.0
    assert effects["electromagnetic_column_em_minus_column_es"][
        "relative_l2_difference"
    ] > 0.0
    assert metrics["growth_fit"]["separated_effects"] is not None
    with np.load(
        root / "postprocess" / "em_growth_source_data.npz", allow_pickle=False
    ) as source:
        np.testing.assert_array_equal(source["column_control_density_m3"], column)
        np.testing.assert_array_equal(
            source["geometry_density_effect_m3"], column - baseline
        )
        np.testing.assert_array_equal(
            source["electromagnetic_density_effect_m3"], coupled - column
        )
