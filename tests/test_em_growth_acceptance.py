from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esf_sim.em_growth_acceptance import (
    FORMAT,
    _configuration_hash_from_report,
    bubble_morphology,
    evaluate_run,
    main,
)


def _write_npz(path: Path, **updates) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)


def _read_records(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_records(root: Path, records: list[dict]) -> None:
    (root / "diagnostics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _synchronize_hash(root: Path, report: dict) -> str:
    digest = _configuration_hash_from_report(report)
    report["configuration_hash"] = digest
    (root / "em_growth_report.json").write_text(json.dumps(report), encoding="utf-8")
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    status["configuration_hash"] = digest
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
    _write_npz(root / "checkpoint.npz", configuration_hash=np.asarray(digest))
    return digest


def _write_synthetic_mature_run(root: Path, *, weak_coupling: bool = False) -> None:
    root.mkdir()
    nx = 11  # First/last archive columns represent the same periodic point.
    ny = 21
    local = np.full((ny, nx), 100.0, dtype=np.float64)
    column = local.copy()
    coupled = column.copy()
    if not weak_coupling:
        for row in range(9, 20):  # 430--630 km inclusive.
            coupled[row, (9, 0, 1)] = 20.0
            coupled[row, -1] = coupled[row, 0]

    report = {
        "format": "esf_dimensional_em_growth_v2",
        "completed_time_s": 3150.0,
        "steps": 2,
        "dt_reductions": 0,
        "li_config": {
            "nx": nx,
            "ny": ny,
            "x_min_km": 0.0,
            "x_max_km": 100.0,
            "y_min_km": 250.0,
            "y_max_km": 650.0,
            "dx_km": 10.0,
            "dy_km": 20.0,
            "relative_noise": 0.0,
        },
        "em_config": {
            "comparison_tracks": "local_column_em",
            "t_end_s": 3150.0,
            "growth_fit_start_s": 300.0,
            "feedback_gain": 1.0,
            "feedback_ramp_time_s": 20.0,
            "parallel_averaging_profile": "dipole_chapman_flux_tube",
            "parallel_averaging_length_km": None,
            "right_boundary_model": "matched",
            "right_boundary_conductance_s": None,
            "initial_line_state": "zero",
            "initial_characteristic_seed_v": 0.0,
            "field_line_cells": 2,
            "field_line_length_km": 20000.0,
            "alfven_speed_km_s": 1000.0,
            "hemisphere_count": 2,
            "max_x_wavenumber": 1,
            "max_y_wavenumber": 0,
            "include_sine_modes": True,
        },
        "scope": {
            "parallel_averaging_profile": "dipole_chapman_flux_tube",
            "fac_forcing_representation": "surface_current",
            "remote_boundary_model": "matched",
            "remote_boundary_reflection_coefficient": 0.0,
        },
    }
    digest = _configuration_hash_from_report(report)
    report["configuration_hash"] = digest
    (root / "em_growth_report.json").write_text(json.dumps(report), encoding="utf-8")
    (root / "status.json").write_text(
        json.dumps({
            "status": "completed", "time_s": 3150.0, "step": 2,
            "configuration_hash": digest,
        }),
        encoding="utf-8",
    )

    zeros = np.zeros((2, 2), dtype=np.float64)
    np.savez_compressed(
        root / "final_states.npz",
        schema_version=np.int64(2),
        baseline_density_m3=local,
        column_control_density_m3=column,
        coupled_density_m3=coupled,
        coupled_w_plus_v=zeros,
        coupled_w_minus_v=zeros,
        time_s=np.float64(3150.0),
    )
    times = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3150.0])
    local_amplitude = np.array([10.0, 9.0, 8.0, 10.0, 15.0, 25.0, 40.0])
    column_amplitude = np.array([10.0, 9.5, 8.5, 11.0, 16.0, 26.0, 42.0])
    coupled_amplitude = (
        column_amplitude.copy() if weak_coupling
        else np.array([10.0, 9.0, 8.0, 10.5, 17.0, 28.0, 45.0])
    )
    np.savez_compressed(
        root / "checkpoint.npz",
        schema_version=np.int64(3),
        time_s=np.float64(3150.0),
        step=np.int64(2),
        configuration_hash=np.asarray(digest),
        comparison_tracks=np.asarray("local_column_em"),
        baseline_density_m3=local,
        column_control_density_m3=column,
        coupled_density_m3=coupled,
        coupled_w_plus_v=zeros,
        coupled_w_minus_v=zeros,
        times_s=times,
        baseline_amplitudes_m3=local_amplitude,
        column_control_amplitudes_m3=column_amplitude,
        coupled_amplitudes_m3=coupled_amplitude,
    )

    fac_ratio = 0.001 if weak_coupling else 0.30
    records = []
    for index, time_s in enumerate((20.0, 3150.0), 1):
        records.append({
            "step": index,
            "time_s": time_s,
            "dt_s": 0.25,
            "current_dt_s": 0.25,
            "baseline_cfl": 0.10,
            "column_control_cfl": 0.12,
            "coupled_cfl": 0.20,
            "baseline_potential_relative_residual": 1.0e-12,
            "column_control_potential_relative_residual": 2.0e-12,
            "coupled_potential_relative_residual": 3.0e-12,
            "coupling_relative_residual": 4.0e-12,
            "minimum_baseline_density_m3": 100.0,
            "minimum_column_control_density_m3": 100.0,
            "minimum_coupled_density_m3": float(np.min(coupled)),
            "fac_to_gravity_ratio": fac_ratio,
            "activation_fraction": 1.0,
            "maximum_abs_fac_forcing": 1.0,
            "net_fac_forcing_integral": 0.0,
            # Deliberately large: open-y raw drift is not a numerical
            # conservation residual without the boundary flux history.
            "baseline_local_density_content_relative_change": 0.25,
            "column_control_conserved_density_content_relative_change": 0.35,
            "coupled_conserved_density_content_relative_change": 0.45,
            "relative_power_pairing_defect": 5.0e-13,
            "boundary_power_modal_w": 0.0,
            "boundary_power_grid_w": 0.0,
            "remote_boundary_power_modal_w": 0.0,
            "remote_boundary_reflection_coefficient": 0.0,
        })
    _write_records(root, records)


def _check(result: dict, name: str) -> dict:
    return next(item for item in result["checks"] if item["name"] == name)


def test_mature_acceptance_handles_duplicate_endpoint_and_periodic_bubble(tmp_path):
    root = tmp_path / "mature"
    _write_synthetic_mature_run(root)
    result = evaluate_run(root)

    assert result["pass"]
    assert result["stage"]["name"] == "3150"
    assert result["grid"]["periodic_endpoint_duplicate_removed"]
    assert result["grid"]["nx_unique"] == 10
    assert result["grid"]["periodic_endpoint_relative_mismatch"]["column_em"] == 0.0
    morphology = result["morphology"]["column_em"]
    assert morphology["component_count"] == 1
    assert morphology["cell_count"] == 33
    assert morphology["bottom_km"] == 430.0
    assert morphology["top_km"] == 630.0
    assert morphology["vertical_span_km"] == 200.0
    assert morphology["area_km2"] == 6600.0
    assert morphology["maximum_width_cells"] == 3
    assert morphology["layers_at_least_two_cells_wide"] == 11
    assert morphology["maximum_depletion"] == pytest.approx(1.0 - 20.0 / 76.0)
    growth = result["growth"]["column_em"]["post_minimum_positive_growth"]
    assert growth["minimum_time_s"] == 1000.0
    assert growth["final_to_minimum_amplitude"] == pytest.approx(45.0 / 8.0)
    assert growth["window"]["start_s"] == 1000.0
    assert growth["window"]["end_s"] == 3150.0
    assert growth["window"]["rate_s_inv"] > 1.0e-4
    assert result["fac_and_power"]["line_energy_budget"]["evaluable"]
    assert _check(result, "column_content_balance")["status"] == "warning"


def test_width_is_longest_periodic_contiguous_run_not_row_population():
    density = np.full((2, 6), 100.0)
    # The four cells on row zero belong to one component through row one, but
    # they are two disjoint runs.  Summing row pixels would report width four;
    # the largest actual interface width is the three-cell bridge on row one.
    for row, columns in enumerate(((0, 1, 3, 4), (1, 2, 3))):
        density[row, columns] = 10.0
    result = bubble_morphology(
        density, y_km=np.arange(2.0), dx_km=1.0, dy_km=1.0, threshold=0.25
    )
    assert result["cell_count"] == 7
    assert result["maximum_width_cells"] == 3
    assert result["maximum_width_km"] == 3.0


def test_weak_coupling_cannot_masquerade_as_recovered_epb(tmp_path):
    root = tmp_path / "weak"
    _write_synthetic_mature_run(root, weak_coupling=True)
    result = evaluate_run(root)
    assert not result["pass"]
    assert _check(result, "fac_to_gravity_ratio")["status"] == "fail"
    assert _check(result, "mature_em_column_relative_l2")["status"] == "fail"
    assert _check(result, "mature_bubble_exists")["status"] == "fail"


def test_fac_balance_is_normalized_per_record_with_unique_periodic_area(tmp_path):
    root = tmp_path / "fac"
    _write_synthetic_mature_run(root)
    records = _read_records(root)
    records[0]["maximum_abs_fac_forcing"] = 1.0e6
    records[0]["net_fac_forcing_integral"] = 0.0
    records[1]["maximum_abs_fac_forcing"] = 1.0e-6
    # nx_unique*ny*dx*dy*1e6 = 4.2e10 m2, hence ratio = 1e-9.
    records[1]["net_fac_forcing_integral"] = 4.2e-5
    _write_records(root, records)
    result = evaluate_run(root)
    balance = result["fac_and_power"]["surface_fac_balance"]
    assert balance["record_count"] == 2
    assert balance["maximum_ratio"] == pytest.approx(1.0e-9)
    assert _check(result, "net_surface_current_balance_proxy")["status"] == "fail"


def test_periodic_endpoint_mismatch_is_a_hard_gate(tmp_path):
    root = tmp_path / "endpoint"
    _write_synthetic_mature_run(root)
    with np.load(root / "final_states.npz", allow_pickle=False) as archive:
        coupled = archive["coupled_density_m3"].copy()
    coupled[:, -1] *= 1.01
    _write_npz(root / "final_states.npz", coupled_density_m3=coupled)
    _write_npz(root / "checkpoint.npz", coupled_density_m3=coupled)
    result = evaluate_run(root)
    assert not result["pass"]
    assert _check(result, "periodic_endpoint_match_column_em")["status"] == "fail"


def test_passive_remote_rejects_negative_instantaneous_power_despite_positive_integral(
    tmp_path,
):
    root = tmp_path / "active_remote_pulse"
    _write_synthetic_mature_run(root)
    records = _read_records(root)
    records[0]["remote_boundary_power_modal_w"] = -1.0
    records[1]["remote_boundary_power_modal_w"] = 2.0
    _write_records(root, records)
    result = evaluate_run(root)
    assert result["fac_and_power"]["remote_absorbed_energy_j"] > 0.0
    assert _check(result, "passive_remote_absorption")["status"] == "fail"


@pytest.mark.parametrize("failure", ("diagnostic_time", "final_time", "schema"))
def test_stale_terminal_artifacts_are_rejected(tmp_path, failure):
    root = tmp_path / failure
    _write_synthetic_mature_run(root)
    if failure == "diagnostic_time":
        records = _read_records(root)
        records[-1]["time_s"] = 3149.0
        _write_records(root, records)
        message = "do not reach the terminal state"
    elif failure == "final_time":
        _write_npz(root / "final_states.npz", time_s=np.float64(3149.0))
        message = "checkpoint and final-state times differ"
    else:
        _write_npz(root / "checkpoint.npz", schema_version=np.int64(2))
        message = "schema 3"
    with pytest.raises(ValueError, match=message):
        evaluate_run(root)


def test_report_configuration_tampering_is_rejected_even_if_hash_strings_agree(tmp_path):
    root = tmp_path / "tampered_config"
    _write_synthetic_mature_run(root)
    report = json.loads((root / "em_growth_report.json").read_text(encoding="utf-8"))
    report["em_config"]["feedback_gain"] = 0.5
    (root / "em_growth_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match archived configuration"):
        evaluate_run(root)


def test_nonzero_initial_line_state_marks_energy_budget_not_evaluable(tmp_path):
    root = tmp_path / "energy_unknown"
    _write_synthetic_mature_run(root)
    report = json.loads((root / "em_growth_report.json").read_text(encoding="utf-8"))
    report["em_config"]["initial_line_state"] = "seeded"
    _synchronize_hash(root, report)
    result = evaluate_run(root)
    budget = result["fac_and_power"]["line_energy_budget"]
    assert budget["evaluable"] is False
    assert budget["verification_level"] == "not_evaluable"
    assert _check(result, "line_energy_budget")["status"] == "fail"


def test_cli_writes_strict_json_and_returns_gate_status(tmp_path, capsys):
    root = tmp_path / "cli"
    _write_synthetic_mature_run(root)
    assert main([str(root)]) == 0
    capsys.readouterr()
    output = root / "em_growth_acceptance.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == FORMAT
    assert payload["pass"] is True
    assert "NaN" not in output.read_text(encoding="utf-8")
