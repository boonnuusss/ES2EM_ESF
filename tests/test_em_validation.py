from __future__ import annotations

import json

from esf_sim import cli
from esf_sim.em_validation import run_em_validation


def test_fast_em_validation_is_ordered_complete_and_writes_json(tmp_path):
    destination = tmp_path / "reports"
    report = run_em_validation(destination)

    assert report["all_pass"]
    assert [level["level"] for level in report["levels"]] == list(range(1, 10))
    assert [level["name"] for level in report["levels"]] == [
        "uniform_va_wave_accuracy_and_walen_relation",
        "conductance_load_analytic_reflection",
        "nonuniform_va_impedance_jump_propagation",
        "lossless_energy_and_poynting_joule_budget",
        "finite_line_va_infinity_to_laundal_dynamic_endpoint",
        "li_electrostatic_limit_adapter",
        "frozen_li_snapshot_one_way_alfven_drive",
        "controlled_two_way_feedback_toy_closure",
        "dimensional_fac_laplacian_nonlinear_feedback",
    ]
    for level in report["levels"]:
        assert {"name", "status", "metric", "threshold"} <= set(level)
        assert level["status"] == "pass"

    stage8 = report["levels"][-2]
    assert stage8["details"]["scope"] == "control_layer_and_toy_closure_only"
    assert not stage8["details"][
        "physical_dimensional_fac_to_li_closure_complete"
    ]
    stage9 = report["levels"][-1]
    assert stage9["details"]["physical_dimensional_fac_to_li_closure_complete"]
    assert stage9["details"]["transverse_laplacian_fac_mapping_complete"]
    assert stage9["details"]["small_grid_nonlinear_two_way_run_complete"]
    assert not stage9["details"]["formal_nonlinear_epb_two_way_run_complete"]
    assert not stage9["details"]["parallel_mass_transport_included"]
    assert report["execution_scope"] == {
        "formal_li_snapshot_used": False,
        "pardiso_used": False,
        "long_epb_simulation_used": False,
        "stages_1_through_7": "physical_reduced_em_components",
        "stage_8": "controlled_feedback_layer_with_toy_closure",
        "stage_9": "dimensional_transverse_modal_fac_and_tiny_nonlinear_feedback",
    }

    path = destination / "em_validation_report.json"
    assert report["report_path"] == str(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["all_pass"]
    assert len(saved["levels"]) == 9
    # report_path is added only after the atomic payload has been written.
    assert "report_path" not in saved


def test_validate_em_cli_exit_status_follows_all_pass(monkeypatch, capsys):
    passing = {
        "all_pass": True,
        "levels": [
            {"name": "check", "status": "pass", "metric": 0.0, "threshold": 1.0}
        ],
    }
    monkeypatch.setattr(cli, "run_em_validation", lambda output: passing)
    assert cli.main(["validate-em"]) == 0
    assert json.loads(capsys.readouterr().out)["all_pass"]

    failing = {
        "all_pass": False,
        "levels": [
            {"name": "check", "status": "fail", "metric": 2.0, "threshold": 1.0}
        ],
    }
    monkeypatch.setattr(cli, "run_em_validation", lambda output: failing)
    assert cli.main(["validate-em", "--output", "unused.json"]) == 1
    assert not json.loads(capsys.readouterr().out)["all_pass"]
