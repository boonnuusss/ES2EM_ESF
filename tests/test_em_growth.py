from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from esf_sim import cli
from esf_sim import em_growth
from esf_sim._model_impl import CFLViolation
from esf_sim.bidirectional_em import (
    DimensionalEMState,
    PhysicalCouplingConvergenceError,
)
from esf_sim.config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
    SimulationConfig,
)
from esf_sim.em_growth import (
    EMGrowthConfig,
    MinimumTimeStepError,
    run_em_growth_experiment,
)
from esf_sim.modal_alfven import ModalAlfvenState


def _li_config() -> SimulationConfig:
    return SimulationConfig(
        nx=9,
        ny=9,
        x_min_km=0.0,
        x_max_km=8.0,
        y_min_km=250.0,
        y_max_km=258.0,
        dx_km=1.0,
        dy_km=1.0,
        perturbation_wavelength_km=9.0,
        scheme="pdm7",
        pdm_alpha=0.4,
        dt_s=0.01,
        cfl_limit=0.25,
        potential_operator=MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        potential_rtol=1.0e-9,
    )


def _em_config(output: Path, *, t_end_s: float = 0.04) -> EMGrowthConfig:
    return EMGrowthConfig(
        background_field_t=3.0e-5,
        field_line_length_km=8_000.0,
        field_line_cells=8,
        alfven_speed_km_s=800.0,
        max_x_wavenumber=1,
        max_y_wavenumber=0,
        line_spatial_order=1,
        feedback_ramp_time_s=0.0,
        coupling_relative_tolerance=1.0e-8,
        closure="mimetic",
        use_pardiso=False,
        dt_s=0.01,
        t_end_s=t_end_s,
        growth_fit_start_s=0.0,
        checkpoint_interval_s=0.02,
        initial_characteristic_seed_v=1.0e-3,
        output_dir=str(output),
        experiment_label="pytest_tiny_dimensional_em",
    )


def test_checkpointed_growth_experiment_and_resume_are_deterministic(tmp_path):
    output = tmp_path / "growth"
    report = run_em_growth_experiment(
        _li_config(), _em_config(output), resume=False
    )
    assert report["completed_time_s"] == pytest.approx(0.04)
    assert report["steps"] == 4
    assert report["initial_density_array_equal"]
    assert report["scope"]["dimensional_fac_li_closure"]
    assert report["scope"]["transverse_laplacian_fac"]
    assert report["scope"]["two_way_nonlinear_density_feedback"]
    assert not report["scope"]["common_mode_parallel_mass_transport"]
    assert report["relative_final_density_trajectory_difference"] > 0.0
    assert np.isfinite(report["baseline_growth_rate_s_inv"])
    assert np.isfinite(report["coupled_growth_rate_s_inv"])
    assert report["minimum_baseline_density_m3"] > 0.0
    assert report["minimum_coupled_density_m3"] > 0.0
    for name in (
        "checkpoint.npz",
        "final_states.npz",
        "diagnostics.jsonl",
        "em_growth_report.json",
        "status.json",
    ):
        assert (output / name).exists()
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"

    before = np.load(output / "final_states.npz")["coupled_density_m3"].copy()
    resumed = run_em_growth_experiment(
        _li_config(), _em_config(output), resume=True
    )
    after = np.load(output / "final_states.npz")["coupled_density_m3"]
    assert np.array_equal(after, before)
    assert resumed["steps"] == report["steps"]
    assert resumed["coupled_growth_rate_s_inv"] == pytest.approx(
        report["coupled_growth_rate_s_inv"], rel=0.0, abs=0.0
    )


def test_checkpoint_rejects_changed_physics_configuration(tmp_path):
    output = tmp_path / "growth"
    original = _em_config(output)
    run_em_growth_experiment(_li_config(), original, resume=False)
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        run_em_growth_experiment(
            _li_config(), replace(original, background_field_t=4.0e-5), resume=True
        )
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        run_em_growth_experiment(
            _li_config(), replace(original, right_boundary_model="matched"), resume=True
        )


def test_remote_boundary_configuration_is_validated():
    base = _em_config(Path("unused"))
    with pytest.raises(ValueError, match="requires zero"):
        replace(
            base,
            right_boundary_model="matched",
            right_boundary_voltage_v=1.0,
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        replace(
            base,
            right_boundary_model="conductance",
            right_boundary_conductance_s=-1.0,
        )
    matched = replace(base, right_boundary_model="matched")
    assert matched.right_boundary_conductance_s is None


def test_parallel_averaging_profile_configuration_is_validated():
    base = _em_config(Path("unused"))
    with pytest.raises(ValueError, match="must be null"):
        replace(
            base,
            fac_source_model="symmetric_flux_tube",
            parallel_averaging_profile="dipole_chapman_flux_tube",
            parallel_averaging_length_km=625.0,
        )
    profile = replace(
        base,
        fac_source_model="symmetric_flux_tube",
        parallel_averaging_profile="dipole_chapman_flux_tube",
        parallel_averaging_length_km=None,
    )
    assert profile.dipole_quadrature_order == 128
    with pytest.raises(ValueError, match="requires symmetric_flux_tube"):
        replace(
            base,
            parallel_averaging_profile="dipole_chapman_flux_tube",
        )


def test_stable_checkpoint_can_be_extended_to_a_later_end_time(tmp_path):
    resumed_output = tmp_path / "resumed"
    fresh_output = tmp_path / "fresh"
    run_em_growth_experiment(
        _li_config(), _em_config(resumed_output, t_end_s=0.02), resume=False
    )
    extended = run_em_growth_experiment(
        _li_config(), _em_config(resumed_output, t_end_s=0.04), resume=True
    )
    fresh = run_em_growth_experiment(
        _li_config(), _em_config(fresh_output, t_end_s=0.04), resume=False
    )
    with np.load(resumed_output / "final_states.npz") as resumed_data, np.load(
        fresh_output / "final_states.npz"
    ) as fresh_data:
        assert np.array_equal(
            resumed_data["coupled_density_m3"],
            fresh_data["coupled_density_m3"],
        )
    assert extended["steps"] == fresh["steps"] == 4


def test_run_em_growth_cli_writes_report(tmp_path, capsys):
    output = tmp_path / "cli_growth"
    li_path = tmp_path / "li.json"
    em_path = tmp_path / "em.json"
    li_path.write_text(json.dumps(_li_config().to_dict()), encoding="utf-8")
    em_path.write_text(json.dumps(_em_config(output).to_dict()), encoding="utf-8")
    assert (
        cli.main(
            [
                "run-em-growth",
                "--config",
                str(li_path),
                "--em-config",
                str(em_path),
                "--no-resume",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed_time_s"] == pytest.approx(0.04)
    assert Path(payload["report"]).exists()


def test_tiny_two_column_report_separates_local_es_and_column_em(tmp_path):
    output = tmp_path / "two_column"
    em = replace(
        _em_config(output, t_end_s=0.01),
        fac_source_model="symmetric_flux_tube",
        parallel_averaging_profile="dipole_chapman_flux_tube",
        parallel_averaging_length_km=None,
        hemisphere_count=2,
        right_boundary_model="matched",
        feedback_ramp_time_s=0.0,
        initial_characteristic_seed_v=0.0,
    )
    report = run_em_growth_experiment(_li_config(), em, resume=False)
    scope = report["scope"]
    assert scope["parallel_column_coefficients_are_independent"]
    assert scope["fac_forcing_representation"] == "surface_current"
    assert scope["coupled_density_conservation_measure"] == (
        "gravity_column_weight_times_density"
    )
    assert scope["baseline_density_conservation_measure"] == "local_density"
    assert scope["parallel_column_reduction_version"] == (
        "two_moment_surface_fac_column_continuity_v2"
    )
    geometry = scope["parallel_averaging_geometry"]
    assert len(geometry["gravity_column_weight_km"]) == _li_config().ny
    assert len(geometry["collision_column_weight_km"]) == _li_config().ny
    record = json.loads(
        (output / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["fac_forcing_representation"] == "surface_current"
    assert record["fac_forcing_units"] == "A m^-2"
    assert record["coupled_density_conservation_measure"] == (
        "gravity_column_weight_times_density"
    )


def _three_track_config(output: Path, *, t_end_s: float) -> EMGrowthConfig:
    return replace(
        _em_config(output, t_end_s=t_end_s),
        comparison_tracks="local_column_em",
        fac_source_model="symmetric_flux_tube",
        parallel_averaging_profile="dipole_chapman_flux_tube",
        parallel_averaging_length_km=None,
        hemisphere_count=2,
        right_boundary_model="matched",
        feedback_ramp_time_s=20.0,
        initial_line_state="zero",
        initial_characteristic_seed_v=0.0,
    )


def test_column_control_matches_zero_feedback_em_at_t0_direct_stage(tmp_path):
    li, _, coupled, basis, line = em_growth._build_models(
        _li_config(), _three_track_config(tmp_path / "unused", t_end_s=0.01)
    )
    density = em_growth.initial_density(li)
    zeros = np.zeros((line.cells, basis.mode_count), dtype=np.float64)
    state = DimensionalEMState(
        density.copy(), ModalAlfvenState(zeros.copy(), zeros.copy())
    )

    column_rhs, _ = coupled.evaluate_column_es_rhs(density)
    em_rhs, em_diagnostics = coupled.evaluate_rhs(0.0, state)
    assert em_diagnostics.effective_gain == 0.0
    np.testing.assert_array_equal(em_rhs.density_m3, column_rhs)

    dt_s = 1.0e-4
    np.testing.assert_array_equal(
        density + dt_s * em_rhs.density_m3,
        density + dt_s * column_rhs,
    )


def test_three_track_schema_resume_and_snapshots_are_atomic(tmp_path):
    resumed_output = tmp_path / "three_resumed"
    fresh_output = tmp_path / "three_fresh"
    initial = replace(
        _three_track_config(resumed_output, t_end_s=0.01),
        snapshot_interval_s=0.01,
    )
    first = run_em_growth_experiment(_li_config(), initial, resume=False)
    assert first["format"] == "esf_dimensional_em_growth_v2"
    assert first["scope"]["comparison_tracks"] == "local_column_em"
    with np.load(resumed_output / "checkpoint.npz", allow_pickle=False) as saved:
        assert int(saved["schema_version"]) == 3
        assert saved["column_control_density_m3"].shape == (9, 9)
        assert saved["column_control_amplitudes_m3"].shape == (2,)
    with np.load(
        resumed_output / "snapshots" / "snapshot_000000.npz",
        allow_pickle=False,
    ) as snapshot:
        assert int(snapshot["schema_version"]) == 2
        np.testing.assert_array_equal(
            snapshot["baseline_density_m3"],
            snapshot["column_control_density_m3"],
        )
        np.testing.assert_array_equal(
            snapshot["baseline_density_m3"], snapshot["coupled_density_m3"]
        )

    extended = run_em_growth_experiment(
        _li_config(), replace(initial, t_end_s=0.02), resume=True
    )
    fresh = run_em_growth_experiment(
        _li_config(),
        replace(initial, output_dir=str(fresh_output), t_end_s=0.02),
        resume=False,
    )
    assert extended["steps"] == fresh["steps"] == 2
    with np.load(resumed_output / "final_states.npz") as resumed, np.load(
        fresh_output / "final_states.npz"
    ) as direct:
        for key in (
            "baseline_density_m3",
            "column_control_density_m3",
            "coupled_density_m3",
            "coupled_w_plus_v",
            "coupled_w_minus_v",
        ):
            np.testing.assert_array_equal(resumed[key], direct[key])
    manifest = json.loads(
        (resumed_output / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == "esf_three_track_density_snapshots_v2"
    assert manifest["count"] == 3


def test_column_control_subcycles_dyadically_without_changing_macro_time():
    class SyntheticColumnControl:
        config = SimpleNamespace(cfl_limit=0.25)

        def evaluate_column_es_rhs(self, density):
            return np.ones_like(density), SimpleNamespace(
                cfl_rate=30.0,
                maximum_speed_x=0.0,
                maximum_speed_y=0.0,
                potential=SimpleNamespace(relative_residual=0.0),
            )

    density = np.ones((3, 4), dtype=np.float64)
    advanced = em_growth._advance_column_control_with_subcycling(
        SyntheticColumnControl(),
        time_s=2.0,
        density_m3=density,
        macro_dt_s=0.01,
        adaptive=True,
        minimum_dt_s=0.0025,
    )
    assert advanced.substeps == 2
    assert advanced.substep_dt_s == pytest.approx(0.005)
    assert advanced.maximum_substep_cfl == pytest.approx(0.15)
    np.testing.assert_allclose(advanced.density_m3, density + 0.01)


def test_rejected_density_trial_does_not_replace_last_stable_checkpoint(
    tmp_path, monkeypatch
):
    output = tmp_path / "rejected_trial"
    li = _li_config()

    class RejectingModel:
        def initial_alfven_state_from_li_potential(self, density):
            shape = (4, 1)
            return ModalAlfvenState(np.zeros(shape), np.zeros(shape))

        def ssprk3_step(self, time_s, state, dt_s):
            rejected = np.array(state.density_m3, copy=True)
            rejected[0, 0] = -1.0
            return DimensionalEMState(rejected, state.alfven), SimpleNamespace()

    rejecting = RejectingModel()
    basis = SimpleNamespace(mode_count=1)
    line = SimpleNamespace(maximum_stable_dt_s=1.0)
    monkeypatch.setattr(
        em_growth,
        "_build_models",
        lambda li_config, em_config: (
            li_config,
            rejecting,
            rejecting,
            basis,
            line,
        ),
    )
    config = replace(
        _em_config(output, t_end_s=0.01),
        dt_s=0.01,
        initial_characteristic_seed_v=0.0,
    )
    with pytest.raises(RuntimeError, match="non-positive"):
        run_em_growth_experiment(li, config, resume=False)

    with np.load(output / "checkpoint.npz", allow_pickle=False) as stable:
        assert float(stable["time_s"]) == 0.0
        assert int(stable["step"]) == 0
        assert np.min(stable["baseline_density_m3"]) > 0.0
        assert np.min(stable["coupled_density_m3"]) > 0.0
    with np.load(output / "failed_trial.npz", allow_pickle=False) as rejected:
        assert float(rejected["time_s"]) == pytest.approx(0.01)
        assert int(rejected["step"]) == 1
        assert np.min(rejected["baseline_density_m3"]) < 0.0
        assert np.min(rejected["coupled_density_m3"]) < 0.0


def _mock_em_diagnostics():
    return SimpleNamespace(
        electrostatic=SimpleNamespace(
            maximum_speed_x=0.0,
            maximum_speed_y=0.0,
            potential=SimpleNamespace(relative_residual=0.0),
        ),
        fac_closure=None,
        combined_cfl_rate_s_inv=0.0,
        effective_gain=0.0,
        activation_fraction=0.0,
        iterations=0,
        relative_coupling_residual=0.0,
        net_d_j_parallel_ds_a_m=0.0,
        relative_power_pairing_defect=0.0,
        boundary_power_modal_w=0.0,
        boundary_power_grid_w=0.0,
        remote_boundary_power_modal_w=0.0,
        remote_boundary_reflection_coefficient=-1.0,
    )


def test_cfl_rejection_halves_dt_and_resume_preserves_reduced_dt(
    tmp_path, monkeypatch
):
    output = tmp_path / "adaptive"
    attempted_dt: list[float] = []

    class CFLModel:
        def initial_alfven_state_from_li_potential(self, density):
            shape = (4, 1)
            return ModalAlfvenState(np.zeros(shape), np.zeros(shape))

        def ssprk3_step(self, time_s, state, dt_s):
            attempted_dt.append(float(dt_s))
            if dt_s > 0.005 + 1.0e-15:
                raise CFLViolation("synthetic CFL rejection")
            return state, _mock_em_diagnostics()

    model = CFLModel()
    monkeypatch.setattr(
        em_growth,
        "_build_models",
        lambda li_config, em_config: (
            li_config,
            model,
            model,
            SimpleNamespace(mode_count=1),
            SimpleNamespace(maximum_stable_dt_s=1.0, cells=4),
        ),
    )
    config = replace(
        _em_config(output, t_end_s=0.015),
        dt_s=0.01,
        minimum_dt_s=0.0025,
        initial_characteristic_seed_v=0.0,
    )
    report = run_em_growth_experiment(_li_config(), config, resume=False)
    assert attempted_dt[0] == pytest.approx(0.01)
    assert attempted_dt[1] == pytest.approx(0.005)
    assert report["dt_reductions"] == 1
    assert report["final_current_dt_s"] == pytest.approx(0.005)
    with np.load(output / "checkpoint.npz", allow_pickle=False) as checkpoint:
        assert int(checkpoint["schema_version"]) == 2
        assert float(checkpoint["current_dt_s"]) == pytest.approx(0.005)
        assert int(checkpoint["dt_reductions"]) == 1

    attempted_dt.clear()
    run_em_growth_experiment(
        _li_config(), replace(config, t_end_s=0.02), resume=True
    )
    assert attempted_dt[0] == pytest.approx(0.005)
    assert all(dt <= 0.005 + 1.0e-15 for dt in attempted_dt)


def test_baseline_only_cfl_subcycles_without_reducing_em_macro_dt_and_resumes(
    tmp_path, monkeypatch
):
    output = tmp_path / "baseline_subcycling"
    baseline_attempts: list[tuple[float, float]] = []
    coupled_attempts: list[tuple[float, float]] = []

    class BaselineOnlyCFLModel:
        def initial_alfven_state_from_li_potential(self, density):
            shape = (4, 1)
            return ModalAlfvenState(np.zeros(shape), np.zeros(shape))

        def ssprk3_step(self, time_s, state, dt_s):
            baseline_attempts.append((float(time_s), float(dt_s)))
            if dt_s > 0.005 + 1.0e-15:
                raise CFLViolation("synthetic baseline-only CFL rejection")
            advanced = DimensionalEMState(
                state.density_m3 + dt_s,
                state.alfven,
            )
            diagnostics = _mock_em_diagnostics()
            diagnostics.combined_cfl_rate_s_inv = 30.0
            return advanced, diagnostics

    class AcceptingCoupledModel:
        def ssprk3_step(self, time_s, state, dt_s):
            coupled_attempts.append((float(time_s), float(dt_s)))
            return state, _mock_em_diagnostics()

    baseline = BaselineOnlyCFLModel()
    coupled = AcceptingCoupledModel()
    monkeypatch.setattr(
        em_growth,
        "_build_models",
        lambda li_config, em_config: (
            li_config,
            baseline,
            coupled,
            SimpleNamespace(mode_count=1),
            SimpleNamespace(maximum_stable_dt_s=1.0, cells=4),
        ),
    )
    initial = replace(
        _em_config(output, t_end_s=0.01),
        dt_s=0.01,
        minimum_dt_s=0.0025,
        snapshot_interval_s=0.01,
        initial_line_state="zero",
        initial_characteristic_seed_v=0.0,
    )
    first = run_em_growth_experiment(_li_config(), initial, resume=False)

    assert first["steps"] == 1
    assert first["final_current_dt_s"] == pytest.approx(0.01)
    assert first["dt_reductions"] == 0
    assert first["baseline_subcycling"]["events"] == 1
    assert first["baseline_subcycling"]["extra_baseline_steps"] == 1
    assert [dt for _, dt in coupled_attempts] == pytest.approx([0.01])
    assert [dt for _, dt in baseline_attempts] == pytest.approx(
        [0.01, 0.005, 0.005]
    )

    baseline_attempts.clear()
    coupled_attempts.clear()
    extended = run_em_growth_experiment(
        _li_config(), replace(initial, t_end_s=0.02), resume=True
    )
    assert extended["steps"] == 2
    assert extended["final_current_dt_s"] == pytest.approx(0.01)
    assert extended["dt_reductions"] == 0
    assert extended["baseline_subcycling"]["events"] == 2
    assert extended["baseline_subcycling"]["extra_baseline_steps"] == 2
    assert [dt for _, dt in coupled_attempts] == pytest.approx([0.01])
    assert [dt for _, dt in baseline_attempts] == pytest.approx(
        [0.01, 0.005, 0.005]
    )

    with np.load(output / "checkpoint.npz", allow_pickle=False) as checkpoint:
        assert float(checkpoint["time_s"]) == pytest.approx(0.02)
        assert float(checkpoint["current_dt_s"]) == pytest.approx(0.01)
    manifest = json.loads(
        (output / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["count"] == 3
    assert manifest["last_time_s"] == pytest.approx(0.02)
    diagnostics = [
        json.loads(line)
        for line in (output / "diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["baseline_substeps"] for record in diagnostics] == [2, 2]
    assert [record["baseline_cfl"] for record in diagnostics] == pytest.approx(
        [0.15, 0.15]
    )
    events = [
        json.loads(line)
        for line in (output / "timestep_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["event"] for record in events] == [
        "baseline_subcycling",
        "baseline_subcycling",
    ]


def test_baseline_subcycling_pauses_at_floor_without_calling_or_reducing_em(
    tmp_path, monkeypatch
):
    output = tmp_path / "baseline_floor"
    baseline_attempts: list[float] = []
    coupled_attempts: list[float] = []

    class RejectingBaseline:
        def ssprk3_step(self, time_s, state, dt_s):
            baseline_attempts.append(float(dt_s))
            raise CFLViolation("synthetic persistent baseline CFL rejection")

    class CoupledMustNotRun:
        def ssprk3_step(self, time_s, state, dt_s):
            coupled_attempts.append(float(dt_s))
            return state, _mock_em_diagnostics()

    monkeypatch.setattr(
        em_growth,
        "_build_models",
        lambda li_config, em_config: (
            li_config,
            RejectingBaseline(),
            CoupledMustNotRun(),
            SimpleNamespace(mode_count=1),
            SimpleNamespace(maximum_stable_dt_s=1.0, cells=4),
        ),
    )
    config = replace(
        _em_config(output, t_end_s=0.01),
        dt_s=0.01,
        minimum_dt_s=0.005,
        initial_line_state="zero",
        initial_characteristic_seed_v=0.0,
    )
    with pytest.raises(MinimumTimeStepError) as raised:
        run_em_growth_experiment(_li_config(), config, resume=False)
    assert raised.value.component == "electrostatic baseline"
    assert baseline_attempts == pytest.approx([0.01, 0.005])
    assert coupled_attempts == []

    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "paused_minimum_dt"
    assert status["failed_component"] == "electrostatic baseline"
    assert status["current_dt_s"] == pytest.approx(0.01)
    assert status["dt_reductions"] == 0
    with np.load(output / "checkpoint.npz", allow_pickle=False) as checkpoint:
        assert float(checkpoint["time_s"]) == 0.0
        assert float(checkpoint["current_dt_s"]) == pytest.approx(0.01)
        assert int(checkpoint["dt_reductions"]) == 0


def test_coupling_failure_at_minimum_dt_pauses_with_stable_checkpoint(
    tmp_path, monkeypatch
):
    output = tmp_path / "minimum_dt"

    class NonconvergingModel:
        def initial_alfven_state_from_li_potential(self, density):
            shape = (4, 1)
            return ModalAlfvenState(np.zeros(shape), np.zeros(shape))

        def ssprk3_step(self, time_s, state, dt_s):
            raise PhysicalCouplingConvergenceError(1, 1.0)

    model = NonconvergingModel()
    monkeypatch.setattr(
        em_growth,
        "_build_models",
        lambda li_config, em_config: (
            li_config,
            model,
            model,
            SimpleNamespace(mode_count=1),
            SimpleNamespace(maximum_stable_dt_s=1.0, cells=4),
        ),
    )
    config = replace(
        _em_config(output, t_end_s=0.01),
        dt_s=0.01,
        minimum_dt_s=0.005,
        initial_characteristic_seed_v=0.0,
    )
    with pytest.raises(MinimumTimeStepError):
        run_em_growth_experiment(_li_config(), config, resume=False)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "paused_minimum_dt"
    assert status["current_dt_s"] == pytest.approx(0.005)
    with np.load(output / "checkpoint.npz", allow_pickle=False) as checkpoint:
        assert float(checkpoint["time_s"]) == 0.0
        assert float(checkpoint["current_dt_s"]) == pytest.approx(0.005)
        assert int(checkpoint["dt_reductions"]) == 1


def test_paired_snapshots_land_on_exact_times_and_extend_contiguously(tmp_path):
    output = tmp_path / "snapshots"
    initial = replace(
        _em_config(output, t_end_s=0.04), snapshot_interval_s=0.015
    )
    first = run_em_growth_experiment(_li_config(), initial, resume=False)
    assert first["snapshots"]["count"] == 3
    for index, expected_time in enumerate((0.0, 0.015, 0.03)):
        with np.load(
            output / "snapshots" / f"snapshot_{index:06d}.npz",
            allow_pickle=False,
        ) as snapshot:
            assert float(snapshot["time_s"]) == expected_time
            assert snapshot["baseline_density_m3"].dtype == np.float64
            assert snapshot["coupled_density_m3"].dtype == np.float64

    extended = run_em_growth_experiment(
        _li_config(), replace(initial, t_end_s=0.06), resume=True
    )
    assert extended["snapshots"]["count"] == 5
    manifest = json.loads(
        (output / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["count"] == 5
    assert manifest["last_time_s"] == pytest.approx(0.06)
    with np.load(
        output / "snapshots" / "snapshot_000004.npz", allow_pickle=False
    ) as snapshot:
        assert float(snapshot["time_s"]) == 0.06


def test_terminal_diagnostic_written_off_interval(tmp_path):
    output = tmp_path / "terminal_off_interval"
    config = replace(
        _three_track_config(output, t_end_s=0.03), diagnostics_interval_steps=4
    )
    report = run_em_growth_experiment(_li_config(), config)
    records = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[-1]["step"] == report["steps"] == 3
    assert records[-1]["time_s"] == report["completed_time_s"]
    assert records[-1]["diagnostic_sampling"] == "accepted_step_representative_stage"


def test_completed_checkpoint_repairs_missing_terminal_without_advancing(tmp_path, monkeypatch):
    output = tmp_path / "terminal_repair"
    config = replace(
        _three_track_config(output, t_end_s=0.03), diagnostics_interval_steps=2
    )
    report = run_em_growth_experiment(_li_config(), config)
    checkpoint = output / "checkpoint.npz"
    with np.load(checkpoint, allow_pickle=False) as saved:
        original = {key: saved[key].copy() for key in saved.files}
    diagnostic_path = output / "diagnostics.jsonl"
    lines = diagnostic_path.read_text().splitlines()
    assert len(lines) == 2
    diagnostic_path.write_text(lines[0] + "\n")

    def forbidden_step(*args, **kwargs):
        raise AssertionError("completed checkpoint recovery must not advance")

    monkeypatch.setattr(em_growth.DimensionalBidirectionalModel, "ssprk3_step", forbidden_step)
    repaired = run_em_growth_experiment(_li_config(), config)
    assert repaired["configuration_hash"] == report["configuration_hash"]
    with np.load(checkpoint, allow_pickle=False) as saved:
        for key, value in original.items():
            np.testing.assert_array_equal(saved[key], value)
    records = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[-1]["step"] == 3 and records[-1]["time_s"] == 0.03
    assert records[-1]["diagnostic_sampling"] == "reconstructed_terminal_state_rhs_not_stage_maximum"
    assert records[-1]["column_control_amplitude_m3"] > 0
    once = diagnostic_path.read_bytes()
    run_em_growth_experiment(_li_config(), config)
    assert diagnostic_path.read_bytes() == once
