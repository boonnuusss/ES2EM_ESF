from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from esf_sim.config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
    SimulationConfig,
)
from esf_sim.em_growth import (
    EMGrowthConfig,
    _configuration_hash,
    _parallel_averaging_geometry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exploratory_alpha04_configs_are_consistent_and_conservative():
    li = SimulationConfig.from_json(
        ROOT / "configs" / "em_growth_li_101x201_alpha04.json"
    )
    em = EMGrowthConfig.from_json(
        ROOT / "configs" / "em_growth_reduced_em_101x201_alpha04.json"
    )

    li.validate()
    assert (li.nx, li.ny, li.pdm_alpha) == (101, 201, 0.4)
    assert li.potential_operator == MIMETIC_HARMONIC_LOG_RHS_OPERATOR
    assert li.potential_boundary_y == BEST_BASELINE_POTENTIAL_BOUNDARY
    assert li.relative_noise == 0.0
    assert _configuration_hash(li, em) == (
        "bf473a242fd8cbb42419c3c90e4eeb519f9ae6d622d7410e67008a005390adb0"
    )

    assert em.closure == "mimetic"
    assert em.coupling_solver == "modal_woodbury"
    assert em.use_pardiso
    assert em.field_line_cells == 32
    assert em.initial_characteristic_seed_v == 0.0
    assert "exploratory" in em.experiment_label
    assert em.output_dir == li.output_dir
    assert em.t_end_s == li.t_end_s == pytest.approx(3150.0)

    ds_km = em.field_line_length_km / em.field_line_cells
    crossing_time_s = ds_km / em.alfven_speed_km_s
    assert em.dt_s / crossing_time_s == pytest.approx(em.line_cfl_limit)
    assert em.line_cfl_limit == pytest.approx(0.4)
    assert em.fac_source_model == "symmetric_flux_tube"
    assert em.parallel_averaging_length_km == pytest.approx(625.0)
    assert em.hemisphere_count == 2
    assert em.initial_line_state == "zero"
    assert em.adaptive_dt_halving
    assert em.minimum_dt_s == pytest.approx(1.0 / 64.0)
    assert em.snapshot_interval_s == pytest.approx(10.0)
    assert em.right_boundary_model == "fixed_voltage"
    assert em.right_boundary_conductance_s is None


def test_matched_remote_candidate_changes_only_terminal_and_output_identity():
    fixed = EMGrowthConfig.from_json(
        ROOT / "configs" / "em_growth_reduced_em_101x201_alpha04.json"
    )
    matched = EMGrowthConfig.from_json(
        ROOT / "configs" / "em_growth_reduced_em_101x201_alpha04_matched.json"
    )
    ignored = {
        "right_boundary_model",
        "right_boundary_conductance_s",
        "output_dir",
        "experiment_label",
        "comparison_tracks",
    }
    fixed_payload = fixed.to_dict()
    matched_payload = matched.to_dict()
    for key in ignored:
        fixed_payload.pop(key)
        matched_payload.pop(key)
    assert fixed_payload == matched_payload
    assert matched.right_boundary_model == "matched"
    assert matched.right_boundary_voltage_v == 0.0
    assert matched.right_boundary_conductance_s is None


def test_matched_dipole_chapman_candidate_changes_only_column_geometry():
    matched = EMGrowthConfig.from_json(
        ROOT / "configs" / "em_growth_reduced_em_101x201_alpha04_matched.json"
    )
    dipole = EMGrowthConfig.from_json(
        ROOT
        / "configs"
        / "em_growth_reduced_em_101x201_alpha04_matched_dipole_chapman.json"
    )
    ignored = {
        "parallel_averaging_length_km",
        "parallel_averaging_profile",
        "dipole_earth_radius_km",
        "dipole_column_lower_altitude_km",
        "dipole_quadrature_order",
        "output_dir",
        "experiment_label",
        "comparison_tracks",
    }
    matched_payload = matched.to_dict()
    dipole_payload = dipole.to_dict()
    for key in ignored:
        matched_payload.pop(key)
        dipole_payload.pop(key)
    assert matched_payload == dipole_payload
    assert dipole.parallel_averaging_profile == "dipole_chapman_flux_tube"
    assert dipole.comparison_tracks == "local_column_em"
    assert dipole.parallel_averaging_length_km is None
    assert dipole.dipole_column_lower_altitude_km == pytest.approx(100.0)
    assert dipole.dipole_quadrature_order == 128
    li = SimulationConfig.from_json(
        ROOT / "configs" / "em_growth_li_101x201_alpha04.json"
    )
    gravity_m, collision_m, metadata = _parallel_averaging_geometry(li, dipole)
    assert gravity_m.shape == (li.ny,)
    assert collision_m.shape == (li.ny,)
    assert gravity_m[0] / 1000.0 == pytest.approx(317.33664714993546)
    assert gravity_m[94] / 1000.0 == pytest.approx(1327.253718258658)
    assert gravity_m[-1] / 1000.0 == pytest.approx(9639.779397734665)
    assert collision_m[0] / 1000.0 == pytest.approx(326.704187561715)
    assert collision_m[94] / 1000.0 == pytest.approx(1928.053732447361)
    assert collision_m[-1] / 1000.0 == pytest.approx(100052.88742973867)
    assert len(metadata["altitude_km"]) == li.ny
    assert len(metadata["gravity_column_weight_km"]) == li.ny
    assert len(metadata["collision_column_weight_km"]) == li.ny
    assert len(metadata["background_density_column_m2"]) == li.ny
    assert len(metadata["background_collision_column_m2_s"]) == li.ny
    assert len(metadata["collision_moment_m_s"]) == li.ny
    assert len(metadata["effective_collision_frequency_s"]) == li.ny
    assert _configuration_hash(
        li, replace(dipole, comparison_tracks="legacy_pair")
    ) == (
        "fdae2aeb6b74cf0eceb34eeb32edda76fd1ff79474998c22cea023a5ca3ab65d"
    )
    assert _configuration_hash(li, dipole) != _configuration_hash(
        li, replace(dipole, comparison_tracks="legacy_pair")
    )
