import hashlib
import json
from pathlib import Path

import pytest

from esf_sim import _model_impl
from esf_sim._figure3_impl import _snapshot_matches
from esf_sim.config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    BEST_BASELINE_POTENTIAL_OPERATOR,
    CONSERVATIVE_HARMONIC_OPERATOR,
    DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
    SimulationConfig,
)
from esf_sim.point_centered_potential import (
    EqualExtrapolatedLogPointCenteredPotentialSolver,
    LogPointCenteredPotentialSolver,
)
from esf_sim.potential import PotentialSolver


def test_official_configs_select_promoted_operator_and_boundary():
    for path in (
        "configs/figure2.json",
        "configs/figure3.json",
        "configs/figure6.json",
        "configs/best_baseline.json",
    ):
        config = SimulationConfig.from_json(path)
        config.validate()
        assert config.potential_operator == BEST_BASELINE_POTENTIAL_OPERATOR
        assert config.potential_boundary_y == BEST_BASELINE_POTENTIAL_BOUNDARY
        assert "best_baseline" in config.output_dir


def test_legacy_configs_preserve_their_operator_and_boundary():
    for path in (
        "configs/figure2_legacy_harmonic.json",
        "configs/figure3_legacy_harmonic.json",
        "configs/figure6_legacy_harmonic.json",
    ):
        config = SimulationConfig.from_json(path)
        assert config.potential_operator == CONSERVATIVE_HARMONIC_OPERATOR
        assert config.potential_boundary_y == DIRICHLET_ZERO_POTENTIAL_BOUNDARY
    for path in (
        "configs/figure2_legacy_point_centered_dirichlet.json",
        "configs/figure3_legacy_point_centered_dirichlet.json",
        "configs/figure6_legacy_point_centered_dirichlet.json",
        "configs/best_baseline_v1_dirichlet.json",
    ):
        config = SimulationConfig.from_json(path)
        assert config.potential_operator == BEST_BASELINE_POTENTIAL_OPERATOR
        assert config.potential_boundary_y == DIRICHLET_ZERO_POTENTIAL_BOUNDARY


def test_production_model_dispatches_selected_potential_solver():
    small = SimulationConfig(nx=9, ny=9, x_max_km=10.0, y_max_km=258.0)
    harmonic = _model_impl.ESFModel(small)
    point_dirichlet = _model_impl.ESFModel(
        small.with_updates(potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR)
    )
    point_extrapolated = _model_impl.ESFModel(
        small.with_updates(
            potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
            potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        )
    )
    assert isinstance(harmonic.potential_solver, PotentialSolver)
    assert type(point_dirichlet.potential_solver) is LogPointCenteredPotentialSolver
    assert type(point_extrapolated.potential_solver) is (
        EqualExtrapolatedLogPointCenteredPotentialSolver
    )


def test_unknown_potential_operator_or_boundary_is_rejected():
    with pytest.raises(ValueError, match="potential_operator"):
        SimulationConfig(potential_operator="unknown")
    with pytest.raises(ValueError, match="potential_boundary_y"):
        SimulationConfig(potential_boundary_y="unknown")


def test_promoted_snapshot_and_manifest_are_consistent():
    manifest_path = Path("output/best_baseline/baseline_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = Path(manifest["selected_snapshot"]["path"])
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert manifest["status"] == "active_best_baseline"
    assert (
        manifest["potential_discretization"]["operator"]
        == BEST_BASELINE_POTENTIAL_OPERATOR
    )
    assert (
        manifest["potential_discretization"]["vertical_potential_boundary"]
        == BEST_BASELINE_POTENTIAL_BOUNDARY
    )
    assert digest == manifest["selected_snapshot"]["sha256"]
    assert _snapshot_matches(
        snapshot,
        3150.0,
        BEST_BASELINE_POTENTIAL_OPERATOR,
        BEST_BASELINE_POTENTIAL_BOUNDARY,
    )
    assert not _snapshot_matches(
        snapshot,
        3150.0,
        BEST_BASELINE_POTENTIAL_OPERATOR,
        DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
    )