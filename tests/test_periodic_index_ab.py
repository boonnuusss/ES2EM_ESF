import numpy as np

from esf_sim.config import SimulationConfig
from esf_sim.periodic_index_ab import (
    ACTIVE_HORIZONTAL_CELLS,
    add_four_external_ghosts,
    author_index_evidence,
    b_active_config,
    expand_author_interior,
)


def test_author_201_storage_indices_are_exact() -> None:
    active = np.arange(ACTIVE_HORIZONTAL_CELLS, dtype=float)[None, :]
    interior = expand_author_interior(active)
    stored = add_four_external_ghosts(active)

    assert interior.shape == (1, 201)
    assert stored.shape == (1, 209)
    np.testing.assert_array_equal(interior[..., :198], active)
    np.testing.assert_array_equal(interior[..., 198:], active[..., :3])
    np.testing.assert_array_equal(stored[..., :4], interior[..., 194:198])
    np.testing.assert_array_equal(stored[..., 4:202], active)
    np.testing.assert_array_equal(stored[..., 202:], active[..., :7])


def test_b_changes_only_horizontal_active_layout(tmp_path) -> None:
    base = SimulationConfig()
    calibrated = b_active_config(base, tmp_path)
    calibrated.validate()

    assert calibrated.nx == 198
    assert calibrated.x_min_km == 0.0
    assert calibrated.x_max_km == 246.25
    assert calibrated.ny == base.ny
    assert calibrated.dx_km == base.dx_km
    assert calibrated.dy_km == base.dy_km
    assert calibrated.pdm_alpha == 0.4
    assert calibrated.t_end_s == 3150.0

    excluded = {
        "nx",
        "x_min_km",
        "x_max_km",
        "pdm_alpha",
        "t_end_s",
        "output_dir",
    }
    base_dict = base.to_dict()
    calibrated_dict = calibrated.to_dict()
    for key in base_dict.keys() - excluded:
        assert calibrated_dict[key] == base_dict[key]


def test_public_author_arrays_confirm_copy_relations() -> None:
    evidence = author_index_evidence("data/reference/numDensity.mat")
    assert {item["active_horizontal_count"] for item in evidence.values()} == {
        98,
        198,
        298,
        398,
    }
    assert all(item["external_copy_relations_exact"] for item in evidence.values())
    assert all(
        item["left_copy_max_abs_error_m3"] == 0.0
        and item["right_copy_max_abs_error_m3"] == 0.0
        for item in evidence.values()
    )
