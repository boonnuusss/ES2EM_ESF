import json

import numpy as np

from esf_sim.config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    BEST_BASELINE_POTENTIAL_OPERATOR,
    SimulationConfig,
)
from esf_sim.figure3_timeseries import validate_v2_baseline
from esf_sim.timeseries import ChunkedFrameArchive, load_frame, validate_frame_archive


def small_config(tmp_path, *, t_end_s=3.0):
    return SimulationConfig(
        nx=8,
        ny=8,
        x_min_km=0.0,
        x_max_km=7.0,
        y_min_km=0.0,
        y_max_km=7.0,
        dx_km=1.0,
        dy_km=1.0,
        t_end_s=t_end_s,
        potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        output_dir=str(tmp_path),
    )


def test_chunked_archive_contains_every_second(tmp_path):
    config = small_config(tmp_path)
    root = tmp_path / "frames"
    archive = ChunkedFrameArchive(root, config, chunk_frames=2)
    field = np.arange(64, dtype=np.float64).reshape(8, 8)
    archive.prepare(resume_time_s=0.0, density=field, resume=False)
    for time_s in (1.0, 2.0, 3.0):
        archive.capture(time_s, field + time_s)
    report = archive.finalize(completed=True)

    assert report["valid"]
    assert report["frame_count"] == 4
    assert report["one_second_spacing"]
    assert report["storage_dtype"] == "float32"
    np.testing.assert_array_equal(load_frame(root, 2), (field + 2).astype(np.float32))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"]
    assert manifest["frame_count"] == 4


def test_resume_discards_frames_beyond_checkpoint(tmp_path):
    config = small_config(tmp_path)
    root = tmp_path / "frames"
    field = np.ones((8, 8), dtype=np.float64)
    archive = ChunkedFrameArchive(root, config, chunk_frames=2)
    archive.prepare(resume_time_s=0.0, density=field, resume=False)
    archive.capture(1.0, field)
    archive.capture(2.0, field)
    archive.finalize(completed=False)

    resumed = ChunkedFrameArchive(root, config, chunk_frames=2)
    resumed.prepare(resume_time_s=1.0, density=field, resume=True)
    assert resumed.next_time_s == 2.0
    resumed.capture(2.0, field * 2)
    resumed.capture(3.0, field * 3)
    resumed.finalize(completed=True)
    report = validate_frame_archive(
        root,
        expected_times=np.arange(4, dtype=np.float64),
        expected_shape=(8, 8),
    )
    assert report["valid"]
    np.testing.assert_array_equal(load_frame(root, 2), np.full((8, 8), 2.0))


def test_figure3_timeseries_rejects_mimetic_operator():
    valid = SimulationConfig(
        potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
    )
    validate_v2_baseline(valid)
    invalid = valid.with_updates(potential_operator="mimetic_harmonic_log_rhs")
    try:
        validate_v2_baseline(invalid)
    except ValueError as error:
        assert "formal v2 baseline" in str(error)
    else:
        raise AssertionError("mimetic operator was not rejected")
