from __future__ import annotations

import json

import numpy as np

from esf_sim.config import SimulationConfig
from esf_sim.figure3_extension import ContinuationFrameArchive


def test_continuation_archive_starts_at_nonzero_time(tmp_path):
    config = SimulationConfig(t_end_s=12.0)
    archive = ContinuationFrameArchive(
        tmp_path / "frames",
        config,
        start_time_s=10.0,
        chunk_frames=2,
    )
    density = np.ones((config.ny, config.nx))
    archive.prepare(resume_time_s=10.0, density=density, resume=True)
    assert archive.next_time_s == 11.0
    archive.capture(11.0, density * 2.0)
    archive.capture(12.0, density * 3.0)
    report = archive.finalize(completed=True)
    manifest = json.loads(archive.manifest_path.read_text(encoding="utf-8"))
    assert report["frame_count"] == 3
    assert manifest["first_time_s"] == 10.0
    assert manifest["target_time_s"] == 12.0
    assert manifest["completed"] is True
