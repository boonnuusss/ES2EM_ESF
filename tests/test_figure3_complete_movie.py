from __future__ import annotations

import numpy as np

from esf_sim.figure3_complete_movie import DEFAULT_ROOT, prepare_timelines


def test_complete_movie_archive_seams_and_terminal_holds():
    timelines = prepare_timelines(DEFAULT_ROOT)
    assert all(timeline.seam_exact for timeline in timelines.values())
    assert timelines["alpha_0p2"].last_integer_time_s == 4150
    assert timelines["alpha_0p4"].last_integer_time_s == 4150
    assert timelines["alpha_0p6"].last_integer_time_s == 3063
    assert timelines["alpha_0p8"].last_integer_time_s == 2874
    assert not timelines["alpha_0p6"].is_held(3063)
    assert timelines["alpha_0p6"].is_held(3064)
    assert np.array_equal(
        timelines["alpha_0p6"].frame(3064),
        timelines["alpha_0p6"].terminal_density,
    )
