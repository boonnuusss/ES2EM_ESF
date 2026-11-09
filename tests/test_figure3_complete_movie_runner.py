from esf_sim.figure3_complete_movie import DEFAULT_ROOT
from esf_sim.figure3_complete_movie_runner import workflow


def test_seekable_reader_can_return_to_zero_after_seam_check():
    timelines = workflow.prepare_timelines(DEFAULT_ROOT)
    for timeline in timelines.values():
        assert timeline.frame(0).shape == (401, 201)
