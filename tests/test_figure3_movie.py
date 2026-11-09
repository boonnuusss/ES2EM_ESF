from pathlib import Path

from esf_sim.figure3_movie import CASES, SequentialArchive


def test_movie_case_contract():
    assert [case.alpha for case in CASES] == [0.2, 0.4, 0.6, 0.8]
    assert [case.target_time_s for case in CASES] == [3150, 3150, 3050, 2860]
    assert [case.recovered for case in CASES] == [True, False, False, False]


def test_movie_archives_have_complete_manifests():
    root = Path("output/best_baseline/figure3_v2_timeseries")
    for case in CASES:
        archive = SequentialArchive(root / case.relative_directory, case)
        assert archive.manifest["completed"] is True
        assert archive.manifest["frame_count"] == case.target_time_s + 1
        assert archive.manifest["density_shape"] == [401, 201]


def test_movie_archive_holds_final_frame_after_case_target():
    root = Path("output/best_baseline/figure3_v2_timeseries")
    case = CASES[-1]
    archive = SequentialArchive(root / case.relative_directory, case)
    final = archive.frame(case.target_time_s)
    held = archive.frame(3150)
    assert final.shape == (401, 201)
    assert (final == held).all()
