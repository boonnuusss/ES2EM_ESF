import numpy as np
from scipy.io import loadmat

from esf_sim.figure3 import FIGURE3_CASES


def test_figure3_case_contract():
    assert [case.alpha for case in FIGURE3_CASES] == [0.2, 0.4, 0.6, 0.8]
    assert [case.time_s for case in FIGURE3_CASES] == [3150.0, 3150.0, 3050.0, 2860.0]
    assert [case.panel_label for case in FIGURE3_CASES] == ["a)", "b)", "c)", "d)"]


def test_figure3_reference_arrays_have_expected_physical_shape():
    raw = loadmat("data/reference/numDensity.mat")
    for case in FIGURE3_CASES:
        with_ghosts = np.asarray(raw[case.reference_variable])
        assert with_ghosts.shape == (209, 409)
        assert with_ghosts[4:-4, 4:-4].T.shape == (401, 201)
