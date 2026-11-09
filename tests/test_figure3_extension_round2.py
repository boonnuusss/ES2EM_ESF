from __future__ import annotations

import math
import numpy as np

from esf_sim.figure3_extension_round2 import round2_cases


def test_round2_sources_are_complete_round1_checkpoints():
    for case in round2_cases():
        assert case.checkpoint_path.exists()
        with np.load(case.checkpoint_path, allow_pickle=False) as saved:
            assert math.isclose(float(saved["time_s"]), 4150.0, abs_tol=1.0e-9)
            assert int(saved["history_length"]) == 3
        assert case.target_time_s - case.start_time_s == 1000.0
