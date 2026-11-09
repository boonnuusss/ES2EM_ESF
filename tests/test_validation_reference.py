from pathlib import Path

import numpy as np
from scipy.io import loadmat

from esf_sim.validation import slotted_cylinder_initial


def test_slotted_cylinder_matches_sciencedb_initial_array() -> None:
    density, x, y = slotted_cylinder_initial(256)
    reference = loadmat(Path("data/reference/circAdvecData.mat"))

    np.testing.assert_array_equal(density, reference["n0"].T)
    np.testing.assert_array_equal(x, reference["x"].T)
    np.testing.assert_array_equal(y, reference["y"].T)


def test_slot_extends_below_cylinder_center() -> None:
    density, x, y = slotted_cylinder_initial(256)
    in_lower_slot = (
        (np.abs(x - 0.5) <= 0.03)
        & (y < 0.75)
        & (np.hypot(x - 0.5, y - 0.75) <= 0.15)
    )

    assert np.any(in_lower_slot)
    assert np.all(density[in_lower_slot] == 1.0e-10)

