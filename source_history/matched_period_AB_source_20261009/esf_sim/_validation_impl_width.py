"""Internal validation implementation with a documented jump-width metric."""

from __future__ import annotations

import numpy as np

from . import _validation_impl_original as _original


_ssprk3 = _original._ssprk3
elliptic_manufactured_solution_test = _original.elliptic_manufactured_solution_test
slotted_cylinder_initial = _original.slotted_cylinder_initial


def _transition_width(profile: np.ndarray, coordinate: np.ndarray) -> int:
    """Count cells between the 1% and 99% levels of one cylinder edge."""

    candidates = np.where((profile > 0.01) & (profile < 0.99) & (coordinate < 0.48))[0]
    if not candidates.size:
        return 0
    groups = np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1)
    return int(max(group.size for group in groups))


_original._transition_width = _transition_width
circular_advection_test = _original.circular_advection_test
