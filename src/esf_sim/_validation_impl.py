"""Validation implementation with the paper's full-depth cylinder slot."""

from __future__ import annotations

import numpy as np

from . import _validation_impl_original as _original
from ._validation_impl_width import _transition_width


_ssprk3 = _original._ssprk3
elliptic_manufactured_solution_test = _original.elliptic_manufactured_solution_test


def slotted_cylinder_initial(
    ncells: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the paper/ScienceDB Zalesak cylinder and full-depth slot.

    The slot is open from its top edge at ``y=0.85`` down through the
    cylinder.  The previous implementation accidentally stopped it at
    ``y=0.75``, leaving the lower half of the slot filled.
    """

    coordinates = np.linspace(0.0, 1.0, ncells)
    x, y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(x - 0.5, y - 0.75)
    inside_cylinder = radius <= 0.15
    slot = (np.abs(x - 0.5) <= 0.03) & (y <= 0.85)
    density = np.full_like(x, 1.0e-10)
    density[inside_cylinder & ~slot] = 1.0
    return density, x, y


# The original driver resolves these helpers through its module globals.
_original._transition_width = _transition_width
_original.slotted_cylinder_initial = slotted_cylinder_initial
circular_advection_test = _original.circular_advection_test

