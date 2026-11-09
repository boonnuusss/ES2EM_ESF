"""Read-only handling of the authors' ScienceDB acceptance data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.io import loadmat


REFERENCE_MD5 = {
    "ReadMe.txt": "A922F4117B0CECFD29566CD26AF017CA",
    "circAdvecData.mat": "BF0542002E253F498C06498DFF54381E",
    "numDensity.mat": "F052A4FCEA234ADED59DA7023552D739",
}

FIGURE2_REFERENCE_VARIABLES = {
    "first_order": ("n2_upwind1_2800", 2800.0),
    "van_leer": ("n2_vanLeer2_3150", 3150.0),
    "pdm7": ("n2_02_3150", 3150.0),
}


def md5sum(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_reference_files(directory: str | Path) -> dict[str, dict[str, object]]:
    root = Path(directory)
    report: dict[str, dict[str, object]] = {}
    for name, expected in REFERENCE_MD5.items():
        path = root / name
        actual = md5sum(path) if path.exists() else None
        report[name] = {
            "exists": path.exists(),
            "expected_md5": expected,
            "actual_md5": actual,
            "matches": actual == expected,
        }
    return report


def load_figure2_reference(
    path: str | Path,
) -> dict[str, tuple[np.ndarray, float, str]]:
    """Load physical cells only, transposed to solver order (ny, nx)."""

    raw: Mapping[str, np.ndarray] = loadmat(path)
    result: dict[str, tuple[np.ndarray, float, str]] = {}
    for scheme, (variable, time_s) in FIGURE2_REFERENCE_VARIABLES.items():
        if variable not in raw:
            raise KeyError(f"{variable!r} is missing from {path}")
        with_ghosts = np.asarray(raw[variable], dtype=np.float64)
        if with_ghosts.shape != (209, 409):
            raise ValueError(
                f"{variable} has shape {with_ghosts.shape}; expected (209, 409)"
            )
        density = with_ghosts[4:-4, 4:-4].T.copy()
        result[scheme] = (density, time_s, variable)
    return result
