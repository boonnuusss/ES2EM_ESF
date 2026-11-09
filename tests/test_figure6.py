from pathlib import Path

import numpy as np
from scipy.io import whosmat

from esf_sim.config import SimulationConfig
from esf_sim.figure6 import FIGURE6_CASES, case_config, depletion_finger_diagnostic


def test_figure6_grids_span_the_same_physical_domain() -> None:
    base = SimulationConfig.from_json("configs/figure6.json")
    for case in FIGURE6_CASES:
        config = case_config(base, case, Path("output/test"))
        config.validate()
        assert np.isclose(config.dx_km, 250.0 / (case.nx - 1))
        assert np.isclose(config.dy_km, 400.0 / (case.ny - 1))
        np.testing.assert_allclose(config.x_coordinates_m[[0, -1]] / 1.0e3, [0, 250])
        np.testing.assert_allclose(config.y_coordinates_m[[0, -1]] / 1.0e3, [250, 650])


def test_figure6_reference_shapes_include_four_ghosts() -> None:
    shapes = {name: shape for name, shape, _ in whosmat("data/reference/numDensity.mat")}
    for case in FIGURE6_CASES:
        assert shapes[case.reference_variable] == (case.nx + 8, case.ny + 8)


def test_depletion_finger_diagnostic_detects_multiple_troughs() -> None:
    x = np.linspace(-10.0, 10.0, 101)
    y = np.linspace(350.0, 520.0, 31)
    density = np.ones((y.size, x.size))
    for center in (-6.0, -2.0, 2.0, 6.0):
        density -= 0.8 * np.exp(-((x - center) / 0.5) ** 2)[None, :]
    result = depletion_finger_diagnostic(density, x, y)
    assert result["maximum_depletion_fingers_per_row"] == 4
    assert np.isclose(result["median_detected_spacing_km"], 4.0)
