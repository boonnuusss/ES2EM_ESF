import numpy as np

from esf_sim.config import SimulationConfig
from esf_sim.model import initial_density


def test_paper_grid_and_chapman_minimum():
    config = SimulationConfig()
    config.validate()
    density = initial_density(config)
    assert density.shape == (401, 201)
    assert np.isclose(density.max(), 1.1e12, rtol=1.0e-12)
    assert np.isclose(density.min(), 1.5289848e7, rtol=2.0e-6)


def test_noise_is_deterministic_and_relative():
    config = SimulationConfig(relative_noise=1.0e-12)
    first = initial_density(config)
    second = initial_density(config)
    assert np.array_equal(first, second)
    baseline = initial_density(config.with_updates(relative_noise=0.0))
    assert np.linalg.norm(first - baseline) / np.linalg.norm(baseline) < 2.0e-12
