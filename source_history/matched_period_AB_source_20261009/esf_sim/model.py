"""Public ESF model interface and paper-exact initial condition."""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from . import _model_impl as _impl


CFLViolation = _impl.CFLViolation
ESFModel = _impl.ESFModel
RhsDiagnostics = _impl.RhsDiagnostics


def chapman_profile(y_m: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """Return equation (12) of Li et al. (2021) in SI units."""

    z = (y_m - config.chapman_peak_m) / config.chapman_scale_height_m
    return config.peak_density_m3 * np.exp(1.0 - z - np.exp(-z))


def initial_density(config: SimulationConfig) -> np.ndarray:
    """Construct the Chapman layer with the requested cosine perturbation."""

    background = chapman_profile(config.y_coordinates_m, config)[:, None]
    modulation = 1.0 + config.perturbation_amplitude * np.cos(
        2.0 * np.pi * config.x_coordinates_m / config.domain_width_m
    )
    density = background * modulation[None, :]
    if config.relative_noise:
        rng = np.random.default_rng(config.random_seed)
        density *= 1.0 + config.relative_noise * rng.standard_normal(density.shape)
    return density


def run_simulation(
    config: SimulationConfig,
    *,
    resume: bool = True,
    maximum_steps: int | None = None,
    frame_archive_directory=None,
    frame_interval_s: float = 1.0,
    frame_chunk_size: int = 100,
):
    """Run one scheme using the paper-exact initial condition."""

    _impl.chapman_profile = chapman_profile
    _impl.initial_density = initial_density
    return _impl.run_simulation(
        config,
        resume=resume,
        maximum_steps=maximum_steps,
        frame_archive_directory=frame_archive_directory,
        frame_interval_s=frame_interval_s,
        frame_chunk_size=frame_chunk_size,
    )
