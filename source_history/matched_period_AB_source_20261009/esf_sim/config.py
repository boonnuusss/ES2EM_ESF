"""Configuration for the independent Figure 2 reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
from typing import Any

import numpy as np


VALID_SCHEMES = ("first_order", "van_leer", "pdm7")
CONSERVATIVE_HARMONIC_OPERATOR = "conservative_harmonic"
BEST_BASELINE_POTENTIAL_OPERATOR = "point_centered_equation_2_21_log_coefficients"
MIMETIC_HARMONIC_LOG_RHS_OPERATOR = "mimetic_harmonic_log_rhs"
EXPONENTIAL_CONSERVATIVE_OPERATOR = "exponential_conservative"
DIRICHLET_ZERO_POTENTIAL_BOUNDARY = "dirichlet_zero"
BEST_BASELINE_POTENTIAL_BOUNDARY = "equal_extrapolation"
VALID_POTENTIAL_OPERATORS = (
    CONSERVATIVE_HARMONIC_OPERATOR,
    BEST_BASELINE_POTENTIAL_OPERATOR,
    MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
    EXPONENTIAL_CONSERVATIVE_OPERATOR,
)
VALID_POTENTIAL_BOUNDARIES = (
    DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
    BEST_BASELINE_POTENTIAL_BOUNDARY,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for one ESF run.

    Distances exposed in JSON use kilometres. The numerical core uses SI units.
    """

    scheme: str = "pdm7"
    nx: int = 201
    ny: int = 401
    x_min_km: float = 0.0
    x_max_km: float = 250.0
    y_min_km: float = 250.0
    y_max_km: float = 650.0
    dx_km: float = 1.25
    dy_km: float = 1.0

    peak_density_m3: float = 1.0e12
    chapman_peak_km: float = 438.0
    chapman_scale_km: float = 70.0
    perturbation_amplitude: float = 0.10
    perturbation_wavelength_km: float = 250.0
    gravity_m_s2: float = 9.8
    collision_nu0_s: float = 26.7
    collision_scale_km: float = 81.4

    dt_s: float = 1.0
    cfl_limit: float = 0.25
    t_end_s: float = 3150.0
    pdm_alpha: float = 0.2
    potential_operator: str = CONSERVATIVE_HARMONIC_OPERATOR
    potential_boundary_y: str = DIRICHLET_ZERO_POTENTIAL_BOUNDARY
    potential_rtol: float = 1.0e-8
    potential_maxiter: int = 400
    preconditioner_refresh_steps: int = 25
    preconditioner_iteration_limit: int = 80
    density_floor_m3: float = 1.0

    checkpoint_interval_s: float = 100.0
    diagnostics_interval_steps: int = 1
    output_dir: str = "output"
    reference_mat: str = "data/reference/numDensity.mat"
    relative_noise: float = 0.0
    random_seed: int = 2021

    def __post_init__(self) -> None:
        if self.scheme not in VALID_SCHEMES:
            raise ValueError(f"scheme must be one of {VALID_SCHEMES}, got {self.scheme!r}")
        if self.nx < 8 or self.ny < 8:
            raise ValueError("nx and ny must both be at least 8")
        if self.dt_s <= 0 or self.t_end_s <= 0:
            raise ValueError("dt_s and t_end_s must be positive")
        if not (0 < self.cfl_limit <= 1):
            raise ValueError("cfl_limit must be in (0, 1]")
        if not (0 < self.pdm_alpha <= 1):
            raise ValueError("pdm_alpha must be in (0, 1]")
        if self.potential_operator not in VALID_POTENTIAL_OPERATORS:
            raise ValueError(
                "potential_operator must be one of "
                f"{VALID_POTENTIAL_OPERATORS}, got {self.potential_operator!r}"
            )
        if self.potential_boundary_y not in VALID_POTENTIAL_BOUNDARIES:
            raise ValueError(
                "potential_boundary_y must be one of "
                f"{VALID_POTENTIAL_BOUNDARIES}, got {self.potential_boundary_y!r}"
            )
        if self.relative_noise < 0:
            raise ValueError("relative_noise must be non-negative")

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown configuration keys: {unknown}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **updates: Any) -> "SimulationConfig":
        return replace(self, **updates)

    def validate(self) -> None:
        """Validate that bounds, spacing, and grid counts agree."""

        expected_nx = int(round((self.x_max_km - self.x_min_km) / self.dx_km)) + 1
        expected_ny = int(round((self.y_max_km - self.y_min_km) / self.dy_km)) + 1
        if self.nx != expected_nx or self.ny != expected_ny:
            raise ValueError(
                "grid counts are inconsistent with bounds and spacing: "
                f"expected ({expected_nx}, {expected_ny}), got ({self.nx}, {self.ny})"
            )

    @property
    def dx_m(self) -> float:
        return self.dx_km * 1000.0

    @property
    def dy_m(self) -> float:
        return self.dy_km * 1000.0

    @property
    def x_coordinates_m(self) -> np.ndarray:
        return self.x_min_km * 1000.0 + np.arange(self.nx) * self.dx_m

    @property
    def y_coordinates_m(self) -> np.ndarray:
        return self.y_min_km * 1000.0 + np.arange(self.ny) * self.dy_m

    @property
    def domain_width_m(self) -> float:
        return self.perturbation_wavelength_km * 1000.0

    @property
    def chapman_peak_m(self) -> float:
        return self.chapman_peak_km * 1000.0

    @property
    def chapman_scale_height_m(self) -> float:
        return self.chapman_scale_km * 1000.0

    @property
    def collision_frequency_scale_height_m(self) -> float:
        return self.collision_scale_km * 1000.0

    @property
    def output_directory(self) -> Path:
        return Path(self.output_dir)

    @property
    def checkpoint_interval_steps(self) -> int:
        return max(1, int(round(self.checkpoint_interval_s / self.dt_s)))

    @property
    def positivity_tolerance_m3(self) -> float:
        return 1.0e-12 * self.peak_density_m3
