"""Dipole/Chapman mapping for equatorial flux-tube column coefficients.

The Li ESF grid uses altitude at the magnetic equator.  A grid row at apex
radius ``r_a = R_E + h_a`` labels the dipole field line

``r(lambda) = r_a cos(lambda)**2``.

For a dipole, ``B/B_a = sqrt(1+3 sin(lambda)**2)/cos(lambda)**6`` and
``ds/dlambda = r_a cos(lambda) sqrt(1+3 sin(lambda)**2)``.  Magnetic-flux
conservation therefore reduces the volume weight to the particularly simple

``(B_a/B) ds = r_a cos(lambda)**7 dlambda``.

Two different full-conjugate columns are required by the integrated Li
closure:

``N_col = integral n_C (B_a/B) ds``

and

``Cnu_col = integral n_C nu_in (B_a/B) ds``.

It is convenient for a time-dependent equatorial density to store the two
positive length weights

``H_g = N_col / n_C(h_a)`` and
``H_nu = Cnu_col / (n_C(h_a) nu_in(h_a))``.

They are not interchangeable because ``nu_in`` changes by orders of
magnitude along a field line.  The column-integrated equation is

``div(H_nu n nu grad psi) = -g d_x(H_g n) + B_a J_surface/m_i``.

``n_C`` is exactly the Chapman background used by :mod:`esf_sim._model_impl`,
not a second empirical profile.  The lower integration altitude is explicit;
100 km is the default because the prescribed Chapman density is already
negligible there.  Fixed Gauss--Legendre quadrature makes the mapping
deterministic and inexpensive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
DEFAULT_EARTH_RADIUS_M = 6_371_000.0
DEFAULT_COLUMN_LOWER_ALTITUDE_M = 100_000.0
DEFAULT_QUADRATURE_ORDER = 128


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _chapman_log_shape(
    altitude_m: FloatArray,
    *,
    peak_altitude_m: float,
    scale_height_m: float,
) -> FloatArray:
    z = (altitude_m - peak_altitude_m) / scale_height_m
    return 1.0 - z - np.exp(-z)


@dataclass(frozen=True)
class DipoleChapmanColumnProfile:
    """Auditable altitude grid and its two full-conjugate column weights."""

    apex_altitude_m: FloatArray
    gravity_column_weight_m: FloatArray
    collision_column_weight_m: FloatArray
    background_density_column_m2: FloatArray
    background_collision_column_m2_s: FloatArray
    collision_moment_m_s: FloatArray
    effective_collision_frequency_s: FloatArray
    earth_radius_m: float
    lower_altitude_m: float
    chapman_peak_altitude_m: float
    chapman_scale_height_m: float
    chapman_peak_density_m3: float
    collision_nu0_s: float
    collision_frequency_scale_height_m: float
    quadrature_order: int

    def summary(self) -> dict[str, float | int | str]:
        gravity = self.gravity_column_weight_m
        collision = self.collision_column_weight_m
        effective_collision = self.effective_collision_frequency_s
        return {
            "model": "dipole_chapman_flux_tube_columns",
            "earth_radius_km": self.earth_radius_m / 1000.0,
            "lower_altitude_km": self.lower_altitude_m / 1000.0,
            "chapman_peak_altitude_km": (
                self.chapman_peak_altitude_m / 1000.0
            ),
            "chapman_scale_height_km": (
                self.chapman_scale_height_m / 1000.0
            ),
            "chapman_peak_density_m3": self.chapman_peak_density_m3,
            "collision_nu0_s": self.collision_nu0_s,
            "collision_frequency_scale_height_km": (
                self.collision_frequency_scale_height_m / 1000.0
            ),
            "quadrature_order": int(self.quadrature_order),
            "minimum_gravity_column_weight_km": float(
                np.min(gravity) / 1000.0
            ),
            "maximum_gravity_column_weight_km": float(
                np.max(gravity) / 1000.0
            ),
            "first_gravity_column_weight_km": float(gravity[0] / 1000.0),
            "last_gravity_column_weight_km": float(gravity[-1] / 1000.0),
            "minimum_collision_column_weight_km": float(
                np.min(collision) / 1000.0
            ),
            "maximum_collision_column_weight_km": float(
                np.max(collision) / 1000.0
            ),
            "first_collision_column_weight_km": float(
                collision[0] / 1000.0
            ),
            "last_collision_column_weight_km": float(
                collision[-1] / 1000.0
            ),
            "minimum_effective_collision_frequency_s": float(
                np.min(effective_collision)
            ),
            "maximum_effective_collision_frequency_s": float(
                np.max(effective_collision)
            ),
        }


def dipole_chapman_column_profile(
    apex_altitude_m: ArrayLike,
    *,
    chapman_peak_altitude_m: float,
    chapman_scale_height_m: float,
    chapman_peak_density_m3: float = 1.0e12,
    collision_nu0_s: float = 26.7,
    collision_frequency_scale_height_m: float = 81_400.0,
    earth_radius_m: float = DEFAULT_EARTH_RADIUS_M,
    lower_altitude_m: float = DEFAULT_COLUMN_LOWER_ALTITUDE_M,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
) -> DipoleChapmanColumnProfile:
    """Return separate gravity and collision full-conjugate column weights.

    Every apex must lie above ``lower_altitude_m``.  The returned arrays are
    immutable copies so a checkpointed trajectory cannot be changed through
    an aliased configuration array.
    """

    altitude = np.asarray(apex_altitude_m, dtype=np.float64)
    if altitude.ndim != 1 or altitude.size == 0:
        raise ValueError("apex_altitude_m must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(altitude)):
        raise ValueError("apex_altitude_m must be finite")
    earth_radius = _positive_finite("earth_radius_m", earth_radius_m)
    peak = float(chapman_peak_altitude_m)
    if not np.isfinite(peak):
        raise ValueError("chapman_peak_altitude_m must be finite")
    scale = _positive_finite("chapman_scale_height_m", chapman_scale_height_m)
    peak_density = _positive_finite(
        "chapman_peak_density_m3", chapman_peak_density_m3
    )
    collision_nu0 = _positive_finite("collision_nu0_s", collision_nu0_s)
    collision_scale = _positive_finite(
        "collision_frequency_scale_height_m",
        collision_frequency_scale_height_m,
    )
    lower = float(lower_altitude_m)
    if not np.isfinite(lower) or lower < 0.0:
        raise ValueError("lower_altitude_m must be non-negative and finite")
    if np.any(altitude <= lower):
        raise ValueError("every apex altitude must exceed lower_altitude_m")
    order = int(quadrature_order)
    if isinstance(quadrature_order, bool) or order != quadrature_order or order < 16:
        raise ValueError("quadrature_order must be an integer of at least 16")

    nodes, weights = np.polynomial.legendre.leggauss(order)
    apex_radius = earth_radius + altitude
    lower_radius = earth_radius + lower
    lambda_max = np.arccos(np.sqrt(lower_radius / apex_radius))
    magnetic_latitude = 0.5 * lambda_max[:, None] * (nodes[None, :] + 1.0)
    cosine = np.cos(magnetic_latitude)
    field_line_altitude = apex_radius[:, None] * cosine**2 - earth_radius
    log_density = _chapman_log_shape(
        field_line_altitude,
        peak_altitude_m=peak,
        scale_height_m=scale,
    )
    log_apex_density = _chapman_log_shape(
        altitude,
        peak_altitude_m=peak,
        scale_height_m=scale,
    )
    density_ratio = np.exp(log_density - log_apex_density[:, None])
    half_gravity_integral = (
        apex_radius
        * 0.5
        * lambda_max
        * np.sum(
            weights[None, :] * cosine**7 * density_ratio,
            axis=1,
        )
    )
    collision_ratio = density_ratio * np.exp(
        -(field_line_altitude - altitude[:, None]) / collision_scale
    )
    half_collision_integral = (
        apex_radius
        * 0.5
        * lambda_max
        * np.sum(
            weights[None, :] * cosine**7 * collision_ratio,
            axis=1,
        )
    )
    gravity_weight = 2.0 * half_gravity_integral
    collision_weight = 2.0 * half_collision_integral
    for name, values in (
        ("gravity column", gravity_weight),
        ("collision column", collision_weight),
    ):
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise RuntimeError(
                f"dipole/Chapman mapping produced invalid {name} weights"
            )

    apex_density = peak_density * np.exp(log_apex_density)
    apex_collision = collision_nu0 * np.exp(-altitude / collision_scale)
    density_column = apex_density * gravity_weight
    collision_column = apex_density * apex_collision * collision_weight
    collision_moment = apex_collision * collision_weight
    effective_collision = collision_moment / gravity_weight

    altitude = np.array(altitude, copy=True)
    gravity_weight = np.array(gravity_weight, copy=True)
    collision_weight = np.array(collision_weight, copy=True)
    density_column = np.array(density_column, copy=True)
    collision_column = np.array(collision_column, copy=True)
    for values in (
        altitude,
        gravity_weight,
        collision_weight,
        density_column,
        collision_column,
        collision_moment,
        effective_collision,
    ):
        values.setflags(write=False)
    return DipoleChapmanColumnProfile(
        apex_altitude_m=altitude,
        gravity_column_weight_m=gravity_weight,
        collision_column_weight_m=collision_weight,
        background_density_column_m2=density_column,
        background_collision_column_m2_s=collision_column,
        collision_moment_m_s=collision_moment,
        effective_collision_frequency_s=effective_collision,
        earth_radius_m=earth_radius,
        lower_altitude_m=lower,
        chapman_peak_altitude_m=peak,
        chapman_scale_height_m=scale,
        chapman_peak_density_m3=peak_density,
        collision_nu0_s=collision_nu0,
        collision_frequency_scale_height_m=collision_scale,
        quadrature_order=order,
    )


__all__ = [
    "DEFAULT_COLUMN_LOWER_ALTITUDE_M",
    "DEFAULT_EARTH_RADIUS_M",
    "DEFAULT_QUADRATURE_ORDER",
    "DipoleChapmanColumnProfile",
    "dipole_chapman_column_profile",
]
