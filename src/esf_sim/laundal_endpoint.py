"""Dynamic ionospheric endpoint in the instantaneous-Alfven limit.

This is the local, height-integrated ion-inertia part of Laundal et al.
(2024), equations (2), (3), and (5).  The external transverse electric field
is applied instantaneously, corresponding to the paper's deliberately
simplified ``V_A -> infinity`` magnetospheric closure.  The class is a load,
not an Alfven propagator; it is intended to terminate ``alfven_1d`` later.

With ``B = B0 * zhat`` and zero neutral wind,

    E = -v_e x B,
    dv_i/dt = (e/m_i) (E + v_i x B) - nu_in v_i.

The exact constant-field update below avoids resolving the gyroperiod with an
explicit Euler step.  Its steady limit recovers the Pedersen/Hall conductance
relations given in Appendix A of Laundal et al.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ELEMENTARY_CHARGE_C = 1.602176634e-19


def _vector2(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape (2,)")
    return array.copy()


@dataclass(frozen=True)
class DynamicEndpointState:
    ion_velocity_m_s: np.ndarray

    def validated(self) -> "DynamicEndpointState":
        return DynamicEndpointState(_vector2("ion_velocity_m_s", self.ion_velocity_m_s))


@dataclass(frozen=True)
class DynamicEndpointParameters:
    """Height-integrated single-ion parameters for the dynamic load."""

    surface_density_m2: float
    ion_mass_kg: float
    collision_frequency_s: float
    background_field_t: float

    def __post_init__(self) -> None:
        positive = {
            "surface_density_m2": self.surface_density_m2,
            "ion_mass_kg": self.ion_mass_kg,
            "collision_frequency_s": self.collision_frequency_s,
            "background_field_t": self.background_field_t,
        }
        for name, raw in positive.items():
            value = float(raw)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)

    @property
    def gyrofrequency_s(self) -> float:
        return ELEMENTARY_CHARGE_C * self.background_field_t / self.ion_mass_kg

    @property
    def ion_magnetization(self) -> float:
        return self.gyrofrequency_s / self.collision_frequency_s

    @property
    def pedersen_conductance_s(self) -> float:
        k = self.ion_magnetization
        return (
            self.surface_density_m2
            * ELEMENTARY_CHARGE_C
            / self.background_field_t
            * k
            / (1.0 + k * k)
        )

    @property
    def hall_conductance_s(self) -> float:
        k = self.ion_magnetization
        return (
            self.surface_density_m2
            * ELEMENTARY_CHARGE_C
            / self.background_field_t
            / (1.0 + k * k)
        )

    @classmethod
    def from_conductances(
        cls,
        *,
        pedersen_conductance_s: float,
        hall_conductance_s: float,
        ion_mass_kg: float,
        background_field_t: float,
    ) -> "DynamicEndpointParameters":
        sigma_p = float(pedersen_conductance_s)
        sigma_h = float(hall_conductance_s)
        if not np.isfinite(sigma_p) or sigma_p <= 0.0:
            raise ValueError("pedersen_conductance_s must be positive and finite")
        if not np.isfinite(sigma_h) or sigma_h <= 0.0:
            raise ValueError("hall_conductance_s must be positive and finite")
        mass = float(ion_mass_kg)
        field = float(background_field_t)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("ion_mass_kg must be positive and finite")
        if not np.isfinite(field) or field <= 0.0:
            raise ValueError("background_field_t must be positive and finite")
        magnetization = sigma_p / sigma_h
        gyro = ELEMENTARY_CHARGE_C * field / mass
        collision = gyro / magnetization
        density = (
            sigma_p
            * field
            / ELEMENTARY_CHARGE_C
            * (1.0 + magnetization**2)
            / magnetization
        )
        return cls(density, mass, collision, field)


class InstantaneousLaundalEndpoint:
    """Exact dynamic ion response to a piecewise-constant transverse E field."""

    def __init__(self, parameters: DynamicEndpointParameters):
        self.parameters = parameters

    @staticmethod
    def zero_state() -> DynamicEndpointState:
        return DynamicEndpointState(np.zeros(2, dtype=np.float64))

    def electron_velocity(self, electric_v_m: np.ndarray) -> np.ndarray:
        """Return ``v_e`` from ``E = -v_e x B0``."""

        ex, ey = _vector2("electric_v_m", electric_v_m)
        field = self.parameters.background_field_t
        return np.array([ey / field, -ex / field])

    def rhs(
        self, state: DynamicEndpointState, electric_v_m: np.ndarray
    ) -> DynamicEndpointState:
        velocity = state.validated().ion_velocity_m_s
        ex, ey = _vector2("electric_v_m", electric_v_m)
        omega = self.parameters.gyrofrequency_s
        nu = self.parameters.collision_frequency_s
        acceleration = np.array(
            [
                ELEMENTARY_CHARGE_C / self.parameters.ion_mass_kg * ex
                + omega * velocity[1]
                - nu * velocity[0],
                ELEMENTARY_CHARGE_C / self.parameters.ion_mass_kg * ey
                - omega * velocity[0]
                - nu * velocity[1],
            ]
        )
        return DynamicEndpointState(acceleration)

    def steady_velocity(self, electric_v_m: np.ndarray) -> np.ndarray:
        ex, ey = _vector2("electric_v_m", electric_v_m)
        omega = self.parameters.gyrofrequency_s
        nu = self.parameters.collision_frequency_s
        charge_to_mass = ELEMENTARY_CHARGE_C / self.parameters.ion_mass_kg
        denominator = nu * nu + omega * omega
        return charge_to_mass / denominator * np.array(
            [nu * ex + omega * ey, -omega * ex + nu * ey]
        )

    def current_a_m(
        self, state: DynamicEndpointState, electric_v_m: np.ndarray
    ) -> np.ndarray:
        ion = state.validated().ion_velocity_m_s
        electron = self.electron_velocity(electric_v_m)
        return (
            self.parameters.surface_density_m2
            * ELEMENTARY_CHARGE_C
            * (ion - electron)
        )

    def steady_current_a_m(self, electric_v_m: np.ndarray) -> np.ndarray:
        state = DynamicEndpointState(self.steady_velocity(electric_v_m))
        return self.current_a_m(state, electric_v_m)

    def step(
        self,
        state: DynamicEndpointState,
        electric_v_m: np.ndarray,
        dt_s: float,
    ) -> DynamicEndpointState:
        """Advance exactly while the applied electric field is constant."""

        velocity = state.validated().ion_velocity_m_s
        electric = _vector2("electric_v_m", electric_v_m)
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        steady = self.steady_velocity(electric)
        omega = self.parameters.gyrofrequency_s
        nu = self.parameters.collision_frequency_s
        angle = omega * dt
        rotation = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        updated = steady + np.exp(-nu * dt) * (rotation @ (velocity - steady))
        return DynamicEndpointState(updated)

    def kinetic_energy_j_m2(self, state: DynamicEndpointState) -> float:
        velocity = state.validated().ion_velocity_m_s
        return float(
            0.5
            * self.parameters.surface_density_m2
            * self.parameters.ion_mass_kg
            * np.dot(velocity, velocity)
        )

    def energy_rate_terms_w_m2(
        self, state: DynamicEndpointState, electric_v_m: np.ndarray
    ) -> tuple[float, float]:
        """Return electromagnetic input ``j.E`` and ion-neutral dissipation."""

        velocity = state.validated().ion_velocity_m_s
        electric = _vector2("electric_v_m", electric_v_m)
        current = self.current_a_m(state, electric)
        input_power = float(np.dot(current, electric))
        collision_loss = float(
            self.parameters.surface_density_m2
            * self.parameters.ion_mass_kg
            * self.parameters.collision_frequency_s
            * np.dot(velocity, velocity)
        )
        return input_power, collision_loss


@dataclass(frozen=True)
class FieldLineDisplacementState:
    """Relative transverse displacement between magnetosphere and ionosphere."""

    relative_displacement_m: float

    def __post_init__(self) -> None:
        value = float(self.relative_displacement_m)
        if not np.isfinite(value):
            raise ValueError("relative_displacement_m must be finite")
        object.__setattr__(self, "relative_displacement_m", value)


class InstantaneousFieldLineClosure:
    """Laundal straight-field closure obtained when ``tau_A -> 0``.

    A finite field line of length ``H`` becomes linear in field-aligned
    coordinate when its Alfven transit time is negligible.  If ``q`` is the
    magnetosphere-minus-ionosphere footpoint displacement, the small-angle
    magnetic perturbation is ``B0*q/H``.  ``magnetic_perturbation_t`` also
    offers Laundal's bounded finite-angle geometry.
    """

    def __init__(self, *, length_m: float, background_field_t: float):
        self.length_m = float(length_m)
        self.background_field_t = float(background_field_t)
        if not np.isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError("length_m must be positive and finite")
        if not np.isfinite(self.background_field_t) or self.background_field_t <= 0.0:
            raise ValueError("background_field_t must be positive and finite")

    @staticmethod
    def displacement_rhs_m_s(
        magnetospheric_velocity_m_s: float, electron_velocity_m_s: float
    ) -> float:
        values = np.asarray(
            [magnetospheric_velocity_m_s, electron_velocity_m_s], dtype=np.float64
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("endpoint velocities must be finite")
        return float(values[0] - values[1])

    def step(
        self,
        state: FieldLineDisplacementState,
        *,
        magnetospheric_velocity_m_s: float,
        electron_velocity_m_s: float,
        dt_s: float,
    ) -> FieldLineDisplacementState:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        return FieldLineDisplacementState(
            state.relative_displacement_m
            + dt
            * self.displacement_rhs_m_s(
                magnetospheric_velocity_m_s, electron_velocity_m_s
            )
        )

    def magnetic_perturbation_t(
        self, state: FieldLineDisplacementState, *, finite_angle: bool = False
    ) -> float:
        slope = state.relative_displacement_m / self.length_m
        if finite_angle:
            slope /= np.sqrt(1.0 + slope * slope)
        return float(self.background_field_t * slope)

    def displacement_profile_m(
        self,
        s_m: np.ndarray,
        *,
        ionospheric_displacement_m: float,
        magnetospheric_displacement_m: float,
    ) -> np.ndarray:
        """Return the exact linear field-line shape in the zero-transit limit."""

        s = np.asarray(s_m, dtype=np.float64)
        if np.any(~np.isfinite(s)) or np.any(s < 0.0) or np.any(s > self.length_m):
            raise ValueError("s_m must be finite and lie inside the field line")
        fraction = s / self.length_m
        return (
            float(ionospheric_displacement_m) * (1.0 - fraction)
            + float(magnetospheric_displacement_m) * fraction
        )


def finite_alfven_to_instantaneous_slope_ratio(
    *, angular_frequency_s: float, length_m: float, alfven_speed_m_s: float
) -> float:
    """Exact harmonic finite-line slope divided by the Laundal limit.

    For a line with one fixed end and harmonic displacement at the other,
    ``xi(s)=Q sin(k s)/sin(k H)``.  The endpoint slope divided by ``Q/H`` is
    ``(k H)/sin(k H)`` and approaches one quadratically as
    ``tau_A / period -> 0``.  This is the relevant finite-length limit; simply
    sending the semi-infinite Alfven conductance to zero is a different
    boundary problem.
    """

    omega = float(angular_frequency_s)
    length = float(length_m)
    speed = float(alfven_speed_m_s)
    if not np.isfinite(omega) or omega < 0.0:
        raise ValueError("angular_frequency_s must be non-negative and finite")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length_m must be positive and finite")
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("alfven_speed_m_s must be positive and finite")
    phase = omega * length / speed
    if phase == 0.0:
        return 1.0
    if abs(np.sin(phase)) < 1.0e-14:
        raise ValueError("harmonic response is singular at a field-line resonance")
    return float(phase / np.sin(phase))


__all__ = [
    "DynamicEndpointParameters",
    "DynamicEndpointState",
    "ELEMENTARY_CHARGE_C",
    "FieldLineDisplacementState",
    "InstantaneousFieldLineClosure",
    "InstantaneousLaundalEndpoint",
    "finite_alfven_to_instantaneous_slope_ratio",
]
