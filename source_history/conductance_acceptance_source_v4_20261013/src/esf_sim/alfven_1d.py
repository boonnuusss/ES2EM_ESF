"""One-dimensional reduced shear-Alfven transmission-line solver.

The module is deliberately independent of the historical Li et al. ESF
solver.  It provides the field-aligned building block needed by a later
``2-D perpendicular + 1-D parallel`` coupling while keeping the existing
electrostatic path available as a regression oracle.

The ideal, straight-field equations advanced here are

    rho * du/dt = B0 / mu0 * db/ds
             db/dt = B0       * du/ds,

where ``u`` is one transverse plasma-velocity component and ``b`` is the
corresponding transverse magnetic perturbation.  For a right-going wave
``b = -sqrt(mu0*rho) * u``.  A prescribed density profile may vary along
the field line; the background magnetic field is constant in this first
reduced model so that the discrete physical-energy budget is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.sparse import bmat, csr_matrix, diags, eye
from scipy.sparse.linalg import factorized


MU0 = 4.0e-7 * np.pi


def _as_finite_vector(name: str, values: np.ndarray, size: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or (size is not None and array.size != size):
        expected = "one-dimensional" if size is None else f"shape ({size},)"
        raise ValueError(f"{name} must have {expected}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.copy()


@dataclass(frozen=True)
class FieldLineGrid:
    """Uniform cell-centred field-line grid and fixed background profiles."""

    s_m: np.ndarray
    background_field_t: np.ndarray
    mass_density_kg_m3: np.ndarray

    def __post_init__(self) -> None:
        s = _as_finite_vector("s_m", self.s_m)
        if s.size < 4:
            raise ValueError("field-line grid requires at least four cells")
        spacing = np.diff(s)
        if np.any(spacing <= 0.0) or not np.allclose(
            spacing, spacing[0], rtol=1.0e-12, atol=0.0
        ):
            raise ValueError("s_m must be strictly increasing and uniformly spaced")
        field = _as_finite_vector("background_field_t", self.background_field_t, s.size)
        density = _as_finite_vector(
            "mass_density_kg_m3", self.mass_density_kg_m3, s.size
        )
        if np.any(field <= 0.0):
            raise ValueError(
                "background magnetic field must be a positive magnitude; "
                "the +s coordinate carries its direction"
            )
        if not np.allclose(field, field[0], rtol=1.0e-13, atol=0.0):
            raise ValueError(
                "the first reduced solver requires constant background_field_t; "
                "vary V_A through mass_density_kg_m3"
            )
        if np.any(density <= 0.0):
            raise ValueError("mass density must be positive")
        object.__setattr__(self, "s_m", s)
        object.__setattr__(self, "background_field_t", field)
        object.__setattr__(self, "mass_density_kg_m3", density)

    @classmethod
    def uniform(
        cls,
        *,
        length_m: float,
        cells: int,
        background_field_t: float,
        mass_density_kg_m3: float,
    ) -> "FieldLineGrid":
        """Construct periodic-style cell centres on ``[0, length)``."""

        if not np.isfinite(length_m) or length_m <= 0.0:
            raise ValueError("length_m must be positive and finite")
        if cells < 4:
            raise ValueError("cells must be at least four")
        spacing = float(length_m) / int(cells)
        s = (np.arange(cells, dtype=np.float64) + 0.5) * spacing
        return cls(
            s,
            np.full(cells, float(background_field_t)),
            np.full(cells, float(mass_density_kg_m3)),
        )

    @property
    def cells(self) -> int:
        return self.s_m.size

    @property
    def spacing_m(self) -> float:
        return float(self.s_m[1] - self.s_m[0])

    @property
    def length_m(self) -> float:
        return self.cells * self.spacing_m

    @cached_property
    def alfven_speed_m_s(self) -> np.ndarray:
        return np.abs(self.background_field_t) / np.sqrt(
            MU0 * self.mass_density_kg_m3
        )

    @cached_property
    def magnetic_impedance_t_s_m(self) -> np.ndarray:
        """Return ``|b/u| = sqrt(mu0*rho)`` for a travelling wave."""

        return np.sqrt(MU0 * self.mass_density_kg_m3)

    @cached_property
    def alfven_conductance_s(self) -> np.ndarray:
        """Return the local Alfven conductance ``1/(mu0*V_A)``."""

        return 1.0 / (MU0 * self.alfven_speed_m_s)


@dataclass(frozen=True)
class AlfvenState:
    """Cell-centred transverse velocity and magnetic perturbation."""

    velocity_m_s: np.ndarray
    magnetic_t: np.ndarray

    def validated(self, cells: int) -> "AlfvenState":
        return AlfvenState(
            _as_finite_vector("velocity_m_s", self.velocity_m_s, cells),
            _as_finite_vector("magnetic_t", self.magnetic_t, cells),
        )

    def linear_combination(
        self, other: "AlfvenState", self_weight: float, other_weight: float
    ) -> "AlfvenState":
        return AlfvenState(
            self_weight * self.velocity_m_s + other_weight * other.velocity_m_s,
            self_weight * self.magnetic_t + other_weight * other.magnetic_t,
        )


@dataclass(frozen=True)
class ConductanceLoad:
    """Pedersen-conductance load for an Alfven characteristic boundary.

    The reflected transverse electric-field/velocity amplitude is

    ``R = (Sigma_A - Sigma_P) / (Sigma_A + Sigma_P)``.

    Thus an open end (``Sigma_P=0``) has ``R=+1``, a matched load has
    ``R=0``, and the highly conducting limit has ``R=-1``.
    """

    pedersen_conductance_s: float

    def __post_init__(self) -> None:
        value = float(self.pedersen_conductance_s)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Pedersen conductance must be non-negative and finite")
        object.__setattr__(self, "pedersen_conductance_s", value)

    def reflection_coefficient(self, alfven_conductance_s: float) -> float:
        sigma_a = float(alfven_conductance_s)
        if not np.isfinite(sigma_a) or sigma_a <= 0.0:
            raise ValueError("Alfven conductance must be positive and finite")
        return (sigma_a - self.pedersen_conductance_s) / (
            sigma_a + self.pedersen_conductance_s
        )


@dataclass(frozen=True)
class CharacteristicBoundary:
    """Left and right reflection coefficients for the velocity characteristic."""

    left_reflection: float = 0.0
    right_reflection: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("left_reflection", self.left_reflection),
            ("right_reflection", self.right_reflection),
        ):
            if not np.isfinite(value) or abs(value) > 1.0:
                raise ValueError(f"{name} must lie in [-1, 1]")

    @classmethod
    def from_loads(
        cls,
        grid: FieldLineGrid,
        *,
        left: ConductanceLoad,
        right: ConductanceLoad,
    ) -> "CharacteristicBoundary":
        return cls(
            left.reflection_coefficient(grid.alfven_conductance_s[0]),
            right.reflection_coefficient(grid.alfven_conductance_s[-1]),
        )


@dataclass(frozen=True)
class AlfvenEnergy:
    kinetic_j_m2: float
    magnetic_j_m2: float

    @property
    def total_j_m2(self) -> float:
        return self.kinetic_j_m2 + self.magnetic_j_m2


def characteristic_amplitudes(
    state: AlfvenState, grid: FieldLineGrid
) -> tuple[np.ndarray, np.ndarray]:
    """Return right- and left-going velocity characteristics ``z+``/``z-``."""

    state = state.validated(grid.cells)
    scaled_b = state.magnetic_t / grid.magnetic_impedance_t_s_m
    right = state.velocity_m_s - scaled_b
    left = state.velocity_m_s + scaled_b
    return right, left


def state_from_characteristics(
    right_m_s: np.ndarray, left_m_s: np.ndarray, grid: FieldLineGrid
) -> AlfvenState:
    right = _as_finite_vector("right_m_s", right_m_s, grid.cells)
    left = _as_finite_vector("left_m_s", left_m_s, grid.cells)
    velocity = 0.5 * (right + left)
    magnetic = 0.5 * grid.magnetic_impedance_t_s_m * (left - right)
    return AlfvenState(velocity, magnetic)


def alfven_energy(state: AlfvenState, grid: FieldLineGrid) -> AlfvenEnergy:
    """Return transverse kinetic and magnetic energy per unit tube area."""

    state = state.validated(grid.cells)
    ds = grid.spacing_m
    kinetic = 0.5 * ds * np.sum(
        grid.mass_density_kg_m3 * state.velocity_m_s**2
    )
    magnetic = 0.5 * ds / MU0 * np.sum(state.magnetic_t**2)
    return AlfvenEnergy(float(kinetic), float(magnetic))


def poynting_flux_w_m2(state: AlfvenState, grid: FieldLineGrid) -> np.ndarray:
    """Return signed field-aligned Poynting flux ``-B0*u*b/mu0``."""

    state = state.validated(grid.cells)
    return -grid.background_field_t * state.velocity_m_s * state.magnetic_t / MU0


def _periodic_central_derivative(cells: int, spacing_m: float) -> csr_matrix:
    rows = np.arange(cells, dtype=np.int64)
    columns_forward = (rows + 1) % cells
    columns_backward = (rows - 1) % cells
    data = np.concatenate(
        [
            np.full(cells, 0.5 / spacing_m),
            np.full(cells, -0.5 / spacing_m),
        ]
    )
    return csr_matrix(
        (
            data,
            (np.concatenate([rows, rows]), np.concatenate([columns_forward, columns_backward])),
        ),
        shape=(cells, cells),
    )


class PeriodicMidpointAlfvenSolver:
    """Energy-compatible centred spatial operator and implicit-midpoint time step."""

    def __init__(self, grid: FieldLineGrid):
        self.grid = grid
        derivative = _periodic_central_derivative(grid.cells, grid.spacing_m)
        zero = csr_matrix((grid.cells, grid.cells), dtype=np.float64)
        u_from_b = diags(
            grid.background_field_t / (MU0 * grid.mass_density_kg_m3)
        ) @ derivative
        b_from_u = diags(grid.background_field_t) @ derivative
        self.operator = bmat(
            [[zero, u_from_b], [b_from_u, zero]], format="csr"
        )
        self._factorizations: dict[float, object] = {}

    def step(self, state: AlfvenState, dt_s: float) -> AlfvenState:
        state = state.validated(self.grid.cells)
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        identity = eye(2 * self.grid.cells, format="csc")
        key = dt
        solve = self._factorizations.get(key)
        if solve is None:
            solve = factorized(identity - 0.5 * dt * self.operator)
            self._factorizations[key] = solve
        vector = np.concatenate([state.velocity_m_s, state.magnetic_t])
        updated = solve((identity + 0.5 * dt * self.operator) @ vector)
        return AlfvenState(
            np.asarray(updated[: self.grid.cells]),
            np.asarray(updated[self.grid.cells :]),
        )

    def advance(
        self, state: AlfvenState, *, duration_s: float, dt_s: float
    ) -> AlfvenState:
        duration = float(duration_s)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_s must be non-negative and finite")
        result = state.validated(self.grid.cells)
        time_s = 0.0
        while time_s < duration - 1.0e-15:
            step = min(float(dt_s), duration - time_s)
            result = self.step(result, step)
            time_s += step
        return result


def _interface_star(
    u_left: np.ndarray,
    b_left: np.ndarray,
    impedance_left: np.ndarray,
    u_right: np.ndarray,
    b_right: np.ndarray,
    impedance_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact linear-Alfven interface state for piecewise constant media."""

    right_from_left = u_left - b_left / impedance_left
    left_from_right = u_right + b_right / impedance_right
    denominator = impedance_left + impedance_right
    u_star = (
        impedance_left * right_from_left
        + impedance_right * left_from_right
    ) / denominator
    b_star = (
        impedance_left
        * impedance_right
        * (left_from_right - right_from_left)
        / denominator
    )
    return u_star, b_star


class UpwindAlfvenSolver:
    """Characteristic upwind solver for loads and non-uniform Alfven speed."""

    def __init__(
        self,
        grid: FieldLineGrid,
        boundary: CharacteristicBoundary | None = None,
        *,
        periodic: bool = False,
        spatial_order: int = 2,
    ):
        if periodic and boundary is not None:
            raise ValueError("periodic solver cannot also use characteristic loads")
        self.grid = grid
        self.boundary = boundary or CharacteristicBoundary()
        self.periodic = bool(periodic)
        if spatial_order not in (1, 2):
            raise ValueError("spatial_order must be one or two")
        self.spatial_order = int(spatial_order)

    @property
    def maximum_stable_dt_s(self) -> float:
        return self.grid.spacing_m / float(np.max(self.grid.alfven_speed_m_s))

    @staticmethod
    def _minmod(*values: np.ndarray) -> np.ndarray:
        stacked = np.stack(values)
        same_positive = np.all(stacked > 0.0, axis=0)
        same_negative = np.all(stacked < 0.0, axis=0)
        magnitude = np.min(np.abs(stacked), axis=0)
        return np.where(
            same_positive,
            magnitude,
            np.where(same_negative, -magnitude, 0.0),
        )

    def _slopes(self, values: np.ndarray) -> np.ndarray:
        """Monotonized-central slopes in cell-index units."""

        if self.spatial_order == 1:
            return np.zeros_like(values)
        if self.periodic:
            backward = values - np.roll(values, 1)
            forward = np.roll(values, -1) - values
            return self._minmod(
                2.0 * backward,
                0.5 * (backward + forward),
                2.0 * forward,
            )
        slopes = np.zeros_like(values)
        backward = values[1:-1] - values[:-2]
        forward = values[2:] - values[1:-1]
        slopes[1:-1] = self._minmod(
            2.0 * backward,
            0.5 * (backward + forward),
            2.0 * forward,
        )
        return slopes

    def _face_states(self, state: AlfvenState) -> tuple[np.ndarray, np.ndarray]:
        state = state.validated(self.grid.cells)
        n = self.grid.cells
        impedance = self.grid.magnetic_impedance_t_s_m
        u_slope = self._slopes(state.velocity_m_s)
        b_slope = self._slopes(state.magnetic_t)
        u_left_edge = state.velocity_m_s - 0.5 * u_slope
        u_right_edge = state.velocity_m_s + 0.5 * u_slope
        b_left_edge = state.magnetic_t - 0.5 * b_slope
        b_right_edge = state.magnetic_t + 0.5 * b_slope
        u_face = np.empty(n + 1, dtype=np.float64)
        b_face = np.empty(n + 1, dtype=np.float64)
        u_face[1:n], b_face[1:n] = _interface_star(
            u_right_edge[:-1],
            b_right_edge[:-1],
            impedance[:-1],
            u_left_edge[1:],
            b_left_edge[1:],
            impedance[1:],
        )
        if self.periodic:
            u_edge, b_edge = _interface_star(
                u_right_edge[-1:],
                b_right_edge[-1:],
                impedance[-1:],
                u_left_edge[:1],
                b_left_edge[:1],
                impedance[:1],
            )
            u_face[0] = u_face[n] = u_edge[0]
            b_face[0] = b_face[n] = b_edge[0]
            return u_face, b_face

        # At the left edge z- leaves the domain and z+ enters it.  At the
        # right edge z+ leaves and z- enters.  The load supplies only the
        # incoming characteristic, avoiding over-specification of E and b.
        z_left_out = u_left_edge[0] + b_left_edge[0] / impedance[0]
        z_left_in = self.boundary.left_reflection * z_left_out
        u_face[0] = 0.5 * (z_left_in + z_left_out)
        b_face[0] = 0.5 * impedance[0] * (z_left_out - z_left_in)

        z_right_out = u_right_edge[-1] - b_right_edge[-1] / impedance[-1]
        z_right_in = self.boundary.right_reflection * z_right_out
        u_face[n] = 0.5 * (z_right_out + z_right_in)
        b_face[n] = 0.5 * impedance[-1] * (z_right_in - z_right_out)
        return u_face, b_face

    def rhs(self, state: AlfvenState) -> AlfvenState:
        state = state.validated(self.grid.cells)
        u_face, b_face = self._face_states(state)
        inverse_spacing = 1.0 / self.grid.spacing_m
        du = (
            self.grid.background_field_t
            / (MU0 * self.grid.mass_density_kg_m3)
            * np.diff(b_face)
            * inverse_spacing
        )
        db = (
            self.grid.background_field_t
            * np.diff(u_face)
            * inverse_spacing
        )
        return AlfvenState(du, db)

    @staticmethod
    def _add(state: AlfvenState, rhs: AlfvenState, scale: float) -> AlfvenState:
        return AlfvenState(
            state.velocity_m_s + scale * rhs.velocity_m_s,
            state.magnetic_t + scale * rhs.magnetic_t,
        )

    def step(self, state: AlfvenState, dt_s: float) -> AlfvenState:
        """Advance one SSPRK3 step."""

        state = state.validated(self.grid.cells)
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        if dt > self.maximum_stable_dt_s * (1.0 + 1.0e-12):
            raise ValueError(
                f"Alfven CFL exceeds one: dt={dt:g}, "
                f"limit={self.maximum_stable_dt_s:g}"
            )
        k1 = self.rhs(state)
        stage1 = self._add(state, k1, dt)
        k2 = self.rhs(stage1)
        euler2 = self._add(stage1, k2, dt)
        stage2 = AlfvenState(
            0.75 * state.velocity_m_s + 0.25 * euler2.velocity_m_s,
            0.75 * state.magnetic_t + 0.25 * euler2.magnetic_t,
        )
        k3 = self.rhs(stage2)
        euler3 = self._add(stage2, k3, dt)
        return AlfvenState(
            state.velocity_m_s / 3.0 + 2.0 * euler3.velocity_m_s / 3.0,
            state.magnetic_t / 3.0 + 2.0 * euler3.magnetic_t / 3.0,
        )

    def advance(
        self, state: AlfvenState, *, duration_s: float, cfl: float = 0.8
    ) -> AlfvenState:
        if not np.isfinite(cfl) or not (0.0 < cfl <= 1.0):
            raise ValueError("cfl must lie in (0, 1]")
        duration = float(duration_s)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_s must be non-negative and finite")
        result = state.validated(self.grid.cells)
        time_s = 0.0
        base_dt = cfl * self.maximum_stable_dt_s
        while time_s < duration - 1.0e-15:
            dt = min(base_dt, duration - time_s)
            result = self.step(result, dt)
            time_s += dt
        return result


def right_going_state(
    velocity_m_s: np.ndarray, grid: FieldLineGrid
) -> AlfvenState:
    """Construct a pure right-going state satisfying the Walén relation."""

    velocity = _as_finite_vector("velocity_m_s", velocity_m_s, grid.cells)
    return AlfvenState(
        velocity,
        -grid.magnetic_impedance_t_s_m * velocity,
    )


__all__ = [
    "AlfvenEnergy",
    "AlfvenState",
    "CharacteristicBoundary",
    "ConductanceLoad",
    "FieldLineGrid",
    "MU0",
    "PeriodicMidpointAlfvenSolver",
    "UpwindAlfvenSolver",
    "alfven_energy",
    "characteristic_amplitudes",
    "poynting_flux_w_m2",
    "right_going_state",
    "state_from_characteristics",
]
