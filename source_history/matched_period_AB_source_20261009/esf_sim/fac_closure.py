"""Dimensional field-aligned-current forcing of the Li ESF closure.

The electrostatic Li model solves, for ``psi = Phi / B0``,

    div_perp(n * nu_in * grad_perp(psi)) = -g * d_x(n).

Restoring the field-aligned part of current continuity gives the *local*
volume equation

    div_perp(n * nu_in * grad_perp(psi))
        = -g * d_x(n) + B0 / m_i * d_s(j_parallel).

Consequently a current density in A m^-2 is not, by itself, a valid source
for this *local* model.  Its source is the field-aligned divergence in
A m^-3.  A height-integrated closure instead requires two independent
coefficients, with ``W=B_a/B(s)``:

``N_col = integral n W ds`` and ``Cnu_col = integral n nu_in W ds``.

Writing ``N_col=H_n n_a`` and ``Cnu_col=H_nu n_a nu_a`` gives

``div_perp(H_nu n nu grad(psi))``
``    = -g d_x(H_n n) + B0/m_i * J_parallel_surface``.

Here ``J_parallel_surface`` [A m^-2] is the sum of the outward currents at
the conjugate surfaces, not a volume source formed by dividing by a length.

The production point-centred matrix represents the negative normalized
operator, so its dimensional FAC contribution to the algebraic right-hand
side is ``-B0*d_s(j_parallel)/(m_i*n*nu_in)``.  Keeping that conversion in one
place prevents the transmission-line code from depending on the historical
normalization of equation (2.21).

Supplying both ``gravity_column_weight_m=H_n`` and
``collision_column_weight_m=H_nu`` switches to the conservative column
equation.  Omitting both preserves the historical local/scalar-H operators
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import spsolve

from .config import SimulationConfig
from .point_centered_potential import (
    build_equal_extrapolated_log_point_centered_matrix,
)
from .mimetic_potential import build_harmonic_equal_extrapolated_matrix
from .potential import PotentialDiagnostics


FloatArray = NDArray[np.float64]
ATOMIC_MASS_KG = 1.660_539_066_60e-27
DEFAULT_OXYGEN_ION_MASS_KG = 15.999 * ATOMIC_MASS_KG


class UnforcedPotentialSolver(Protocol):
    """The exact historical electrostatic branch used at zero FAC."""

    def solve(
        self, density: np.ndarray
    ) -> tuple[np.ndarray, PotentialDiagnostics]: ...


@dataclass(frozen=True)
class ParallelCurrentGradient:
    """Local field-aligned current divergence ``d_s j_parallel`` [A m^-3]."""

    d_j_parallel_ds_a_m3: FloatArray

    def __init__(self, d_j_parallel_ds_a_m3: ArrayLike) -> None:
        values = np.array(d_j_parallel_ds_a_m3, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.size == 0:
            raise ValueError(
                "d_j_parallel_ds_a_m3 must be a non-empty two-dimensional array"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("d_j_parallel_ds_a_m3 must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "d_j_parallel_ds_a_m3", values)


@dataclass(frozen=True)
class ParallelSurfaceCurrent:
    """Total conjugate outward FAC surface current ``J`` [A m^-2]."""

    surface_current_a_m2: FloatArray

    def __init__(self, surface_current_a_m2: ArrayLike) -> None:
        values = np.array(surface_current_a_m2, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.size == 0:
            raise ValueError(
                "surface_current_a_m2 must be a non-empty two-dimensional array"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("surface_current_a_m2 must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "surface_current_a_m2", values)


FACForcing = ParallelCurrentGradient | ParallelSurfaceCurrent


@dataclass(frozen=True)
class FACClosureDiagnostics:
    """Auditable diagnostics for one dimensional FAC-forced solve."""

    potential: PotentialDiagnostics
    net_current_gradient_a_m: float
    relative_net_current_gradient: float
    fac_rhs_l2_s_inv: float
    gravity_rhs_l2_s_inv: float
    fac_to_gravity_ratio: float
    forcing_representation: str = "volume_current_gradient"
    forcing_units: str = "A m^-3"


@dataclass(frozen=True)
class FACLinearComponents:
    """Algebraic ingredients ``K psi = r_g + f_fac * d_s(j)``.

    ``matrix`` includes the historical gauge row.  The two arrays have the Li
    grid shape; their gauge entry is zero so an external coupled solver can add
    a low-rank FAC response without corrupting the gauge equation.
    """

    matrix: object
    gravity_rhs_s_inv: FloatArray
    fac_multiplier: FloatArray
    forcing_representation: str = "volume_current_gradient"


def normalized_fac_rhs_s_inv(
    density_m3: ArrayLike,
    collision_frequency_s: ArrayLike,
    forcing: ParallelCurrentGradient,
    *,
    background_field_t: float,
    ion_mass_kg: float,
    density_floor_m3: float = 0.0,
) -> FloatArray:
    """Return the FAC term for the point-centred negative operator [s^-1].

    The returned value is

    ``-B0/(m_i*n*nu_in) * d_s(j_parallel)``.
    """

    density = np.asarray(density_m3, dtype=np.float64)
    if density.shape != forcing.d_j_parallel_ds_a_m3.shape:
        raise ValueError("density and current-gradient shapes differ")
    if not np.all(np.isfinite(density)):
        raise ValueError("density_m3 must be finite")
    floor = float(density_floor_m3)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("density_floor_m3 must be non-negative and finite")
    positive_density = np.maximum(density, floor)
    if np.any(positive_density <= 0.0):
        raise ValueError("density_m3 must be positive after applying the floor")

    collision = np.asarray(collision_frequency_s, dtype=np.float64)
    if collision.ndim == 1:
        if collision.shape != (density.shape[0],):
            raise ValueError("one-dimensional collision frequency has wrong length")
        collision = collision[:, None]
    if collision.shape not in (density.shape, (density.shape[0], 1)):
        raise ValueError("collision_frequency_s has incompatible shape")
    if not np.all(np.isfinite(collision)) or np.any(collision <= 0.0):
        raise ValueError("collision_frequency_s must be positive and finite")

    field = float(background_field_t)
    mass = float(ion_mass_kg)
    if not np.isfinite(field) or field <= 0.0:
        raise ValueError("background_field_t must be positive and finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("ion_mass_kg must be positive and finite")
    return (
        -field
        * forcing.d_j_parallel_ds_a_m3
        / (mass * positive_density * collision)
    )


class DimensionalFACPointCenteredClosure:
    """Equation-(2.21) closure with a local, SI-unit FAC-divergence source.

    Passing ``forcing=None`` is deliberately a literal call to
    ``unforced_solver.solve(density)`` when a historical solver is supplied.
    This preserves the existing best-baseline branch exactly at zero coupling.
    The forced branch reuses the historical solver's cached sparse backend when
    it exposes ``_solve`` (the PARDISO implementation does), otherwise SciPy's
    sparse direct solver is used for small verification problems.
    """

    def __init__(
        self,
        config: SimulationConfig,
        *,
        background_field_t: float,
        ion_mass_kg: float = DEFAULT_OXYGEN_ION_MASS_KG,
        unforced_solver: UnforcedPotentialSolver | None = None,
        current_balance_rtol: float = 1.0e-12,
        gravity_column_weight_m: ArrayLike | None = None,
        collision_column_weight_m: ArrayLike | None = None,
    ) -> None:
        if config.potential_boundary_y != "equal_extrapolation":
            raise ValueError(
                "dimensional FAC point-centred closure requires "
                "potential_boundary_y=equal_extrapolation"
            )
        self.config = config
        self.background_field_t = float(background_field_t)
        self.ion_mass_kg = float(ion_mass_kg)
        self.unforced_solver = unforced_solver
        self.current_balance_rtol = float(current_balance_rtol)
        if not np.isfinite(self.background_field_t) or self.background_field_t <= 0:
            raise ValueError("background_field_t must be positive and finite")
        if not np.isfinite(self.ion_mass_kg) or self.ion_mass_kg <= 0:
            raise ValueError("ion_mass_kg must be positive and finite")
        if (
            not np.isfinite(self.current_balance_rtol)
            or self.current_balance_rtol < 0.0
        ):
            raise ValueError("current_balance_rtol must be non-negative and finite")
        self.collision_frequency_s = config.collision_nu0_s * np.exp(
            -config.y_coordinates_m / config.collision_frequency_scale_height_m
        )
        supplied = (
            gravity_column_weight_m is not None,
            collision_column_weight_m is not None,
        )
        if supplied[0] != supplied[1]:
            raise ValueError(
                "gravity_column_weight_m and collision_column_weight_m "
                "must be supplied together"
            )
        self.gravity_column_weight_m = self._validated_column_weight(
            "gravity_column_weight_m", gravity_column_weight_m
        )
        self.collision_column_weight_m = self._validated_column_weight(
            "collision_column_weight_m", collision_column_weight_m
        )
        self.column_integrated = self.gravity_column_weight_m is not None
        self.last_fac_diagnostics: FACClosureDiagnostics | None = None

    def _validated_column_weight(
        self, name: str, value: ArrayLike | None
    ) -> FloatArray | None:
        if value is None:
            return None
        column = np.asarray(value, dtype=np.float64)
        if column.ndim == 1:
            if column.shape != (self.config.ny,):
                raise ValueError(f"one-dimensional {name} must match ny")
            column = np.broadcast_to(
                column[:, None], (self.config.ny, self.config.nx)
            )
        if column.shape != (self.config.ny, self.config.nx):
            raise ValueError(f"{name} has incompatible shape")
        if not np.all(np.isfinite(column)) or np.any(column <= 0.0):
            raise ValueError(f"{name} must be positive and finite")
        result = np.array(column, copy=True)
        result.setflags(write=False)
        return result

    def _transport_coefficient(self, positive_density: FloatArray) -> FloatArray:
        coefficient = positive_density * self.collision_frequency_s[:, None]
        if self.collision_column_weight_m is not None:
            coefficient = coefficient * self.collision_column_weight_m
        return coefficient

    def _current_balance(
        self, forcing: FACForcing
    ) -> tuple[float, float]:
        """Return the conservative transverse current integral and defect."""

        if isinstance(forcing, ParallelSurfaceCurrent):
            weighted = forcing.surface_current_a_m2
        else:
            weighted = forcing.d_j_parallel_ds_a_m3
        area = self.config.dx_m * self.config.dy_m
        net = float(np.sum(weighted, dtype=np.float64) * area)
        absolute = float(np.sum(np.abs(weighted), dtype=np.float64) * area)
        relative = abs(net) / max(absolute, np.finfo(np.float64).tiny)
        return net, relative

    def _gravity_rhs(self, positive_density: FloatArray) -> FloatArray:
        log_density = np.log(positive_density)
        derivative = (
            np.roll(log_density, -1, axis=1)
            - np.roll(log_density, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        result = (
            self.config.gravity_m_s2
            * derivative
            / self.collision_frequency_s[:, None]
        )
        if self.column_integrated:
            result = result * (
                self.gravity_column_weight_m
                / self.collision_column_weight_m
            )
        return result

    def _validate_forcing(
        self, forcing: FACForcing, expected_shape: tuple[int, int]
    ) -> FloatArray:
        if self.column_integrated:
            if not isinstance(forcing, ParallelSurfaceCurrent):
                raise TypeError(
                    "column-integrated closure requires ParallelSurfaceCurrent"
                )
            values = forcing.surface_current_a_m2
        else:
            if not isinstance(forcing, ParallelCurrentGradient):
                raise TypeError(
                    "local closure requires ParallelCurrentGradient"
                )
            values = forcing.d_j_parallel_ds_a_m3
        if values.shape != expected_shape:
            raise ValueError("forcing and density shapes differ")
        return values

    def _fac_multiplier(
        self, positive_density: FloatArray
    ) -> FloatArray:
        denominator = (
            self.ion_mass_kg
            * positive_density
            * self.collision_frequency_s[:, None]
        )
        if self.collision_column_weight_m is not None:
            denominator = denominator * self.collision_column_weight_m
        return -self.background_field_t / denominator

    def linear_components(self, density: ArrayLike) -> FACLinearComponents:
        """Return the point-centred matrix and dimensional source conversion."""

        density_array = np.asarray(density, dtype=np.float64)
        expected_shape = (self.config.ny, self.config.nx)
        if density_array.shape != expected_shape:
            raise ValueError(
                f"density must have shape {expected_shape}, got {density_array.shape}"
            )
        if not np.all(np.isfinite(density_array)):
            raise ValueError("density must be finite")
        positive = np.maximum(density_array, self.config.density_floor_m3)
        coefficient = self._transport_coefficient(positive)
        matrix = build_equal_extrapolated_log_point_centered_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        gravity = self._gravity_rhs(positive)
        multiplier = self._fac_multiplier(positive) * np.ones_like(positive)
        gravity = np.array(gravity, copy=True)
        multiplier = np.array(multiplier, copy=True)
        gravity.flat[0] = 0.0
        multiplier.flat[0] = 0.0
        return FACLinearComponents(
            matrix,
            gravity,
            multiplier,
            (
                "surface_current"
                if self.column_integrated
                else "volume_current_gradient"
            ),
        )

    def _solve_linear(self, matrix, rhs: FloatArray) -> tuple[FloatArray, bool]:
        backend = getattr(self.unforced_solver, "_solve", None)
        if callable(backend):
            solution, rebuilt = backend(matrix, rhs)
            return np.asarray(solution, dtype=np.float64), bool(rebuilt)
        return np.asarray(spsolve(matrix, rhs), dtype=np.float64), False

    def solve_algebraic(
        self, matrix: object, rhs: ArrayLike
    ) -> tuple[FloatArray, PotentialDiagnostics]:
        """Solve a gauge-fixed coupled algebraic system with standard checks."""

        right = np.asarray(rhs, dtype=np.float64).ravel()
        solution, rebuilt = self._solve_linear(matrix, right)
        residual = matrix @ solution - right
        denominator = max(float(np.linalg.norm(right)), np.finfo(np.float64).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.all(np.isfinite(solution)) or not np.isfinite(relative_residual):
            raise RuntimeError("coupled potential solve produced non-finite values")
        tolerance = max(float(self.config.potential_rtol), 1.0e-12)
        if relative_residual > tolerance:
            raise RuntimeError(
                f"coupled potential residual {relative_residual:.3e} "
                f"exceeds {tolerance:.3e}"
            )
        potential = solution.reshape((self.config.ny, self.config.nx))
        potential -= float(np.mean(potential))
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=rebuilt,
            used_pyamg=False,
        )

    def _internal_solve(
        self,
        density: FloatArray,
        forcing: FACForcing | None,
    ) -> tuple[FloatArray, PotentialDiagnostics]:
        positive_density = np.maximum(density, self.config.density_floor_m3)
        coefficient = self._transport_coefficient(positive_density)
        matrix = build_equal_extrapolated_log_point_centered_matrix(
            coefficient,
            self.config.dx_m,
            self.config.dy_m,
        )
        gravity_rhs = self._gravity_rhs(positive_density)
        fac_rhs = np.zeros_like(gravity_rhs)
        if forcing is not None:
            forcing_values = self._validate_forcing(forcing, density.shape)
            fac_rhs = self._fac_multiplier(positive_density) * forcing_values
        rhs = np.asarray(gravity_rhs + fac_rhs, dtype=np.float64).ravel()
        rhs[0] = 0.0
        solution, rebuilt = self._solve_linear(matrix, rhs)
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.all(np.isfinite(solution)) or not np.isfinite(relative_residual):
            raise RuntimeError("FAC-forced potential solve produced non-finite values")
        tolerance = max(float(self.config.potential_rtol), 1.0e-12)
        if relative_residual > tolerance:
            raise RuntimeError(
                f"FAC-forced potential residual {relative_residual:.3e} "
                f"exceeds {tolerance:.3e}"
            )
        potential = solution.reshape(density.shape)
        potential -= float(np.mean(potential))
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=rebuilt,
            used_pyamg=False,
        )

    def solve(
        self,
        density: ArrayLike,
        forcing: FACForcing | None = None,
    ) -> tuple[FloatArray, PotentialDiagnostics]:
        density_array = np.asarray(density, dtype=np.float64)
        expected_shape = (self.config.ny, self.config.nx)
        if density_array.shape != expected_shape:
            raise ValueError(
                f"density must have shape {expected_shape}, got {density_array.shape}"
            )
        if not np.all(np.isfinite(density_array)):
            raise ValueError("density must be finite")

        # This exact dispatch is the ES regression contract.
        if forcing is None and self.unforced_solver is not None:
            self.last_fac_diagnostics = None
            return self.unforced_solver.solve(density_array)

        if forcing is not None:
            forcing_values = self._validate_forcing(
                forcing, density_array.shape
            )
            net, relative_net = self._current_balance(forcing)
            if relative_net > self.current_balance_rtol:
                raise ValueError(
                    "parallel current has no transverse return path: "
                    f"relative area integral={relative_net:.3e}"
                )
        else:
            forcing_values = np.zeros_like(density_array)
            net = 0.0
            relative_net = 0.0

        potential, potential_diagnostics = self._internal_solve(
            density_array, forcing
        )
        positive = np.maximum(density_array, self.config.density_floor_m3)
        gravity_rhs = self._gravity_rhs(positive)
        fac_rhs = (
            np.zeros_like(gravity_rhs)
            if forcing is None
            else self._fac_multiplier(positive) * forcing_values
        )
        gravity_norm = float(np.linalg.norm(gravity_rhs.ravel()))
        fac_norm = float(np.linalg.norm(fac_rhs.ravel()))
        if gravity_norm == 0.0:
            ratio = 0.0 if fac_norm == 0.0 else float("inf")
        else:
            ratio = fac_norm / gravity_norm
        self.last_fac_diagnostics = FACClosureDiagnostics(
            potential=potential_diagnostics,
            net_current_gradient_a_m=net,
            relative_net_current_gradient=relative_net,
            fac_rhs_l2_s_inv=fac_norm,
            gravity_rhs_l2_s_inv=gravity_norm,
            fac_to_gravity_ratio=ratio,
            forcing_representation=(
                "surface_current"
                if self.column_integrated
                else "volume_current_gradient"
            ),
            forcing_units="A m^-2" if self.column_integrated else "A m^-3",
        )
        return potential, potential_diagnostics


class DimensionalFACMimeticClosure(DimensionalFACPointCenteredClosure):
    """Conservative face-flux Li closure with dimensional FAC forcing.

    This variant is the preferred electromagnetic energy-audit operator.  Its
    matrix represents ``-div((H_nu*n*nu/scale) grad(psi))`` (with
    ``H_nu=1`` on the historical path) and is symmetric before
    imposing the gauge.  The dimensional FAC contribution is therefore
    ``-B0*d_s(j_parallel)/(m_i*scale)``.  The gravity discretization is kept
    identical to :class:`HarmonicMimeticLogPotentialSolver` so an injected
    historical solver remains an exact zero-forcing oracle.
    """

    def _scaled_terms(
        self,
        positive_density: FloatArray,
        forcing: FACForcing | None,
    ) -> tuple[object, FloatArray, FloatArray, float]:
        coefficient = self._transport_coefficient(positive_density)
        matrix, scale = build_harmonic_equal_extrapolated_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        log_density = np.log(positive_density)
        log_density_dx = (
            np.roll(log_density, -1, axis=1)
            - np.roll(log_density, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        if self.column_integrated:
            gravity_rhs = (
                self.gravity_column_weight_m
                * positive_density
                / scale
                * self.config.gravity_m_s2
                * log_density_dx
            )
        else:
            gravity_rhs = (
                coefficient
                / scale
                * self.config.gravity_m_s2
                * log_density_dx
                / self.collision_frequency_s[:, None]
            )
        fac_rhs = np.zeros_like(gravity_rhs)
        if forcing is not None:
            forcing_values = self._validate_forcing(
                forcing, positive_density.shape
            )
            fac_rhs = (
                -self.background_field_t
                * forcing_values
                / (self.ion_mass_kg * scale)
            )
        return matrix, gravity_rhs, fac_rhs, scale

    def linear_components(self, density: ArrayLike) -> FACLinearComponents:
        density_array = np.asarray(density, dtype=np.float64)
        expected_shape = (self.config.ny, self.config.nx)
        if density_array.shape != expected_shape:
            raise ValueError(
                f"density must have shape {expected_shape}, got {density_array.shape}"
            )
        if not np.all(np.isfinite(density_array)):
            raise ValueError("density must be finite")
        positive = np.maximum(density_array, self.config.density_floor_m3)
        matrix, gravity, _, scale = self._scaled_terms(positive, None)
        multiplier = np.full_like(
            positive, -self.background_field_t / (self.ion_mass_kg * scale)
        )
        gravity = np.array(gravity, copy=True)
        gravity.flat[0] = 0.0
        multiplier.flat[0] = 0.0
        return FACLinearComponents(
            matrix,
            gravity,
            multiplier,
            (
                "surface_current"
                if self.column_integrated
                else "volume_current_gradient"
            ),
        )

    def _internal_solve(
        self,
        density: FloatArray,
        forcing: FACForcing | None,
    ) -> tuple[FloatArray, PotentialDiagnostics]:
        positive_density = np.maximum(density, self.config.density_floor_m3)
        matrix, gravity_rhs, fac_rhs, _ = self._scaled_terms(
            positive_density, forcing
        )
        rhs = np.asarray(gravity_rhs + fac_rhs, dtype=np.float64).ravel()
        rhs[0] = 0.0
        solution, rebuilt = self._solve_linear(matrix, rhs)
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.all(np.isfinite(solution)) or not np.isfinite(relative_residual):
            raise RuntimeError("mimetic FAC solve produced non-finite values")
        tolerance = max(float(self.config.potential_rtol), 1.0e-12)
        if relative_residual > tolerance:
            raise RuntimeError(
                f"mimetic FAC residual {relative_residual:.3e} "
                f"exceeds {tolerance:.3e}"
            )
        potential = solution.reshape(density.shape)
        potential -= float(np.mean(potential))
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=rebuilt,
            used_pyamg=False,
        )

    def solve(
        self,
        density: ArrayLike,
        forcing: FACForcing | None = None,
    ) -> tuple[FloatArray, PotentialDiagnostics]:
        # Preserve the literal zero-FAC production branch.
        if forcing is None and self.unforced_solver is not None:
            self.last_fac_diagnostics = None
            return self.unforced_solver.solve(np.asarray(density, dtype=np.float64))

        density_array = np.asarray(density, dtype=np.float64)
        expected_shape = (self.config.ny, self.config.nx)
        if density_array.shape != expected_shape:
            raise ValueError(
                f"density must have shape {expected_shape}, got {density_array.shape}"
            )
        if not np.all(np.isfinite(density_array)):
            raise ValueError("density must be finite")
        if forcing is not None:
            self._validate_forcing(forcing, density_array.shape)
            net, relative_net = self._current_balance(forcing)
            if relative_net > self.current_balance_rtol:
                raise ValueError(
                    "parallel current has no transverse return path: "
                    f"relative area integral={relative_net:.3e}"
                )
        else:
            net = 0.0
            relative_net = 0.0

        positive = np.maximum(density_array, self.config.density_floor_m3)
        _, gravity_rhs, fac_rhs, _ = self._scaled_terms(positive, forcing)
        potential, potential_diagnostics = self._internal_solve(
            density_array, forcing
        )
        gravity_norm = float(np.linalg.norm(gravity_rhs.ravel()))
        fac_norm = float(np.linalg.norm(fac_rhs.ravel()))
        ratio = (
            (0.0 if fac_norm == 0.0 else float("inf"))
            if gravity_norm == 0.0
            else fac_norm / gravity_norm
        )
        self.last_fac_diagnostics = FACClosureDiagnostics(
            potential=potential_diagnostics,
            net_current_gradient_a_m=net,
            relative_net_current_gradient=relative_net,
            fac_rhs_l2_s_inv=fac_norm,
            gravity_rhs_l2_s_inv=gravity_norm,
            fac_to_gravity_ratio=ratio,
            forcing_representation=(
                "surface_current"
                if self.column_integrated
                else "volume_current_gradient"
            ),
            forcing_units="A m^-2" if self.column_integrated else "A m^-3",
        )
        return potential, potential_diagnostics


__all__ = [
    "DEFAULT_OXYGEN_ION_MASS_KG",
    "DimensionalFACMimeticClosure",
    "DimensionalFACPointCenteredClosure",
    "FACClosureDiagnostics",
    "FACLinearComponents",
    "ParallelCurrentGradient",
    "ParallelSurfaceCurrent",
    "normalized_fac_rhs_s_inv",
]
