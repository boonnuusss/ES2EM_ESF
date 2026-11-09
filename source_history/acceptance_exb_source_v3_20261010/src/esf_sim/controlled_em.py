"""Controlled two-way coupling primitives for reduced electromagnetic tests.

This module is deliberately a *verifiable control layer*.  It provides the
bookkeeping needed to test a two-way coupling algorithm (gain ramps, net-FAC
projection, bounded feedback, a boundary fixed-point iteration, and an
adjoint channel map) without pretending that a physical Li-model closure has
already been derived.

In particular, values passed to :func:`condition_fac_source` must already be
in a mutually compatible normalization.  A production ``ForcedLiClosure``
still requires a documented physical normalization that converts field-line
FAC/electrodynamic variables into the units and discrete forcing of the Li
perpendicular model.  The toy fixed-point relation below is a numerical
verification problem, not that missing physical closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _finite_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class FeedbackConfig:
    """Numerical controls for a deliberately bounded feedback experiment."""

    gain: float = 0.0
    ramp_time_s: float = 0.0
    under_relaxation: float = 1.0
    max_iterations: int = 50
    relative_tolerance: float = 1.0e-10
    max_source_ratio: float = 0.1

    def __post_init__(self) -> None:
        gain = _finite_scalar("gain", self.gain)
        ramp_time_s = _finite_scalar("ramp_time_s", self.ramp_time_s)
        under_relaxation = _finite_scalar(
            "under_relaxation", self.under_relaxation
        )
        relative_tolerance = _finite_scalar(
            "relative_tolerance", self.relative_tolerance
        )
        max_source_ratio = _finite_scalar(
            "max_source_ratio", self.max_source_ratio
        )

        if not 0.0 <= gain <= 1.0:
            raise ValueError("gain must lie in [0, 1]")
        if ramp_time_s < 0.0:
            raise ValueError("ramp_time_s must be non-negative")
        if not 0.0 < under_relaxation <= 1.0:
            raise ValueError("under_relaxation must lie in (0, 1]")
        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations, (int, np.integer)
        ):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive")
        if max_source_ratio <= 0.0:
            raise ValueError("max_source_ratio must be positive")

    def ramp_fraction(self, time_s: float) -> float:
        """Return a clipped linear start-up ramp in ``[0, 1]``."""

        time_s = _finite_scalar("time_s", time_s)
        if time_s < 0.0:
            raise ValueError("time_s must be non-negative")
        if self.ramp_time_s == 0.0:
            return 1.0
        return float(np.clip(time_s / self.ramp_time_s, 0.0, 1.0))

    def effective_gain(self, time_s: float) -> float:
        """Return ``gain`` multiplied by the start-up ramp."""

        return self.gain * self.ramp_fraction(time_s)


def _same_shape_finite_array(name: str, value: ArrayLike, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_area(cell_area: ArrayLike, shape: tuple[int, ...]) -> FloatArray:
    area = _same_shape_finite_array("cell_area", cell_area, shape)
    if np.any(area <= 0.0):
        raise ValueError("cell_area must be strictly positive")
    return area


def area_weighted_mean(field: ArrayLike, cell_area: ArrayLike) -> float:
    """Return the finite-volume area-weighted mean of ``field``."""

    values = np.asarray(field, dtype=np.float64)
    if values.size == 0:
        raise ValueError("field must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("field must contain only finite values")
    area = _positive_area(cell_area, values.shape)
    return float(np.sum(area * values) / np.sum(area))


def area_weighted_l2(field: ArrayLike, cell_area: ArrayLike) -> float:
    """Return ``sqrt(sum(area * field**2))``."""

    values = np.asarray(field, dtype=np.float64)
    if values.size == 0:
        raise ValueError("field must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("field must contain only finite values")
    area = _positive_area(cell_area, values.shape)
    return float(np.sqrt(np.sum(area * values * values)))


def project_area_weighted_zero_mean(
    fac: ArrayLike, cell_area: ArrayLike
) -> FloatArray:
    """Project FAC onto the discrete zero-net-current subspace.

    A new array is always returned; neither input is modified.
    """

    values = np.asarray(fac, dtype=np.float64)
    if values.size == 0:
        raise ValueError("fac must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("fac must contain only finite values")
    area = _positive_area(cell_area, values.shape)
    mean = float(np.sum(area * values) / np.sum(area))
    return values - mean


@dataclass(frozen=True)
class FACSourceDiagnostics:
    """Auditable diagnostics for FAC projection and feedback limiting."""

    effective_gain: float
    input_weighted_mean: float
    projected_weighted_mean: float
    raw_fac_l2: float
    projected_fac_l2: float
    reference_source_l2: float
    unlimited_source_l2: float
    applied_scale: float
    unlimited_source_ratio: float
    final_source_ratio: float
    limited: bool


@dataclass(frozen=True)
class ConditionedFACSource:
    """Zero-mean, gain-scaled and source-ratio-limited feedback field."""

    source: FloatArray
    projected_fac: FloatArray
    diagnostics: FACSourceDiagnostics


def condition_fac_source(
    fac: ArrayLike,
    reference_source: ArrayLike,
    cell_area: ArrayLike,
    config: FeedbackConfig,
    *,
    time_s: float,
) -> ConditionedFACSource:
    """Project and bound a pre-normalized FAC feedback source.

    The unlimited source is ``effective_gain * projected_fac``.  Its weighted
    L2 norm is clipped to ``max_source_ratio`` times the reference-source norm.
    If the reference norm is zero, the only finite admissible source is zero.
    """

    fac_array = np.asarray(fac, dtype=np.float64)
    if fac_array.size == 0:
        raise ValueError("fac must not be empty")
    if not np.all(np.isfinite(fac_array)):
        raise ValueError("fac must contain only finite values")
    reference = _same_shape_finite_array(
        "reference_source", reference_source, fac_array.shape
    )
    area = _positive_area(cell_area, fac_array.shape)

    input_mean = area_weighted_mean(fac_array, area)
    projected = project_area_weighted_zero_mean(fac_array, area)
    projected_mean = area_weighted_mean(projected, area)
    gain = config.effective_gain(time_s)
    unlimited = gain * projected

    raw_norm = area_weighted_l2(fac_array, area)
    projected_norm = area_weighted_l2(projected, area)
    reference_norm = area_weighted_l2(reference, area)
    unlimited_norm = area_weighted_l2(unlimited, area)
    allowed_norm = config.max_source_ratio * reference_norm

    if unlimited_norm == 0.0:
        scale = 1.0
        unlimited_ratio = 0.0
    elif reference_norm == 0.0:
        scale = 0.0
        unlimited_ratio = float("inf")
    else:
        unlimited_ratio = unlimited_norm / reference_norm
        scale = min(1.0, allowed_norm / unlimited_norm)

    source = unlimited * scale
    final_norm = area_weighted_l2(source, area)
    if reference_norm == 0.0:
        final_ratio = 0.0 if final_norm == 0.0 else float("inf")
    else:
        final_ratio = final_norm / reference_norm

    diagnostics = FACSourceDiagnostics(
        effective_gain=gain,
        input_weighted_mean=input_mean,
        projected_weighted_mean=projected_mean,
        raw_fac_l2=raw_norm,
        projected_fac_l2=projected_norm,
        reference_source_l2=reference_norm,
        unlimited_source_l2=unlimited_norm,
        applied_scale=scale,
        unlimited_source_ratio=unlimited_ratio,
        final_source_ratio=final_ratio,
        limited=scale < 1.0,
    )
    return ConditionedFACSource(
        source=np.array(source, copy=True),
        projected_fac=np.array(projected, copy=True),
        diagnostics=diagnostics,
    )


def dispatch_feedback(
    config: FeedbackConfig,
    *,
    time_s: float,
    unforced: Callable[[], Any],
    forced: Callable[[float], Any],
) -> Any:
    """Select the exact unforced branch or the controlled forced branch.

    The literal zero-gain branch intentionally does not evaluate a potential
    ramp or call ``forced``.  This makes the electrostatic regression path easy
    to prove with a callback-spy test.
    """

    if config.gain == 0.0:
        return unforced()
    return forced(config.effective_gain(time_s))


class CouplingConvergenceError(RuntimeError):
    """Raised when the toy boundary fixed-point iteration does not converge."""

    def __init__(self, iterations: int, relative_residual: float) -> None:
        self.iterations = int(iterations)
        self.relative_residual = float(relative_residual)
        super().__init__(
            "boundary fixed-point iteration failed after "
            f"{self.iterations} iterations "
            f"(relative residual={self.relative_residual:.6e})"
        )


@dataclass(frozen=True)
class BoundaryFixedPointResult:
    """Converged solution of the toy boundary-load relation."""

    potential: FloatArray
    fac: FloatArray
    iterations: int
    relative_residual: float


class BoundaryFixedPointSolver:
    r"""Solve the verification relation

    .. math::

       \phi = \phi_0 + k j, \qquad j = j_{free} + Y\phi

    by an under-relaxed Picard iteration.  ``k`` and ``Y`` are elementwise toy
    coefficients; this class is not a physical ionospheric closure.
    """

    def __init__(self, config: FeedbackConfig) -> None:
        self.config = config

    def solve(
        self,
        phi0: ArrayLike,
        j_free: ArrayLike,
        k: ArrayLike,
        admittance: ArrayLike,
        *,
        initial_potential: ArrayLike | None = None,
    ) -> BoundaryFixedPointResult:
        phi0_array = np.asarray(phi0, dtype=np.float64)
        if phi0_array.size == 0:
            raise ValueError("phi0 must not be empty")
        if not np.all(np.isfinite(phi0_array)):
            raise ValueError("phi0 must contain only finite values")
        shape = phi0_array.shape
        j_free_array = _same_shape_finite_array("j_free", j_free, shape)

        # Coefficients may be scalar or have exactly the potential shape.
        k_array = np.asarray(k, dtype=np.float64)
        y_array = np.asarray(admittance, dtype=np.float64)
        for name, coefficient in (("k", k_array), ("admittance", y_array)):
            if coefficient.shape not in ((), shape):
                raise ValueError(
                    f"{name} must be scalar or have shape {shape}, "
                    f"got {coefficient.shape}"
                )
            if not np.all(np.isfinite(coefficient)):
                raise ValueError(f"{name} must contain only finite values")

        if initial_potential is None:
            potential = np.array(phi0_array, copy=True)
        else:
            potential = np.array(
                _same_shape_finite_array(
                    "initial_potential", initial_potential, shape
                ),
                copy=True,
            )

        omega = self.config.under_relaxation
        tolerance = self.config.relative_tolerance
        relative_residual = float("inf")
        for iteration in range(1, self.config.max_iterations + 1):
            fac = j_free_array + y_array * potential
            target = phi0_array + k_array * fac
            next_potential = potential + omega * (target - potential)

            residual = target - potential
            denominator = max(
                float(np.linalg.norm(next_potential.ravel())),
                float(np.linalg.norm(phi0_array.ravel())),
                np.finfo(np.float64).tiny,
            )
            relative_residual = float(
                np.linalg.norm(residual.ravel()) / denominator
            )
            potential = next_potential
            if relative_residual <= tolerance:
                final_fac = j_free_array + y_array * potential
                # Report the residual of the returned pair, not the previous
                # iteration, so strict tests can verify both toy equations.
                final_target = phi0_array + k_array * final_fac
                final_denominator = max(
                    float(np.linalg.norm(potential.ravel())),
                    float(np.linalg.norm(phi0_array.ravel())),
                    np.finfo(np.float64).tiny,
                )
                final_residual = float(
                    np.linalg.norm((final_target - potential).ravel())
                    / final_denominator
                )
                return BoundaryFixedPointResult(
                    potential=np.array(potential, copy=True),
                    fac=np.array(final_fac, copy=True),
                    iterations=iteration,
                    relative_residual=final_residual,
                )

        raise CouplingConvergenceError(
            self.config.max_iterations, relative_residual
        )


@dataclass(frozen=True)
class ChannelMap:
    r"""Linear channel-to-grid map with its weighted power adjoint.

    If ``M`` maps channel amplitudes ``c`` to grid values, the adjoint is

    .. math:: M^* = W_c^{-1} M^T W_g,

    which guarantees ``<M c, f>_grid = <c, M* f>_channel``.  This identity is
    the minimal power-consistency check needed before adding a physical map.
    """

    matrix: FloatArray
    cell_area: FloatArray
    channel_weight: FloatArray

    def __init__(
        self,
        matrix: ArrayLike,
        cell_area: ArrayLike,
        channel_weight: ArrayLike | None = None,
    ) -> None:
        matrix_array = np.asarray(matrix, dtype=np.float64)
        if matrix_array.ndim != 2 or 0 in matrix_array.shape:
            raise ValueError("matrix must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(matrix_array)):
            raise ValueError("matrix must contain only finite values")
        n_cells, n_channels = matrix_array.shape

        area = np.asarray(cell_area, dtype=np.float64)
        if area.shape != (n_cells,):
            raise ValueError(f"cell_area must have shape ({n_cells},)")
        if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
            raise ValueError("cell_area must be finite and strictly positive")

        if channel_weight is None:
            weight = np.ones(n_channels, dtype=np.float64)
        else:
            weight = np.asarray(channel_weight, dtype=np.float64)
            if weight.shape != (n_channels,):
                raise ValueError(
                    f"channel_weight must have shape ({n_channels},)"
                )
            if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
                raise ValueError(
                    "channel_weight must be finite and strictly positive"
                )

        object.__setattr__(self, "matrix", np.array(matrix_array, copy=True))
        object.__setattr__(self, "cell_area", np.array(area, copy=True))
        object.__setattr__(self, "channel_weight", np.array(weight, copy=True))

    @property
    def n_cells(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.matrix.shape[1])

    def to_grid(self, channels: ArrayLike) -> FloatArray:
        channels_array = np.asarray(channels, dtype=np.float64)
        if channels_array.shape != (self.n_channels,):
            raise ValueError(
                f"channels must have shape ({self.n_channels},)"
            )
        if not np.all(np.isfinite(channels_array)):
            raise ValueError("channels must contain only finite values")
        return self.matrix @ channels_array

    def adjoint_to_channels(self, grid_field: ArrayLike) -> FloatArray:
        grid_array = np.asarray(grid_field, dtype=np.float64)
        if grid_array.shape != (self.n_cells,):
            raise ValueError(f"grid_field must have shape ({self.n_cells},)")
        if not np.all(np.isfinite(grid_array)):
            raise ValueError("grid_field must contain only finite values")
        return (
            self.matrix.T @ (self.cell_area * grid_array)
        ) / self.channel_weight

    def grid_pairing(self, left: ArrayLike, right: ArrayLike) -> float:
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        if left_array.shape != (self.n_cells,) or right_array.shape != (
            self.n_cells,
        ):
            raise ValueError("grid pairing operands have an invalid shape")
        return float(np.sum(self.cell_area * left_array * right_array))

    def channel_pairing(self, left: ArrayLike, right: ArrayLike) -> float:
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        if left_array.shape != (self.n_channels,) or right_array.shape != (
            self.n_channels,
        ):
            raise ValueError("channel pairing operands have an invalid shape")
        return float(np.sum(self.channel_weight * left_array * right_array))
