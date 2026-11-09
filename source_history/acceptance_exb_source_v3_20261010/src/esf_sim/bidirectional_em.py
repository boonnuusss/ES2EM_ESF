"""Nonlinear, dimensional Li--Alfven feedback on a transverse modal basis.

This module is the first physical two-way path in the project.  It differs
from :mod:`esf_sim.controlled_em`, whose pointwise fixed-point relation is a
control-algorithm test only.  Here the field-aligned current is constructed as

``A_parallel --(-Laplacian_perp/mu0)--> j_parallel``.

The legacy symmetric coupling treats the Li slice as a uniform central
control volume and returns ``q_parallel=(j_N+j_S)/H``.  The variable-column
coupling instead integrates the Li equation along the tube and passes the
surface current ``J_parallel=j_N+j_S`` directly, together with independent
particle and collision column weights.
The older one-sided boundary-gradient source remains available only for
diagnostic regression because its algebraic load is non-coercive and scales as
``1/ds``. The source enters only the Li electric-potential closure, with SI
units carried by
:class:`~esf_sim.fac_closure.ParallelCurrentGradient`.  Density is still a
two-dimensional Li slice, so the reduced model deliberately omits common-mode
parallel mass transport; FAC changes density only through the self-consistent
electric field and ExB advection.

One simulated half-line starts at the Li plane and points along ``+B``; a
configurable hemisphere multiplicity represents mirrored north/south lines.
The parallel averaging length is explicit and is held fixed when the exterior
line grid is refined.  The remote end may be the historical fixed voltage, a
matched outgoing characteristic, or a finite Pedersen-conductance load.  The
latter two are energy-passive and avoid repeatedly returning a lossless
``R=-1`` reflection to the Li plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from ._model_impl import CFLViolation, RhsDiagnostics
from .alfven_1d import MU0
from .fac_closure import (
    DimensionalFACPointCenteredClosure,
    FACClosureDiagnostics,
    ParallelCurrentGradient,
    ParallelSurfaceCurrent,
)
from .modal_alfven import (
    LeftBoundaryResponse,
    ModalAlfvenLine,
    ModalAlfvenState,
    ModalBasis,
)
from .numerics import advection_rhs
from .potential import PotentialDiagnostics


FloatArray = NDArray[np.float64]


class LegacyLiModel(Protocol):
    config: object

    def evaluate_rhs(
        self, density: np.ndarray
    ) -> tuple[np.ndarray, RhsDiagnostics]: ...

    def velocity_from_potential(
        self, potential: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def ssprk3_step(
        self, density: np.ndarray, dt_s: float
    ) -> tuple[np.ndarray, RhsDiagnostics]: ...


@dataclass(frozen=True)
class PhysicalFeedbackConfig:
    """Controls for the dimensional boundary coupling.

    ``gain`` is a port-coupling continuation parameter.  ``gain=1`` applies
    the full derived FAC source; ``gain=0`` is reserved for the exact legacy
    ES branch.  During a C1 start-up ramp the same coupling coefficient ``c``
    multiplies the Li voltage presented to the Alfvén line and the FAC source
    returned to Li.  The two port powers are therefore identical, while the
    line itself always propagates at its physical Alfvén speed.  No source
    clipping is performed: non-convergence rejects the RHS/step.
    """

    gain: float = 1.0
    ramp_time_s: float = 0.0
    under_relaxation: float = 0.5
    max_iterations: int = 80
    relative_tolerance: float = 1.0e-9
    line_cfl_limit: float = 0.8
    fac_source_model: Literal[
        "boundary_gradient", "symmetric_flux_tube"
    ] = "boundary_gradient"
    parallel_averaging_length_m: float | None = None
    gravity_column_weight_m: ArrayLike | None = None
    collision_column_weight_m: ArrayLike | None = None
    hemisphere_count: int = 1
    solver: Literal[
        "modal_woodbury", "direct_krylov", "fixed_point"
    ] = "modal_woodbury"
    background_exb_y_m_s: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.gain,
            self.background_exb_y_m_s,
            self.ramp_time_s,
            self.under_relaxation,
            self.relative_tolerance,
            self.line_cfl_limit,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("feedback parameters must be finite")
        if not 0.0 <= self.gain <= 1.0:
            raise ValueError("gain must lie in [0, 1]")
        if self.ramp_time_s < 0.0:
            raise ValueError("ramp_time_s must be non-negative")
        if not 0.0 < self.under_relaxation <= 1.0:
            raise ValueError("under_relaxation must lie in (0, 1]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive")
        if not 0.0 < self.line_cfl_limit <= 1.0:
            raise ValueError("line_cfl_limit must lie in (0, 1]")
        if self.fac_source_model not in (
            "boundary_gradient",
            "symmetric_flux_tube",
        ):
            raise ValueError(
                "fac_source_model must be 'boundary_gradient' or "
                "'symmetric_flux_tube'"
            )
        if (
            not isinstance(self.hemisphere_count, int)
            or isinstance(self.hemisphere_count, bool)
            or self.hemisphere_count < 1
        ):
            raise ValueError("hemisphere_count must be a positive integer")
        profiles = (
            self.gravity_column_weight_m,
            self.collision_column_weight_m,
        )
        if (profiles[0] is None) != (profiles[1] is None):
            raise ValueError(
                "gravity_column_weight_m and collision_column_weight_m "
                "must be supplied together"
            )
        if profiles[0] is not None:
            for name, profile in zip(
                ("gravity_column_weight_m", "collision_column_weight_m"),
                profiles,
                strict=True,
            ):
                profile_array = np.array(profile, dtype=np.float64, copy=True)
                if (
                    profile_array.ndim != 1
                    or profile_array.size == 0
                    or not np.all(np.isfinite(profile_array))
                    or np.any(profile_array <= 0.0)
                ):
                    raise ValueError(
                        f"{name} must be a positive finite one-dimensional array"
                    )
                profile_array.setflags(write=False)
                object.__setattr__(self, name, profile_array)
        if self.fac_source_model == "symmetric_flux_tube":
            supplied = int(self.parallel_averaging_length_m is not None) + int(
                profiles[0] is not None
            )
            if supplied != 1:
                raise ValueError(
                    "symmetric_flux_tube requires either a scalar "
                    "parallel_averaging_length_m or the two column weights"
                )
            if self.parallel_averaging_length_m is not None and (
                not np.isfinite(float(self.parallel_averaging_length_m))
                or float(self.parallel_averaging_length_m) <= 0.0
            ):
                raise ValueError(
                    "parallel_averaging_length_m must be positive and finite "
                    "for symmetric_flux_tube FAC forcing"
                )
        elif profiles[0] is not None:
            raise ValueError(
                "column weights are only valid for "
                "symmetric_flux_tube FAC forcing"
            )
        elif self.parallel_averaging_length_m is not None:
            length = float(self.parallel_averaging_length_m)
            if not np.isfinite(length) or length <= 0.0:
                raise ValueError(
                    "parallel_averaging_length_m must be positive and finite "
                    "when provided"
                )
        if self.solver not in (
            "modal_woodbury",
            "direct_krylov",
            "fixed_point",
        ):
            raise ValueError(
                "solver must be 'modal_woodbury', 'direct_krylov', or "
                "'fixed_point'"
            )

    def effective_gain(self, time_s: float) -> float:
        time = float(time_s)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time_s must be non-negative and finite")
        if self.ramp_time_s == 0.0:
            return float(self.gain)
        fraction = float(np.clip(time / self.ramp_time_s, 0.0, 1.0))
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        return float(self.gain * smooth)


@dataclass(frozen=True)
class DimensionalEMState:
    """Li volume density plus transverse-modal Alfvén characteristics."""

    density_m3: FloatArray
    alfven: ModalAlfvenState

    def validated(
        self, density_shape: tuple[int, int], line: ModalAlfvenLine
    ) -> "DimensionalEMState":
        density = np.asarray(self.density_m3, dtype=np.float64)
        if density.shape != density_shape or not np.all(np.isfinite(density)):
            raise ValueError(
                f"density_m3 must be finite with shape {density_shape}"
            )
        return DimensionalEMState(
            density.copy(), self.alfven.validated(line.cells, line.mode_count)
        )


@dataclass(frozen=True)
class PhysicalCouplingDiagnostics:
    """Diagnostics for one self-consistent Li--Alfvén RHS evaluation."""

    electrostatic: RhsDiagnostics
    fac_closure: FACClosureDiagnostics | None
    effective_gain: float
    activation_fraction: float
    iterations: int
    relative_coupling_residual: float
    alfven_cfl_rate_s_inv: float
    combined_cfl_rate_s_inv: float
    maximum_abs_d_j_parallel_ds_a_m3: float
    net_d_j_parallel_ds_a_m: float
    boundary_characteristic_defect_v: float
    boundary_power_modal_w: float
    boundary_power_grid_w: float
    remote_boundary_power_modal_w: float
    remote_boundary_reflection_coefficient: float
    relative_power_pairing_defect: float
    fac_forcing_representation: str
    fac_forcing_units: str
    maximum_abs_fac_forcing: float
    net_fac_forcing_integral: float
    density_conservation_measure: str


class PhysicalCouplingConvergenceError(RuntimeError):
    """The dimensional boundary closure did not converge for a trial state."""

    def __init__(self, iterations: int, residual: float) -> None:
        self.iterations = int(iterations)
        self.residual = float(residual)
        super().__init__(
            "dimensional Li--Alfven boundary iteration failed after "
            f"{iterations} iterations (relative residual={residual:.6e})"
        )


def _state_linear_combination(
    weights: tuple[float, ...], states: tuple[DimensionalEMState, ...]
) -> DimensionalEMState:
    if not states or len(weights) != len(states):
        raise ValueError("weights and states must have matching non-zero lengths")
    density = np.zeros_like(states[0].density_m3)
    plus = np.zeros_like(states[0].alfven.w_plus_v)
    minus = np.zeros_like(states[0].alfven.w_minus_v)
    for weight, state in zip(weights, states, strict=True):
        density += weight * state.density_m3
        plus += weight * state.alfven.w_plus_v
        minus += weight * state.alfven.w_minus_v
    return DimensionalEMState(density, ModalAlfvenState(plus, minus))


class DimensionalBidirectionalModel:
    """Self-consistent reduced-EM density/FAC model with no time-level lag."""

    def __init__(
        self,
        legacy_li_model: LegacyLiModel,
        closure: DimensionalFACPointCenteredClosure,
        basis: ModalBasis,
        line: ModalAlfvenLine,
        feedback: PhysicalFeedbackConfig,
        *,
        right_boundary_voltage_v: ArrayLike | float = 0.0,
        right_boundary_model: Literal[
            "fixed_voltage", "matched", "conductance"
        ] = "fixed_voltage",
        right_boundary_conductance_s: float | None = None,
    ) -> None:
        self.legacy_li_model = legacy_li_model
        self.closure = closure
        self.basis = basis
        self.line = line
        self.feedback = feedback
        config = legacy_li_model.config
        self.config = config
        expected_shape = (int(config.ny), int(config.nx))
        if (basis.ny, basis.nx) != expected_shape:
            raise ValueError("modal basis and Li grid shapes differ")
        if line.mode_count != basis.mode_count:
            raise ValueError("field line and modal basis mode counts differ")
        gravity_profile = feedback.gravity_column_weight_m
        collision_profile = feedback.collision_column_weight_m
        self.column_integrated = gravity_profile is not None
        if self.column_integrated:
            gravity_array = np.asarray(gravity_profile, dtype=np.float64)
            collision_array = np.asarray(collision_profile, dtype=np.float64)
            if gravity_array.shape != (int(config.ny),) or (
                collision_array.shape != (int(config.ny),)
            ):
                raise ValueError(
                    "column weights must have one value for every Li y row"
                )
            gravity_field = np.broadcast_to(
                gravity_array[:, None], expected_shape
            ).copy()
            collision_field = np.broadcast_to(
                collision_array[:, None], expected_shape
            ).copy()
            length_field = None
        elif feedback.parallel_averaging_length_m is not None:
            length_field = np.full(
                expected_shape,
                float(feedback.parallel_averaging_length_m),
                dtype=np.float64,
            )
            gravity_field = None
            collision_field = None
        else:
            length_field = None
            gravity_field = None
            collision_field = None
        if length_field is not None:
            length_field.setflags(write=False)
        self.parallel_averaging_length_field_m = length_field
        if gravity_field is not None:
            gravity_field.setflags(write=False)
            collision_field.setflags(write=False)
        self.gravity_column_weight_field_m = gravity_field
        self.collision_column_weight_field_m = collision_field
        if self.column_integrated:
            closure_gravity = getattr(
                closure, "gravity_column_weight_m", None
            )
            closure_collision = getattr(
                closure, "collision_column_weight_m", None
            )
            if (
                closure_gravity is None
                or closure_collision is None
                or not np.array_equal(
                    np.asarray(closure_gravity, dtype=np.float64), gravity_field
                )
                or not np.array_equal(
                    np.asarray(closure_collision, dtype=np.float64),
                    collision_field,
                )
            ):
                raise ValueError(
                    "the bidirectional model and Li closure require identical "
                    "gravity and collision column weights"
                )
        right = np.asarray(right_boundary_voltage_v, dtype=np.float64)
        if right.ndim == 0:
            right = np.full(line.mode_count, float(right))
        if right.shape != (line.mode_count,) or not np.all(np.isfinite(right)):
            raise ValueError("right_boundary_voltage_v has incompatible shape")
        if right_boundary_model not in (
            "fixed_voltage",
            "matched",
            "conductance",
        ):
            raise ValueError(
                "right_boundary_model must be 'fixed_voltage', 'matched', "
                "or 'conductance'"
            )
        conductance: float | None = None
        reflection: float | None = None
        if right_boundary_model == "fixed_voltage":
            if right_boundary_conductance_s is not None:
                raise ValueError(
                    "right_boundary_conductance_s is incompatible with a "
                    "fixed-voltage boundary"
                )
        else:
            if np.any(right != 0.0):
                raise ValueError(
                    "passive remote boundaries require zero "
                    "right_boundary_voltage_v"
                )
            if right_boundary_model == "matched":
                if right_boundary_conductance_s is not None:
                    raise ValueError(
                        "a matched boundary determines its conductance from V_A"
                    )
                conductance = line.alfven_conductance_s
                reflection = 0.0
            else:
                if right_boundary_conductance_s is None:
                    raise ValueError(
                        "right_boundary_conductance_s is required for a "
                        "conductance boundary"
                    )
                conductance = float(right_boundary_conductance_s)
                reflection = line.conductance_reflection_coefficient(conductance)
        self.right_boundary_voltage_v = right.copy()
        self.right_boundary_model = right_boundary_model
        self.right_boundary_conductance_s = conductance
        self.right_boundary_reflection_coefficient = reflection

    def _remote_boundary_kwargs(self) -> dict[str, object]:
        """Return mutually exclusive arguments for the line terminal."""

        if self.right_boundary_model == "fixed_voltage":
            return {"right_voltage_v": self.right_boundary_voltage_v}
        return {
            "right_reflection_coefficient": float(
                self.right_boundary_reflection_coefficient
            )
        }

    def _remote_boundary_characteristics(
        self, state: ModalAlfvenState
    ) -> tuple[FloatArray, FloatArray]:
        """Return outgoing/incoming characteristics at the remote face."""

        outgoing = self.line.right_outgoing_characteristic_v(state)
        if self.right_boundary_model == "fixed_voltage":
            incoming = 2.0 * self.right_boundary_voltage_v - outgoing
        else:
            incoming = float(
                self.right_boundary_reflection_coefficient
            ) * outgoing
        return outgoing, incoming

    def _fac_source_from_response(
        self, response: LeftBoundaryResponse
    ) -> FloatArray:
        """Return the configured FAC forcing field.

        ``boundary_gradient`` preserves the original one-sided local model for
        validation and backwards comparisons.  Scalar
        ``symmetric_flux_tube`` retains the legacy volume source
        ``q=N_hemisphere*j/H`` [A m^-3].  A column-integrated model instead
        returns ``J=N_hemisphere*j`` [A m^-2] directly.

        Unlike a one-sided half-cell derivative, its response to boundary
        voltage is positive.  Combined with the negative Li FAC multiplier,
        this produces the coercive passive load ``K + Y`` instead of the
        resolution-dependent ``K - Y`` pole.
        """

        if self.feedback.fac_source_model == "boundary_gradient":
            return response.current_gradient.d_j_parallel_ds_a_m3
        modal_current = self.basis.parallel_current_coefficients_a_m2(
            response.boundary_apar_coefficients_t_m
        )
        if not self.column_integrated:
            # Preserve the historical scalar-H operation order exactly.
            length = float(self.feedback.parallel_averaging_length_m)
            modal_source = (
                float(self.feedback.hemisphere_count) * modal_current / length
            )
            return self.basis.lift(modal_source)
        return (
            float(self.feedback.hemisphere_count)
            * self.basis.lift(modal_current)
        )

    def _closure_forcing(self, values: FloatArray):
        """Wrap a forcing array in the closure's dimensional representation."""

        if self.column_integrated:
            return ParallelSurfaceCurrent(values)
        return ParallelCurrentGradient(values)

    def _fac_source_balance(self, source_a_m3: FloatArray) -> tuple[float, float]:
        """Return conservative current balance in area or column measure."""

        weighted = np.asarray(source_a_m3, dtype=np.float64)
        area = self.config.dx_m * self.config.dy_m
        net = float(np.sum(weighted, dtype=np.float64) * area)
        absolute = float(np.sum(np.abs(weighted), dtype=np.float64) * area)
        return net, abs(net) / max(absolute, np.finfo(float).tiny)

    def _modal_fac_voltage_response_per_v(self) -> FloatArray:
        """Return diagonal FAC responses in A m^-3/V or A m^-2/V."""

        if self.feedback.fac_source_model == "boundary_gradient":
            return (
                -2.0
                * self.basis.eigenvalues_m2
                / (
                    MU0
                    * self.line.alfven_speed_m_s
                    * self.line.spacing_m
                )
            )
        response = (
            float(self.feedback.hemisphere_count)
            * self.basis.eigenvalues_m2
            / (MU0 * self.line.alfven_speed_m_s)
        )
        if self.column_integrated:
            return response
        length = float(self.feedback.parallel_averaging_length_m)
        return response / length

    def _fac_voltage_response_field_per_v(
        self, mode_index: int
    ) -> FloatArray:
        """Return ``d(FAC forcing field)/d Phi_k`` for one retained mode."""

        index = int(mode_index)
        mode = self.basis.mode_field(index)
        if self.feedback.fac_source_model == "boundary_gradient":
            coefficient = (
                -2.0
                * self.basis.eigenvalues_m2[index]
                / (
                    MU0
                    * self.line.alfven_speed_m_s
                    * self.line.spacing_m
                )
            )
            return coefficient * mode
        current_response = (
            float(self.feedback.hemisphere_count)
            * self.basis.eigenvalues_m2[index]
            / (MU0 * self.line.alfven_speed_m_s)
        )
        if self.column_integrated:
            return current_response * mode
        return current_response * mode / self.parallel_averaging_length_field_m

    def initial_alfven_state_from_li_potential(
        self, density_m3: ArrayLike
    ) -> ModalAlfvenState:
        """Return an ``A_parallel=0`` line compatible with both end voltages.

        A uniform Li voltage is incompatible with a different prescribed
        right-end voltage and launches a grid-scale boundary transient on the
        first step.  The cell-centred linear profile below satisfies the two
        endpoint voltages under the same half-cell ghost construction used by
        :class:`ModalAlfvenLine`.  When the endpoints differ the subsequent
        inductive adjustment is physical and smooth, rather than a boundary
        discontinuity hidden in the initial condition.
        """

        density = np.asarray(density_m3, dtype=np.float64)
        potential = self._unforced_coupled_geometry_potential(density)
        voltage = self.basis.project(
            self.closure.background_field_t * potential
        )
        if self.right_boundary_model == "fixed_voltage":
            fraction = (self.line.s_m / self.line.length_m)[:, None]
            potential_line = (
                (1.0 - fraction) * voltage[None, :]
                + fraction * self.right_boundary_voltage_v[None, :]
            )
            return ModalAlfvenState.from_potential_and_apar(
                potential_line,
                np.zeros_like(potential_line),
                alfven_speed_m_s=self.line.alfven_speed_m_s,
            )

        # Full-gain characteristic steady state for a passive termination:
        # w+ = 2 Phi_L - w- at the Li plane and w- = R w+ remotely.
        reflection = float(self.right_boundary_reflection_coefficient)
        denominator = 1.0 + reflection
        if denominator <= 64.0 * np.finfo(float).eps:
            raise ValueError(
                "the selected nearly short-circuit load has no finite DC "
                "line state; use initial_line_state='zero' with a port ramp"
            )
        plus = np.broadcast_to(
            2.0 * voltage / denominator,
            (self.line.cells, self.line.mode_count),
        ).copy()
        minus = reflection * plus
        return ModalAlfvenState(plus, minus)

    def _relative_boundary_residual(
        self, target: FloatArray, iterate: FloatArray
    ) -> float:
        """Measure convergence only in the retained coupled voltage modes."""

        target_voltage = self.basis.project(
            self.closure.background_field_t * target
        )
        iterate_voltage = self.basis.project(
            self.closure.background_field_t * iterate
        )
        numerator = float(np.linalg.norm(target_voltage - iterate_voltage))
        denominator = max(
            float(np.linalg.norm(target_voltage)),
            float(np.linalg.norm(iterate_voltage)),
            np.finfo(np.float64).tiny,
        )
        return numerator / denominator

    def _unforced_coupled_geometry_potential(
        self, density: FloatArray
    ) -> FloatArray:
        """Solve gravity forcing with the coupled model's column geometry."""

        if not getattr(self.closure, "column_integrated", False):
            potential, _ = self.closure.solve(density, forcing=None)
            return potential
        components = self.closure.linear_components(density)
        right = np.asarray(components.gravity_rhs_s_inv, dtype=np.float64).ravel()
        potential, diagnostics = self.closure.solve_algebraic(
            components.matrix, right
        )
        gravity_norm = float(np.linalg.norm(right))
        self.closure.last_fac_diagnostics = FACClosureDiagnostics(
            potential=diagnostics,
            net_current_gradient_a_m=0.0,
            relative_net_current_gradient=0.0,
            fac_rhs_l2_s_inv=0.0,
            gravity_rhs_l2_s_inv=gravity_norm,
            fac_to_gravity_ratio=0.0,
            forcing_representation=(
                "surface_current"
                if self.column_integrated
                else "volume_current_gradient"
            ),
            forcing_units="A m^-2" if self.column_integrated else "A m^-3",
        )
        return potential

    def _fixed_point_coupled_potential(
        self,
        density: FloatArray,
        line_state: ModalAlfvenState,
        effective_gain: float,
    ) -> tuple[FloatArray, LeftBoundaryResponse, int, float]:
        potential = self._unforced_coupled_geometry_potential(density)
        if effective_gain == 0.0:
            response = self.line.left_boundary_response(
                line_state,
                self.basis,
                boundary_voltage_coefficients_v=np.zeros(
                    self.basis.mode_count, dtype=np.float64
                ),
            )
            return potential, response, 0, 0.0

        omega = self.feedback.under_relaxation
        tolerance = self.feedback.relative_tolerance
        residual = float("inf")
        response: LeftBoundaryResponse | None = None
        for iteration in range(1, self.feedback.max_iterations + 1):
            voltage = self.basis.project(
                self.closure.background_field_t * potential
            )
            response = self.line.left_boundary_response(
                line_state,
                self.basis,
                boundary_voltage_coefficients_v=effective_gain * voltage,
            )
            forcing = self._closure_forcing(
                effective_gain
                * self._fac_source_from_response(response)
            )
            target, _ = self.closure.solve(density, forcing=forcing)
            residual = self._relative_boundary_residual(target, potential)
            if residual <= tolerance:
                potential = target
                # Re-evaluate the physical source at the returned potential.
                voltage = self.basis.project(
                    self.closure.background_field_t * potential
                )
                response = self.line.left_boundary_response(
                    line_state,
                    self.basis,
                    boundary_voltage_coefficients_v=effective_gain * voltage,
                )
                checked, _ = self.closure.solve(
                    density,
                    forcing=self._closure_forcing(
                        effective_gain
                        * self._fac_source_from_response(response)
                    ),
                )
                checked_residual = self._relative_boundary_residual(
                    checked, potential
                )
                if checked_residual <= tolerance:
                    # ``response`` and the closure diagnostics now correspond
                    # to this exact returned boundary potential; ``checked``
                    # is used only to certify its fixed-point residual.
                    return potential, response, iteration, checked_residual
                potential = potential + omega * (checked - potential)
                residual = checked_residual
                continue
            potential = potential + omega * (target - potential)

        raise PhysicalCouplingConvergenceError(
            self.feedback.max_iterations, residual
        )

    def _direct_coupled_potential(
        self,
        density: FloatArray,
        line_state: ModalAlfvenState,
        effective_gain: float,
    ) -> tuple[FloatArray, LeftBoundaryResponse, int, float]:
        """Solve the simultaneous boundary relation as a low-rank system.

        For fixed density and field-line state, ``d_s(j_parallel)`` is affine
        in the Li boundary voltage.  Rather than Picard-iterating that affine
        relation, GMRES solves the combined Li matrix plus modal response in
        one stage.  The unforced Li matrix is an exact preconditioner, so the
        Krylov correction lives only in the retained transverse modal space.
        """

        components = self.closure.linear_components(density)
        matrix = components.matrix
        shape = (self.config.ny, self.config.nx)
        size = self.config.ny * self.config.nx
        zero_voltage = np.zeros(self.basis.mode_count, dtype=np.float64)
        free_response = self.line.left_boundary_response(
            line_state,
            self.basis,
            boundary_voltage_coefficients_v=zero_voltage,
        )
        free_gradient = self._fac_source_from_response(free_response)
        multiplier = components.fac_multiplier
        rhs = np.array(
            components.gravity_rhs_s_inv
            + multiplier * effective_gain * free_gradient,
            copy=True,
        ).ravel()
        rhs[0] = 0.0

        def affine_gradient(potential_flat: np.ndarray) -> FloatArray:
            potential = np.asarray(potential_flat, dtype=np.float64).reshape(shape)
            voltage = self.basis.project(
                self.closure.background_field_t * potential
            )
            response = self.line.left_boundary_response(
                line_state,
                self.basis,
                boundary_voltage_coefficients_v=effective_gain * voltage,
            )
            return self._fac_source_from_response(response)

        def matvec(vector: np.ndarray) -> np.ndarray:
            values = np.asarray(vector, dtype=np.float64)
            jacobian_gradient = affine_gradient(values) - free_gradient
            result = matrix @ values - (
                multiplier * effective_gain * jacobian_gradient
            ).ravel()
            result[0] = values[0]
            return np.asarray(result, dtype=np.float64)

        operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
        preconditioner_rebuilt = False

        def precondition(vector: np.ndarray) -> np.ndarray:
            nonlocal preconditioner_rebuilt
            result, rebuilt = self.closure._solve_linear(
                matrix, np.asarray(vector, dtype=np.float64)
            )
            preconditioner_rebuilt = preconditioner_rebuilt or rebuilt
            return result

        preconditioner = LinearOperator(
            (size, size), matvec=precondition, dtype=np.float64
        )
        iterations = 0

        def count_iteration(_: object) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = gmres(
            operator,
            rhs,
            M=preconditioner,
            rtol=self.feedback.relative_tolerance,
            atol=0.0,
            restart=min(size, max(8, 2 * self.basis.mode_count + 2)),
            maxiter=self.feedback.max_iterations,
            callback=count_iteration,
            callback_type="pr_norm",
        )
        solution = np.asarray(solution, dtype=np.float64)
        residual_vector = operator @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        residual = float(np.linalg.norm(residual_vector) / denominator)
        if info != 0 or not np.isfinite(residual) or (
            residual > 5.0 * self.feedback.relative_tolerance
        ):
            raise PhysicalCouplingConvergenceError(iterations, residual)

        potential = solution.reshape(shape)
        potential -= float(np.mean(potential))
        # A constant gauge shift leaves every physical row and every retained
        # (non-constant) FAC mode unchanged.  Recompute the reported residual
        # for the returned gauge and exclude only the imposed gauge row.
        physical_residual = np.asarray(operator @ potential.ravel() - rhs)
        physical_residual[0] = 0.0
        physical_rhs = np.array(rhs, copy=True)
        physical_rhs[0] = 0.0
        physical_scale = max(
            float(np.linalg.norm(physical_rhs)), np.finfo(float).tiny
        )
        residual = float(np.linalg.norm(physical_residual) / physical_scale)
        voltage = self.basis.project(
            self.closure.background_field_t * potential
        )
        response = self.line.left_boundary_response(
            line_state,
            self.basis,
            boundary_voltage_coefficients_v=effective_gain * voltage,
        )
        applied_gradient = (
            effective_gain
            * self._fac_source_from_response(response)
        )
        net, relative_net = self._fac_source_balance(applied_gradient)
        fac_rhs = multiplier * applied_gradient
        gravity_norm = float(
            np.linalg.norm(components.gravity_rhs_s_inv.ravel())
        )
        fac_norm = float(np.linalg.norm(fac_rhs.ravel()))
        ratio = (
            (0.0 if fac_norm == 0.0 else float("inf"))
            if gravity_norm == 0.0
            else fac_norm / gravity_norm
        )
        potential_diagnostics = PotentialDiagnostics(
            iterations=iterations,
            relative_residual=residual,
            rebuilt_preconditioner=preconditioner_rebuilt,
            used_pyamg=False,
        )
        self.closure.last_fac_diagnostics = FACClosureDiagnostics(
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
        return potential, response, iterations, residual

    def _woodbury_coupled_potential(
        self,
        density: FloatArray,
        line_state: ModalAlfvenState,
        effective_gain: float,
    ) -> tuple[FloatArray, LeftBoundaryResponse, int, float]:
        """Solve the physical modal feedback through a dense Schur system.

        For fixed density and line state the FAC response is affine in the
        retained boundary-voltage coefficients.  Writing the coupled matrix as
        ``K - F Q``, the Woodbury identity reduces the solve to one batched
        application of ``K^{-1}`` and an ``n_mode`` dense system.  PARDISO thus
        factors the 2-D Li matrix once per stage instead of once per GMRES
        preconditioner application.
        """

        components = self.closure.linear_components(density)
        matrix = components.matrix
        shape = (self.config.ny, self.config.nx)
        size = self.config.ny * self.config.nx
        mode_count = self.basis.mode_count
        zero_voltage = np.zeros(mode_count, dtype=np.float64)
        free_response = self.line.left_boundary_response(
            line_state,
            self.basis,
            boundary_voltage_coefficients_v=zero_voltage,
        )
        free_gradient = self._fac_source_from_response(free_response)
        multiplier = components.fac_multiplier
        rhs = np.asarray(
            components.gravity_rhs_s_inv
            + multiplier * effective_gain * free_gradient,
            dtype=np.float64,
        ).ravel()
        rhs[0] = 0.0

        feedback_columns = np.empty((size, mode_count), dtype=np.float64)
        response_fields = np.empty(
            (mode_count, self.config.ny, self.config.nx), dtype=np.float64
        )
        diagonal_modal_response = (
            self._modal_fac_voltage_response_per_v()
            if not self.column_integrated
            else None
        )
        # One factor couples the Li voltage into the line boundary and the
        # other couples the returned current into the Li closure.  This c^2
        # load is the algebraic consequence of an energy-compatible port,
        # rather than a duplicated source activation.
        scaled_multiplier = (effective_gain**2 * multiplier).ravel()
        for index in range(mode_count):
            mode_field = self.basis.mode_field(index)
            if diagonal_modal_response is not None:
                response_fields[index] = (
                    diagonal_modal_response[index] * mode_field
                )
                # Preserve the historical scalar-H arithmetic order.
                feedback_columns[:, index] = (
                    scaled_multiplier
                    * diagonal_modal_response[index]
                    * mode_field.ravel()
                )
            else:
                response_fields[index] = (
                    self._fac_voltage_response_field_per_v(index)
                )
                feedback_columns[:, index] = (
                    scaled_multiplier * response_fields[index].ravel()
                )
        feedback_columns[0] = 0.0

        # PARDISO's combined factor-and-solve phase is not reliable for a
        # matrix-valued RHS on all Windows MKL builds.  Factor with the scalar
        # gravity RHS first, then reuse those factors (phase 33) for all modal
        # columns in one batch.
        unforced, rebuilt = self.closure._solve_linear(matrix, rhs)
        inverse_feedback, _ = self.closure._solve_linear(
            matrix, feedback_columns
        )
        unforced = np.asarray(unforced, dtype=np.float64)
        inverse_feedback = np.asarray(inverse_feedback, dtype=np.float64)
        if inverse_feedback.ndim == 1 and mode_count == 1:
            inverse_feedback = inverse_feedback[:, None]
        if unforced.shape != rhs.shape or inverse_feedback.shape != feedback_columns.shape:
            raise RuntimeError("batched Li solve returned an incompatible shape")
        block_rhs = np.column_stack((rhs, feedback_columns))
        block_solution = np.column_stack((unforced, inverse_feedback))
        block_residual = matrix @ block_solution - block_rhs
        block_scale = max(
            float(np.linalg.norm(block_rhs)), np.finfo(float).tiny
        )
        relative_block_residual = float(
            np.linalg.norm(block_residual) / block_scale
        )
        if (
            not np.isfinite(relative_block_residual)
            or relative_block_residual
            > 5.0 * self.feedback.relative_tolerance
        ):
            raise RuntimeError(
                "batched Li inverse residual "
                f"{relative_block_residual:.3e} exceeds tolerance"
            )
        background_field = self.closure.background_field_t
        projected_unforced = background_field * self.basis.project(
            unforced.reshape(shape)
        )
        projected_feedback = np.empty(
            (mode_count, mode_count), dtype=np.float64
        )
        for index in range(mode_count):
            projected_feedback[:, index] = (
                background_field
                * self.basis.project(inverse_feedback[:, index].reshape(shape))
            )
        schur = np.eye(mode_count) - projected_feedback
        modal_correction = np.linalg.solve(schur, projected_unforced)
        solution = unforced + inverse_feedback @ modal_correction

        def coupled_matvec(values: np.ndarray) -> np.ndarray:
            potential_field = np.asarray(values, dtype=np.float64).reshape(shape)
            voltage = (
                effective_gain
                * background_field
                * self.basis.project(potential_field)
            )
            jacobian_gradient = np.einsum(
                "m,myx->yx", voltage, response_fields, optimize=True
            )
            result = matrix @ np.asarray(values, dtype=np.float64) - (
                effective_gain * multiplier * jacobian_gradient
            ).ravel()
            result[0] = np.asarray(values, dtype=np.float64)[0]
            return np.asarray(result, dtype=np.float64)

        algebraic_residual = coupled_matvec(solution) - rhs
        scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        solver_residual = float(np.linalg.norm(algebraic_residual) / scale)
        if not np.all(np.isfinite(solution)) or not np.isfinite(solver_residual):
            raise PhysicalCouplingConvergenceError(1, solver_residual)
        if solver_residual > 5.0 * self.feedback.relative_tolerance:
            raise PhysicalCouplingConvergenceError(1, solver_residual)

        potential = solution.reshape(shape)
        potential -= float(np.mean(potential))
        physical_residual = coupled_matvec(potential.ravel()) - rhs
        physical_residual[0] = 0.0
        physical_rhs = np.array(rhs, copy=True)
        physical_rhs[0] = 0.0
        physical_scale = max(
            float(np.linalg.norm(physical_rhs)), np.finfo(float).tiny
        )
        residual = float(np.linalg.norm(physical_residual) / physical_scale)
        voltage = self.basis.project(background_field * potential)
        response = self.line.left_boundary_response(
            line_state,
            self.basis,
            boundary_voltage_coefficients_v=effective_gain * voltage,
        )
        applied_gradient = (
            effective_gain
            * self._fac_source_from_response(response)
        )
        net, relative_net = self._fac_source_balance(applied_gradient)
        fac_rhs = multiplier * applied_gradient
        gravity_norm = float(
            np.linalg.norm(components.gravity_rhs_s_inv.ravel())
        )
        fac_norm = float(np.linalg.norm(fac_rhs.ravel()))
        ratio = (
            (0.0 if fac_norm == 0.0 else float("inf"))
            if gravity_norm == 0.0
            else fac_norm / gravity_norm
        )
        self.closure.last_fac_diagnostics = FACClosureDiagnostics(
            potential=PotentialDiagnostics(
                iterations=1,
                relative_residual=residual,
                rebuilt_preconditioner=bool(rebuilt),
                used_pyamg=False,
            ),
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
        return potential, response, 1, residual

    def _coupled_potential(
        self,
        density: FloatArray,
        line_state: ModalAlfvenState,
        effective_gain: float,
    ) -> tuple[FloatArray, LeftBoundaryResponse, int, float]:
        if effective_gain == 0.0:
            potential = self._unforced_coupled_geometry_potential(density)
            response = self.line.left_boundary_response(
                line_state,
                self.basis,
                boundary_voltage_coefficients_v=np.zeros(
                    self.basis.mode_count, dtype=np.float64
                ),
            )
            return potential, response, 0, 0.0
        if self.feedback.solver == "fixed_point":
            return self._fixed_point_coupled_potential(
                density, line_state, effective_gain
            )
        if self.feedback.solver == "modal_woodbury":
            return self._woodbury_coupled_potential(
                density, line_state, effective_gain
            )
        return self._direct_coupled_potential(
            density, line_state, effective_gain
        )

    def _transport_velocity(self, potential: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Add the same imposed +y drift to all density tracks, in m/s."""
        vx, vy = self.legacy_li_model.velocity_from_potential(potential)
        if self.feedback.background_exb_y_m_s != 0.0:
            vy = vy + self.feedback.background_exb_y_m_s
        return vx, vy

    def _density_advection_rhs(
        self,
        density_m3: FloatArray,
        velocity_x_m_s: FloatArray,
        velocity_y_m_s: FloatArray,
    ) -> FloatArray:
        """Advect either local density or the conserved flux-tube column."""

        transported = np.asarray(density_m3, dtype=np.float64)
        divisor: FloatArray | float = 1.0
        if self.column_integrated:
            transported = self.gravity_column_weight_field_m * transported
            divisor = self.gravity_column_weight_field_m
        return advection_rhs(
            transported,
            velocity_x_m_s,
            velocity_y_m_s,
            dx=self.config.dx_m,
            dy=self.config.dy_m,
            scheme=self.config.scheme,
            pdm_alpha=self.config.pdm_alpha,
            periodic_y=False,
        ) / divisor

    def evaluate_column_es_rhs(
        self, density_m3: ArrayLike
    ) -> tuple[FloatArray, RhsDiagnostics]:
        """Evaluate the zero-FAC column-ES control using the same geometry.

        This control isolates flux-tube geometry (two column coefficients and
        column-density continuity) from electromagnetic feedback.  The normal
        ``gain=0`` model remains the literal historical local-ES regression.
        """

        if not self.column_integrated:
            raise ValueError("column-ES control requires column-integrated geometry")
        density = np.asarray(density_m3, dtype=np.float64)
        expected = (self.config.ny, self.config.nx)
        if density.shape != expected or not np.all(np.isfinite(density)):
            raise ValueError(f"density_m3 must be finite with shape {expected}")
        potential = self._unforced_coupled_geometry_potential(density)
        velocity_x, velocity_y = self._transport_velocity(potential)
        density_rhs = self._density_advection_rhs(
            density, velocity_x, velocity_y
        )
        cfl_rate = float(
            np.max(
                np.abs(velocity_x) / self.config.dx_m
                + np.abs(velocity_y) / self.config.dy_m
            )
        )
        potential_diagnostics = self.closure.last_fac_diagnostics.potential
        return density_rhs, RhsDiagnostics(
            cfl_rate=cfl_rate,
            maximum_speed_x=float(np.max(np.abs(velocity_x))),
            maximum_speed_y=float(np.max(np.abs(velocity_y))),
            potential=potential_diagnostics,
        )

    def evaluate_rhs(
        self, time_s: float, state: DimensionalEMState
    ) -> tuple[DimensionalEMState, PhysicalCouplingDiagnostics]:
        # The exact ES-limit contract bypasses every new EM operation.
        if self.feedback.gain == 0.0:
            if self.feedback.background_exb_y_m_s == 0.0:
                density_rhs, diagnostics = self.legacy_li_model.evaluate_rhs(
                    state.density_m3
                )
            else:
                potential, potential_diag = self.legacy_li_model.potential_solver.solve(
                    state.density_m3
                )
                vx, vy = self._transport_velocity(potential)
                density_rhs = self._density_advection_rhs(state.density_m3, vx, vy)
                diagnostics = RhsDiagnostics(
                    cfl_rate=float(np.max(np.abs(vx)/self.config.dx_m + np.abs(vy)/self.config.dy_m)),
                    maximum_speed_x=float(np.max(np.abs(vx))),
                    maximum_speed_y=float(np.max(np.abs(vy))),
                    potential=potential_diag,
                )
            tendency = DimensionalEMState(
                density_rhs,
                ModalAlfvenState(
                    np.zeros_like(state.alfven.w_plus_v),
                    np.zeros_like(state.alfven.w_minus_v),
                ),
            )
            return tendency, PhysicalCouplingDiagnostics(
                electrostatic=diagnostics,
                fac_closure=None,
                effective_gain=0.0,
                activation_fraction=0.0,
                iterations=0,
                relative_coupling_residual=0.0,
                alfven_cfl_rate_s_inv=0.0,
                combined_cfl_rate_s_inv=diagnostics.cfl_rate,
                maximum_abs_d_j_parallel_ds_a_m3=0.0,
                net_d_j_parallel_ds_a_m=0.0,
                boundary_characteristic_defect_v=0.0,
                boundary_power_modal_w=0.0,
                boundary_power_grid_w=0.0,
                remote_boundary_power_modal_w=0.0,
                remote_boundary_reflection_coefficient=(
                    -1.0
                    if self.right_boundary_model == "fixed_voltage"
                    else float(self.right_boundary_reflection_coefficient)
                ),
                relative_power_pairing_defect=0.0,
                fac_forcing_representation="none",
                fac_forcing_units="none",
                maximum_abs_fac_forcing=0.0,
                net_fac_forcing_integral=0.0,
                density_conservation_measure="local_density",
            )

        state = state.validated((self.config.ny, self.config.nx), self.line)
        effective_gain = self.feedback.effective_gain(time_s)
        activation = effective_gain / self.feedback.gain
        potential, response, iterations, residual = self._coupled_potential(
            state.density_m3, state.alfven, effective_gain
        )
        voltage = self.basis.project(
            self.closure.background_field_t * potential
        )
        velocity_x, velocity_y = self._transport_velocity(potential)
        density_rhs = self._density_advection_rhs(
            state.density_m3, velocity_x, velocity_y
        )
        density_cfl = float(
            np.max(
                np.abs(velocity_x) / self.config.dx_m
                + np.abs(velocity_y) / self.config.dy_m
            )
        )
        line_voltage = effective_gain * voltage
        line_rhs = self.line.rhs(
            state.alfven,
            left_voltage_v=line_voltage,
            **self._remote_boundary_kwargs(),
        )
        line_cfl = self.line.alfven_speed_m_s / self.line.spacing_m

        current_coefficients = self.basis.parallel_current_coefficients_a_m2(
            response.boundary_apar_coefficients_t_m
        )
        current_field = self.basis.lift(current_coefficients)
        voltage_field = self.basis.lift(line_voltage)
        modal_power = float(self.feedback.hemisphere_count) * (
            self.basis.modal_inner_product(line_voltage, current_coefficients)
        )
        gradient = effective_gain * self._fac_source_from_response(response)
        if self.feedback.fac_source_model == "symmetric_flux_tube":
            unscaled_voltage_field = self.basis.lift(voltage)
            if self.column_integrated:
                # The closure receives J_surface directly, so its conjugate
                # power is integral(Phi J_surface dA).
                grid_power = self.basis.field_inner_product(
                    unscaled_voltage_field, gradient
                )
            else:
                # Legacy scalar-H q=N*j/H path.
                grid_power = self.basis.field_inner_product(
                    unscaled_voltage_field,
                    self.parallel_averaging_length_field_m * gradient,
                )
        else:
            grid_power = float(self.feedback.hemisphere_count) * (
                self.basis.field_inner_product(voltage_field, current_field)
            )
        remote_outgoing, remote_incoming = self._remote_boundary_characteristics(
            state.alfven
        )
        remote_voltage = 0.5 * (remote_outgoing + remote_incoming)
        remote_apar = 0.5 * (
            remote_outgoing - remote_incoming
        ) / self.line.alfven_speed_m_s
        remote_current = self.basis.parallel_current_coefficients_a_m2(
            remote_apar
        )
        remote_power = float(self.feedback.hemisphere_count) * (
            self.basis.modal_inner_product(remote_voltage, remote_current)
        )
        power_scale = max(abs(modal_power), abs(grid_power), np.finfo(float).tiny)
        power_defect = abs(modal_power - grid_power) / power_scale
        characteristic_defect = float(
            np.max(
                np.abs(
                    response.incoming_w_plus_v
                    + response.outgoing_w_minus_v
                    - 2.0 * line_voltage
                )
            )
        )
        net_gradient, _ = self._fac_source_balance(gradient)
        potential_diagnostics = (
            self.closure.last_fac_diagnostics.potential
            if self.closure.last_fac_diagnostics is not None
            else self.closure.solve(state.density_m3, None)[1]
        )
        electrostatic = RhsDiagnostics(
            cfl_rate=density_cfl,
            maximum_speed_x=float(np.max(np.abs(velocity_x))),
            maximum_speed_y=float(np.max(np.abs(velocity_y))),
            potential=potential_diagnostics,
        )
        diagnostics = PhysicalCouplingDiagnostics(
            electrostatic=electrostatic,
            fac_closure=self.closure.last_fac_diagnostics,
            effective_gain=effective_gain,
            activation_fraction=activation,
            iterations=iterations,
            relative_coupling_residual=residual,
            alfven_cfl_rate_s_inv=line_cfl,
            combined_cfl_rate_s_inv=max(density_cfl, line_cfl),
            maximum_abs_d_j_parallel_ds_a_m3=float(np.max(np.abs(gradient))),
            net_d_j_parallel_ds_a_m=net_gradient,
            boundary_characteristic_defect_v=characteristic_defect,
            boundary_power_modal_w=modal_power,
            boundary_power_grid_w=grid_power,
            remote_boundary_power_modal_w=remote_power,
            remote_boundary_reflection_coefficient=(
                -1.0
                if self.right_boundary_model == "fixed_voltage"
                else float(self.right_boundary_reflection_coefficient)
            ),
            relative_power_pairing_defect=power_defect,
            fac_forcing_representation=(
                "surface_current"
                if self.column_integrated
                else "volume_current_gradient"
            ),
            fac_forcing_units="A m^-2" if self.column_integrated else "A m^-3",
            maximum_abs_fac_forcing=float(np.max(np.abs(gradient))),
            net_fac_forcing_integral=net_gradient,
            density_conservation_measure=(
                "gravity_column_weight_times_density"
                if self.column_integrated
                else "local_density"
            ),
        )
        return DimensionalEMState(density_rhs, line_rhs), diagnostics

    def ssprk3_step(
        self, time_s: float, state: DimensionalEMState, dt_s: float
    ) -> tuple[DimensionalEMState, PhysicalCouplingDiagnostics]:
        """Advance one coupled SSPRK3 step, rejecting either CFL violation."""

        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        # Preserve the historical step only for zero imposed drift.
        if self.feedback.gain == 0.0 and self.feedback.background_exb_y_m_s == 0.0:
            density, legacy_diag = self.legacy_li_model.ssprk3_step(
                state.density_m3, dt
            )
            result = DimensionalEMState(density, state.alfven)
            diagnostic = PhysicalCouplingDiagnostics(
                electrostatic=legacy_diag,
                fac_closure=None,
                effective_gain=0.0,
                activation_fraction=0.0,
                iterations=0,
                relative_coupling_residual=0.0,
                alfven_cfl_rate_s_inv=0.0,
                combined_cfl_rate_s_inv=legacy_diag.cfl_rate,
                maximum_abs_d_j_parallel_ds_a_m3=0.0,
                net_d_j_parallel_ds_a_m=0.0,
                boundary_characteristic_defect_v=0.0,
                boundary_power_modal_w=0.0,
                boundary_power_grid_w=0.0,
                remote_boundary_power_modal_w=0.0,
                remote_boundary_reflection_coefficient=(
                    -1.0
                    if self.right_boundary_model == "fixed_voltage"
                    else float(self.right_boundary_reflection_coefficient)
                ),
                relative_power_pairing_defect=0.0,
                fac_forcing_representation="none",
                fac_forcing_units="none",
                maximum_abs_fac_forcing=0.0,
                net_fac_forcing_integral=0.0,
                density_conservation_measure="local_density",
            )
            return result, diagnostic

        k0, d0 = self.evaluate_rhs(time_s, state)
        self._check_cfl(d0, dt)
        u1 = _state_linear_combination((1.0, dt), (state, k0))
        k1, d1 = self.evaluate_rhs(time_s + dt, u1)
        self._check_cfl(d1, dt)
        euler2 = _state_linear_combination((1.0, dt), (u1, k1))
        u2 = _state_linear_combination((0.75, 0.25), (state, euler2))
        k2, d2 = self.evaluate_rhs(time_s + 0.5 * dt, u2)
        self._check_cfl(d2, dt)
        euler3 = _state_linear_combination((1.0, dt), (u2, k2))
        result = _state_linear_combination((1.0 / 3.0, 2.0 / 3.0), (state, euler3))
        diagnostic = max(
            (d0, d1, d2), key=lambda item: item.combined_cfl_rate_s_inv
        )
        return result, diagnostic

    def _check_cfl(self, diagnostics: PhysicalCouplingDiagnostics, dt_s: float) -> None:
        density_cfl = diagnostics.electrostatic.cfl_rate * dt_s
        if density_cfl > self.config.cfl_limit:
            raise CFLViolation(
                f"density CFL={density_cfl:.6g} exceeds configured limit "
                f"{self.config.cfl_limit:.6g}"
            )
        line_cfl = diagnostics.alfven_cfl_rate_s_inv * dt_s
        if line_cfl > self.feedback.line_cfl_limit:
            raise CFLViolation(
                f"Alfven-line CFL={line_cfl:.6g} exceeds configured limit "
                f"{self.feedback.line_cfl_limit:.6g}"
            )


def density_mode_amplitude(density_m3: ArrayLike) -> float:
    """Return the L2 amplitude of horizontal density structure."""

    density = np.asarray(density_m3, dtype=np.float64)
    if density.ndim != 2 or not np.all(np.isfinite(density)):
        raise ValueError("density_m3 must be a finite two-dimensional array")
    perturbation = density - np.mean(density, axis=1, keepdims=True)
    return float(np.sqrt(np.mean(perturbation * perturbation)))


__all__ = [
    "DimensionalBidirectionalModel",
    "DimensionalEMState",
    "PhysicalCouplingConvergenceError",
    "PhysicalCouplingDiagnostics",
    "PhysicalFeedbackConfig",
    "density_mode_amplitude",
]
