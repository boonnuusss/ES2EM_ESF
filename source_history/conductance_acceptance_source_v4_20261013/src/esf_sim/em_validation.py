"""Fast, staged validation for the reduced electromagnetic scaffold.

The checks in this module are intentionally small.  They exercise analytic
relations or tiny synthetic grids and never read the formal Li snapshots,
invoke PARDISO, or run a long EPB simulation.  Stage 8 retains the bounded
feedback-controller test; stage 9 verifies the physical dimensional
``A_parallel -> j_parallel -> d_s j_parallel -> Li`` path on a tiny nonlinear
grid.  That final check is reduced-EM verification, not a production-length
Figure-3 calculation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .alfven_1d import (
    MU0,
    AlfvenState,
    ConductanceLoad,
    FieldLineGrid,
    PeriodicMidpointAlfvenSolver,
    UpwindAlfvenSolver,
    alfven_energy,
    characteristic_amplitudes,
    right_going_state,
)
from .config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
    SimulationConfig,
)
from .controlled_em import (
    BoundaryFixedPointSolver,
    FeedbackConfig,
    area_weighted_mean,
    condition_fac_source,
)
from .frozen_li_em import (
    FrozenLiDrive,
    LiSnapshot,
    MagneticNormalization,
    OneWayAlfvenExperiment,
    SmoothRamp,
)
from .laundal_endpoint import (
    DynamicEndpointParameters,
    InstantaneousLaundalEndpoint,
    finite_alfven_to_instantaneous_slope_ratio,
)
from .model import ESFModel, initial_density
from .potential import PotentialDiagnostics
from .reduced_em import CoupledEMState, ESLimitAdapter
from .bidirectional_em import (
    DimensionalBidirectionalModel,
    DimensionalEMState,
    PhysicalFeedbackConfig,
    density_mode_amplitude,
)
from .fac_closure import DimensionalFACMimeticClosure
from .modal_alfven import (
    ModalAlfvenLine,
    ModalAlfvenState,
    ModalBasis,
    TransverseMode,
)


_B0_T = 5.0e-5
_VA_M_S = 1.0e6
_LENGTH_M = 1.0e6
_RHO_KG_M3 = _B0_T**2 / (MU0 * _VA_M_S**2)


def _grid(cells: int) -> FieldLineGrid:
    return FieldLineGrid.uniform(
        length_m=_LENGTH_M,
        cells=cells,
        background_field_t=_B0_T,
        mass_density_kg_m3=_RHO_KG_M3,
    )


def _record(
    level: int,
    name: str,
    metric: float,
    threshold: float,
    *,
    criterion: str = "<=",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = float(metric)
    limit = float(threshold)
    if criterion == "<=":
        passed = bool(np.isfinite(value) and value <= limit)
    elif criterion == ">=":
        passed = bool(np.isfinite(value) and value >= limit)
    else:  # pragma: no cover - private callers use one of the two forms above
        raise ValueError(f"unsupported criterion {criterion!r}")
    return {
        "level": int(level),
        "name": name,
        "status": "pass" if passed else "fail",
        "metric": value,
        "threshold": limit,
        "criterion": criterion,
        "details": details or {},
    }


def _wave_error(cells: int) -> float:
    grid = _grid(cells)
    solver = PeriodicMidpointAlfvenSolver(grid)
    wavenumber = 4.0 * np.pi / _LENGTH_M
    initial_velocity = np.sin(wavenumber * grid.s_m)
    initial = right_going_state(initial_velocity, grid)
    duration_s = 0.23 * _LENGTH_M / _VA_M_S
    dt_s = 0.35 * grid.spacing_m / _VA_M_S
    result = solver.advance(initial, duration_s=duration_s, dt_s=dt_s)
    exact = np.sin(wavenumber * (grid.s_m - _VA_M_S * duration_s))
    return float(np.sqrt(np.mean((result.velocity_m_s - exact) ** 2)))


def _stage_1() -> dict[str, Any]:
    coarse_error = _wave_error(32)
    fine_error = _wave_error(64)
    observed_order = float(np.log2(coarse_error / fine_error))
    grid = _grid(32)
    velocity = np.sin(2.0 * np.pi * grid.s_m / grid.length_m)
    state = right_going_state(velocity, grid)
    right, left = characteristic_amplitudes(state, grid)
    walen_defect = float(np.linalg.norm(left) / np.linalg.norm(right))
    order_threshold = 1.75
    walen_threshold = 1.0e-13
    worst_normalized_defect = max(
        order_threshold / observed_order,
        walen_defect / walen_threshold,
    )
    return _record(
        1,
        "uniform_va_wave_accuracy_and_walen_relation",
        worst_normalized_defect,
        1.0,
        details={
            "coarse_l2_error": coarse_error,
            "fine_l2_error": fine_error,
            "observed_spatial_order": observed_order,
            "spatial_order_threshold": order_threshold,
            "walen_defect": walen_defect,
            "walen_threshold": walen_threshold,
            "formal_snapshot_used": False,
        },
    )


def _stage_2() -> dict[str, Any]:
    grid = _grid(16)
    sigma_a = float(grid.alfven_conductance_s[-1])
    sigma_p = 3.0 * sigma_a
    reflection = ConductanceLoad(sigma_p).reflection_coefficient(sigma_a)
    error = abs(reflection - (-0.5))
    return _record(
        2,
        "conductance_load_analytic_reflection",
        error,
        1.0e-14,
        details={
            "alfven_conductance_s": sigma_a,
            "pedersen_conductance_s": sigma_p,
            "measured_reflection": reflection,
            "analytic_reflection": -0.5,
        },
    )


def _stage_3() -> dict[str, Any]:
    cells = 160
    base = _grid(cells)
    density = np.full(cells, _RHO_KG_M3)
    density[base.s_m >= 0.60 * _LENGTH_M] = 4.0 * _RHO_KG_M3
    grid = FieldLineGrid(
        base.s_m,
        np.full(cells, _B0_T),
        density,
    )
    solver = UpwindAlfvenSolver(grid, periodic=False)
    velocity = np.exp(
        -0.5 * ((grid.s_m - 0.30 * _LENGTH_M) / (0.04 * _LENGTH_M)) ** 2
    )
    initial = right_going_state(velocity, grid)
    incident_right, _ = characteristic_amplitudes(initial, grid)
    result = solver.advance(
        initial, duration_s=0.72 * _LENGTH_M / _VA_M_S, cfl=0.8
    )
    _, reflected_left = characteristic_amplitudes(result, grid)
    measured = float(
        reflected_left[np.argmax(np.abs(reflected_left))]
        / np.max(incident_right)
    )
    impedance_1 = np.sqrt(MU0 * _RHO_KG_M3)
    impedance_2 = np.sqrt(MU0 * 4.0 * _RHO_KG_M3)
    analytic = float(
        (impedance_1 - impedance_2) / (impedance_1 + impedance_2)
    )
    error = abs(measured - analytic)
    return _record(
        3,
        "nonuniform_va_impedance_jump_propagation",
        error,
        3.0e-2,
        details={
            "cells": cells,
            "va_left_m_s": float(grid.alfven_speed_m_s[0]),
            "va_right_m_s": float(grid.alfven_speed_m_s[-1]),
            "measured_reflection": measured,
            "analytic_reflection": analytic,
        },
    )


def _stage_4() -> dict[str, Any]:
    grid = _grid(48)
    phase = 2.0 * np.pi * grid.s_m / grid.length_m
    initial = AlfvenState(
        0.3 * np.sin(phase) + 0.1 * np.cos(3.0 * phase),
        1.0e-9 * np.cos(phase) - 0.4e-9 * np.sin(2.0 * phase),
    )
    solver = PeriodicMidpointAlfvenSolver(grid)
    before = alfven_energy(initial, grid).total_j_m2
    result = solver.advance(
        initial,
        duration_s=0.25 * grid.length_m / _VA_M_S,
        dt_s=0.4 * grid.spacing_m / _VA_M_S,
    )
    after = alfven_energy(result, grid).total_j_m2
    energy_drift = abs(after / before - 1.0)

    sigma_a = float(grid.alfven_conductance_s[-1])
    sigma_p = 0.25 * sigma_a
    reflection = ConductanceLoad(sigma_p).reflection_coefficient(sigma_a)
    incident_e_v_m = 0.08
    incident = sigma_a * incident_e_v_m**2
    reflected = sigma_a * (reflection * incident_e_v_m) ** 2
    load_e = (1.0 + reflection) * incident_e_v_m
    joule = sigma_p * load_e**2
    power_defect = abs(incident - reflected - joule) / incident
    energy_threshold = 5.0e-12
    power_threshold = 1.0e-13
    metric = max(
        energy_drift / energy_threshold,
        power_defect / power_threshold,
    )
    return _record(
        4,
        "lossless_energy_and_poynting_joule_budget",
        metric,
        1.0,
        details={
            "initial_energy_j_m2": before,
            "final_energy_j_m2": after,
            "relative_energy_drift": energy_drift,
            "energy_drift_threshold": energy_threshold,
            "relative_power_defect": power_defect,
            "power_defect_threshold": power_threshold,
        },
    )


def _stage_5() -> dict[str, Any]:
    oxygen_mass_kg = 2.66e-26
    sigma_p = 4.0
    sigma_h = 2.5
    parameters = DynamicEndpointParameters.from_conductances(
        pedersen_conductance_s=sigma_p,
        hall_conductance_s=sigma_h,
        ion_mass_kg=oxygen_mass_kg,
        background_field_t=_B0_T,
    )
    endpoint = InstantaneousLaundalEndpoint(parameters)
    electric = np.array([0.08, 0.0])
    actual = endpoint.steady_current_a_m(electric)
    expected = np.array([sigma_p * electric[0], sigma_h * electric[0]])
    error = float(
        np.linalg.norm(actual - expected)
        / max(np.linalg.norm(expected), np.finfo(np.float64).tiny)
    )
    angular_frequency_s = 2.0 * np.pi / 300.0
    speeds = np.array([2.0e6, 4.0e6, 8.0e6])
    slope_errors = np.array(
        [
            abs(
                finite_alfven_to_instantaneous_slope_ratio(
                    angular_frequency_s=angular_frequency_s,
                    length_m=500.0e3,
                    alfven_speed_m_s=float(speed),
                )
                - 1.0
            )
            for speed in speeds
        ]
    )
    convergence_orders = np.log2(slope_errors[:-1] / slope_errors[1:])
    minimum_order = float(np.min(convergence_orders))
    finest_error = float(slope_errors[-1])
    order_threshold = 1.99
    finest_threshold = 3.0e-7
    conductance_threshold = 2.0e-13
    worst_normalized_defect = max(
        order_threshold / minimum_order,
        finest_error / finest_threshold,
        error / conductance_threshold,
    )
    return _record(
        5,
        "finite_line_va_infinity_to_laundal_dynamic_endpoint",
        worst_normalized_defect,
        1.0,
        details={
            "pedersen_conductance_s": parameters.pedersen_conductance_s,
            "hall_conductance_s": parameters.hall_conductance_s,
            "conductance_relative_error": error,
            "conductance_error_threshold": conductance_threshold,
            "slope_ratio_errors": slope_errors.tolist(),
            "minimum_quadratic_order": minimum_order,
            "order_threshold": order_threshold,
            "finest_slope_ratio_error": finest_error,
            "finest_error_threshold": finest_threshold,
        },
    )


class _FixedPotentialSolver:
    def __init__(self, potential: np.ndarray):
        self.potential = np.array(potential, dtype=np.float64, copy=True)

    def solve(self, density: np.ndarray):
        if density.shape != self.potential.shape:
            raise ValueError("density and potential shapes differ")
        return self.potential.copy(), PotentialDiagnostics(0, 0.0, False, False)


def _tiny_li_config() -> SimulationConfig:
    return SimulationConfig(
        nx=9,
        ny=9,
        x_min_km=0.0,
        x_max_km=8.0,
        y_min_km=250.0,
        y_max_km=258.0,
        dx_km=1.0,
        dy_km=1.0,
        perturbation_wavelength_km=8.0,
        dt_s=0.01,
        scheme="pdm7",
        pdm_alpha=0.4,
    )


def _stage_6() -> dict[str, Any]:
    config = _tiny_li_config()
    density = initial_density(config)
    yy, xx = np.meshgrid(np.arange(config.ny), np.arange(config.nx), indexing="ij")
    potential = 500.0 * (np.sin(2.0 * np.pi * xx / config.nx) + 0.1 * yy)

    legacy = ESFModel(config)
    legacy.potential_solver = _FixedPotentialSolver(potential)
    expected, _ = legacy.ssprk3_step(density, config.dt_s)

    adapted = ESFModel(config)
    adapted.potential_solver = _FixedPotentialSolver(potential)
    state = CoupledEMState(
        density.copy(), AlfvenState(np.zeros(7), np.zeros(7))
    )
    actual, _ = ESLimitAdapter(adapted).ssprk3_step(0.0, state, config.dt_s)
    density_scale = max(float(np.max(np.abs(expected))), 1.0)
    density_defect = float(np.max(np.abs(actual.density_m3 - expected)) / density_scale)
    em_defect = max(
        float(np.max(np.abs(actual.alfven.velocity_m_s))),
        float(np.max(np.abs(actual.alfven.magnetic_t))),
    )
    return _record(
        6,
        "li_electrostatic_limit_adapter",
        max(density_defect, em_defect),
        0.0,
        details={
            "grid": [config.ny, config.nx],
            "pardiso_used": False,
            "density_relative_defect": density_defect,
            "electromagnetic_tendency_defect": em_defect,
        },
    )


class _CountingClosure:
    def __init__(self, potential: np.ndarray):
        self.potential = np.array(potential, dtype=np.float64, copy=True)
        self.calls = 0

    def solve(self, density: np.ndarray):
        self.calls += 1
        return self.potential.copy(), {"calls": self.calls}


class _ProbeOneWayLine:
    def __init__(self):
        self.calls = 0

    def rhs(self, state: object, drive):
        self.calls += 1
        return state, float(np.linalg.norm(drive.electric_potential_v))


def _stage_7() -> dict[str, Any]:
    density = np.arange(20.0).reshape(4, 5) + 10.0
    snapshot = LiSnapshot(density, time_s=1200.0)
    stored = snapshot.density_m3.copy()
    psi = np.arange(20.0).reshape(4, 5) * 1.0e5
    closure = _CountingClosure(psi)
    field_t = 4.0e-5
    drive = FrozenLiDrive(
        snapshot,
        closure,
        normalization=MagneticNormalization(field_t),
        project_potential=lambda field: np.mean(field, axis=0),
        ramp=SmoothRamp(start_s=0.0, rise_time_s=2.0),
    )
    line = _ProbeOneWayLine()
    experiment = OneWayAlfvenExperiment(line, drive)
    _, response = experiment.rhs(2.0, object())
    projected = np.mean(psi, axis=0)
    expected_response = float(np.linalg.norm(field_t * projected))
    snapshot_defect = float(np.max(np.abs(snapshot.density_m3 - stored)))
    response_defect = abs(response - expected_response) / max(expected_response, 1.0)
    call_defect = float(abs(closure.calls - 1) + abs(line.calls - 1))
    return _record(
        7,
        "frozen_li_snapshot_one_way_alfven_drive",
        max(snapshot_defect, response_defect, call_defect),
        0.0,
        details={
            "closure_calls": closure.calls,
            "line_calls": line.calls,
            "snapshot_mutation": snapshot_defect,
            "feedback_path_present": False,
        },
    )


def _stage_8() -> dict[str, Any]:
    config = FeedbackConfig(
        gain=0.2,
        under_relaxation=0.7,
        max_iterations=200,
        relative_tolerance=1.0e-13,
        max_source_ratio=0.1,
    )
    phi0 = np.array([1.0, -2.0, 0.5])
    j_free = np.array([0.2, 0.3, -0.1])
    coefficient = np.array([0.25, 0.1, -0.2])
    admittance = np.array([0.4, -0.25, 0.3])
    result = BoundaryFixedPointSolver(config).solve(
        phi0, j_free, coefficient, admittance
    )
    expected = (phi0 + coefficient * j_free) / (1.0 - coefficient * admittance)
    fixed_point_error = float(
        np.linalg.norm(result.potential - expected) / np.linalg.norm(expected)
    )

    fac = np.array([2.0, -1.0, 4.0, 0.5])
    area = np.array([1.0, 2.0, 1.0, 3.0])
    conditioned = condition_fac_source(
        fac,
        np.full_like(fac, 100.0),
        area,
        config,
        time_s=1.0,
    )
    zero_net_defect = abs(area_weighted_mean(conditioned.source, area))
    metric = max(fixed_point_error, result.relative_residual, zero_net_defect)
    return _record(
        8,
        "controlled_two_way_feedback_toy_closure",
        metric,
        1.0e-10,
        details={
            "scope": "control_layer_and_toy_closure_only",
            "physical_dimensional_fac_to_li_closure_complete": False,
            "formal_nonlinear_epb_two_way_run_complete": False,
            "fixed_point_iterations": result.iterations,
            "fixed_point_relative_error": fixed_point_error,
            "zero_net_fac_defect": zero_net_defect,
        },
    )


def _stage_9() -> dict[str, Any]:
    """Exercise the physical dimensional FAC path on a tiny nonlinear grid."""

    config = SimulationConfig(
        nx=9,
        ny=9,
        x_min_km=0.0,
        x_max_km=8.0,
        y_min_km=250.0,
        y_max_km=258.0,
        dx_km=1.0,
        dy_km=1.0,
        perturbation_wavelength_km=9.0,
        scheme="pdm7",
        pdm_alpha=0.4,
        dt_s=0.02,
        cfl_limit=0.25,
        potential_operator=MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
        potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        potential_rtol=1.0e-10,
    )
    closure = DimensionalFACMimeticClosure(
        config, background_field_t=_B0_T
    )
    # Construct the legacy spatial model without its PARDISO-selecting
    # initializer; the injected conservative closure uses SciPy on this 9x9
    # validation grid.
    legacy = ESFModel.__new__(ESFModel)
    legacy.config = config
    legacy.potential_solver = closure
    basis = ModalBasis(
        nx=config.nx,
        ny=config.ny,
        dx_m=config.dx_m,
        dy_m=config.dy_m,
        modes=(
            TransverseMode(1, 0, "cos"),
            TransverseMode(1, 0, "sin"),
            TransverseMode(0, 1, "cos"),
        ),
    )
    line = ModalAlfvenLine(
        length_m=8.0e6,
        cells=8,
        alfven_speed_m_s=8.0e5,
        mode_count=basis.mode_count,
        spatial_order=1,
    )
    feedback = PhysicalFeedbackConfig(
        gain=1.0,
        under_relaxation=0.4,
        max_iterations=160,
        relative_tolerance=5.0e-9,
    )
    coupled = DimensionalBidirectionalModel(
        legacy, closure, basis, line, feedback
    )
    es_limit = DimensionalBidirectionalModel(
        legacy, closure, basis, line, PhysicalFeedbackConfig(gain=0.0)
    )

    y = np.arange(config.ny, dtype=np.float64)[:, None]
    x = np.arange(config.nx, dtype=np.float64)[None, :]
    density = 5.0e11 * (
        1.0
        + 0.06 * np.cos(2.0 * np.pi * x / config.nx)
        + 0.015 * (y - np.mean(y)) / config.ny
    )
    line_state = coupled.initial_alfven_state_from_li_potential(density)
    plus = line_state.w_plus_v.copy()
    minus = line_state.w_minus_v.copy()
    plus[0, 0] += 2.0e-3
    plus[0, 1] -= 0.8e-3
    seeded = ModalAlfvenState(plus, minus)
    initial = DimensionalEMState(density.copy(), seeded)

    expected_rhs, _ = legacy.evaluate_rhs(density)
    exact_rhs, _ = es_limit.evaluate_rhs(0.0, initial)
    exact_es = bool(np.array_equal(exact_rhs.density_m3, expected_rhs))

    baseline = initial
    electromagnetic = initial
    time_s = 0.0
    last_diagnostics = None
    for _ in range(3):
        baseline, _ = es_limit.ssprk3_step(time_s, baseline, config.dt_s)
        electromagnetic, last_diagnostics = coupled.ssprk3_step(
            time_s, electromagnetic, config.dt_s
        )
        time_s += config.dt_s
    assert last_diagnostics is not None

    density_difference = float(
        np.linalg.norm(electromagnetic.density_m3 - baseline.density_m3)
        / max(np.linalg.norm(baseline.density_m3), np.finfo(float).tiny)
    )
    amplitude_difference = abs(
        density_mode_amplitude(electromagnetic.density_m3)
        - density_mode_amplitude(baseline.density_m3)
    ) / max(density_mode_amplitude(baseline.density_m3), np.finfo(float).tiny)
    net_scale = (
        last_diagnostics.maximum_abs_d_j_parallel_ds_a_m3
        * config.nx
        * config.ny
        * config.dx_m
        * config.dy_m
    )
    relative_net = abs(last_diagnostics.net_d_j_parallel_ds_a_m) / max(
        net_scale, np.finfo(float).tiny
    )
    criteria = {
        "exact_es_density_rhs": exact_es,
        "coupling_converged": (
            last_diagnostics.relative_coupling_residual
            <= feedback.relative_tolerance
        ),
        "transverse_return_current": relative_net <= 1.0e-12,
        "modal_grid_power_adjoint": (
            last_diagnostics.relative_power_pairing_defect <= 1.0e-12
        ),
        "positive_density": bool(np.min(electromagnetic.density_m3) > 0.0),
        "nonlinear_density_feedback_detected": density_difference > 1.0e-16,
    }
    metric = 0.0 if all(criteria.values()) else 1.0
    return _record(
        9,
        "dimensional_fac_laplacian_nonlinear_feedback",
        metric,
        0.0,
        details={
            "scope": "tiny_grid_physical_reduced_em_feedback",
            "physical_dimensional_fac_to_li_closure_complete": True,
            "transverse_laplacian_fac_mapping_complete": True,
            "small_grid_nonlinear_two_way_run_complete": True,
            "formal_nonlinear_epb_two_way_run_complete": False,
            "parallel_mass_transport_included": False,
            "grid": [config.ny, config.nx, line.cells],
            "retained_transverse_modes": basis.mode_count,
            "steps": 3,
            "final_time_s": time_s,
            "criteria": criteria,
            "coupling_iterations": last_diagnostics.iterations,
            "coupling_relative_residual": (
                last_diagnostics.relative_coupling_residual
            ),
            "relative_net_current_gradient": relative_net,
            "relative_power_pairing_defect": (
                last_diagnostics.relative_power_pairing_defect
            ),
            "relative_density_trajectory_difference": density_difference,
            "relative_density_mode_amplitude_difference": amplitude_difference,
            "minimum_density_m3": float(np.min(electromagnetic.density_m3)),
            "pardiso_used": False,
        },
    )


_CHECKS: tuple[Callable[[], dict[str, Any]], ...] = (
    _stage_1,
    _stage_2,
    _stage_3,
    _stage_4,
    _stage_5,
    _stage_6,
    _stage_7,
    _stage_8,
    _stage_9,
)


def _failed_check(level: int, error: Exception) -> dict[str, Any]:
    return {
        "level": int(level),
        "name": f"stage_{level}_validation",
        "status": "error",
        "metric": None,
        "threshold": None,
        "criterion": None,
        "details": {"error_type": type(error).__name__, "error": str(error)},
    }


def _report_path(output: str | Path) -> Path:
    path = Path(output)
    return path if path.suffix.lower() == ".json" else path / "em_validation_report.json"


def run_em_validation(output: str | Path | None = None) -> dict[str, Any]:
    """Run stages 1--9 and optionally write the JSON report atomically."""

    levels: list[dict[str, Any]] = []
    for level, check in enumerate(_CHECKS, start=1):
        try:
            levels.append(check())
        except Exception as error:  # keep later stages independently auditable
            levels.append(_failed_check(level, error))

    report: dict[str, Any] = {
        "format": "esf_reduced_em_validation_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_pass": all(level["status"] == "pass" for level in levels),
        "levels": levels,
        "execution_scope": {
            "formal_li_snapshot_used": False,
            "pardiso_used": False,
            "long_epb_simulation_used": False,
            "stages_1_through_7": "physical_reduced_em_components",
            "stage_8": "controlled_feedback_layer_with_toy_closure",
            "stage_9": "dimensional_transverse_modal_fac_and_tiny_nonlinear_feedback",
        },
        "remaining_work": [
            "add two hemispheres and a metric/SBP field-line geometry",
            "add common-mode parallel mass transport for a full 3-D plasma model",
            "run resolution studies before a formal 201x401 nonlinear EPB comparison",
        ],
    }

    if output is not None:
        destination = _report_path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(destination)
        report["report_path"] = str(destination)
    return report


__all__ = ["run_em_validation"]
