"""Checkpointed nonlinear electrostatic-versus-reduced-EM growth experiment.

The runner is intentionally separate from the Figure reproduction commands.
Every electromagnetic parameter that cancelled from the Li electrostatic
equation (``B0``, ion mass, field-line length, ``V_A`` and retained transverse
modes) is explicit in :class:`EMGrowthConfig`; a formal experiment must archive
those choices instead of silently inventing them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import _model_impl
from .alfven_1d import MU0
from .bidirectional_em import (
    DimensionalBidirectionalModel,
    DimensionalEMState,
    PhysicalFeedbackConfig,
    PhysicalCouplingConvergenceError,
    density_mode_amplitude,
)
from .config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    BEST_BASELINE_POTENTIAL_OPERATOR,
    MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
    SimulationConfig,
)
from .fac_closure import (
    ATOMIC_MASS_KG,
    DimensionalFACMimeticClosure,
    DimensionalFACPointCenteredClosure,
)
from .flux_tube_column import dipole_chapman_column_profile
from .modal_alfven import ModalAlfvenLine, ModalAlfvenState, ModalBasis
from .model import ESFModel, initial_density


@dataclass(frozen=True)
class EMGrowthConfig:
    """Configuration of one paired ES/reduced-EM trajectory."""

    background_field_t: float = 3.0e-5
    ion_mass_amu: float = 15.999
    field_line_length_km: float = 20_000.0
    field_line_cells: int = 65
    alfven_speed_km_s: float = 1_000.0
    max_x_wavenumber: int = 4
    max_y_wavenumber: int = 4
    include_sine_modes: bool = True
    line_spatial_order: int = 2
    right_boundary_voltage_v: float = 0.0
    right_boundary_model: str = "fixed_voltage"
    right_boundary_conductance_s: float | None = None

    feedback_gain: float = 1.0
    feedback_ramp_time_s: float = 20.0
    coupling_solver: str = "modal_woodbury"
    coupling_under_relaxation: float = 0.5
    coupling_max_iterations: int = 80
    coupling_relative_tolerance: float = 1.0e-8
    line_cfl_limit: float = 0.8
    fac_source_model: str = "boundary_gradient"
    parallel_averaging_length_km: float | None = None
    parallel_averaging_profile: str = "constant"
    dipole_earth_radius_km: float = 6371.0
    dipole_column_lower_altitude_km: float = 100.0
    dipole_quadrature_order: int = 128
    hemisphere_count: int = 1

    closure: str = "mimetic"
    use_pardiso: bool = True
    dt_s: float = 0.25
    adaptive_dt_halving: bool = True
    minimum_dt_s: float = 1.0e-6
    t_end_s: float = 60.0
    growth_fit_start_s: float = 20.0
    diagnostics_interval_steps: int = 1
    checkpoint_interval_s: float = 10.0
    snapshot_interval_s: float | None = None
    initial_line_state: str = "li_potential_profile"
    initial_characteristic_seed_v: float = 0.0
    initial_seed_mode: int = 0
    output_dir: str = "output/em_growth"
    experiment_label: str = "reduced_em_preliminary"
    comparison_tracks: str = "legacy_pair"
    # Default preserves historical trajectories and hashes. The explicit new
    # layout stores only unique periodic cells, with period nx * dx.
    periodic_x_layout: str = "legacy_stored_cells"
    # Imposed upward (+y) density drift, m/s; not a background-current closure.
    background_exb_y_m_s: float = 0.0

    def __post_init__(self) -> None:
        if self.periodic_x_layout not in ("legacy_stored_cells", "unique_periodic_cells"):
            raise ValueError("unsupported periodic_x_layout")
        if not np.isfinite(float(self.background_exb_y_m_s)):
            raise ValueError("background_exb_y_m_s must be finite")
        positive = {
            "background_field_t": self.background_field_t,
            "ion_mass_amu": self.ion_mass_amu,
            "field_line_length_km": self.field_line_length_km,
            "alfven_speed_km_s": self.alfven_speed_km_s,
            "dt_s": self.dt_s,
            "minimum_dt_s": self.minimum_dt_s,
            "t_end_s": self.t_end_s,
            "checkpoint_interval_s": self.checkpoint_interval_s,
            "coupling_relative_tolerance": self.coupling_relative_tolerance,
        }
        for name, value in positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.field_line_cells < 4:
            raise ValueError("field_line_cells must be at least four")
        if self.max_x_wavenumber < 0 or self.max_y_wavenumber < 0:
            raise ValueError("maximum transverse wavenumbers must be non-negative")
        if self.max_x_wavenumber == 0 and self.max_y_wavenumber == 0:
            raise ValueError("at least one non-zero transverse mode is required")
        if self.line_spatial_order not in (1, 2):
            raise ValueError("line_spatial_order must be one or two")
        if not np.isfinite(float(self.right_boundary_voltage_v)):
            raise ValueError("right_boundary_voltage_v must be finite")
        if self.right_boundary_model not in (
            "fixed_voltage",
            "matched",
            "conductance",
        ):
            raise ValueError(
                "right_boundary_model must be 'fixed_voltage', 'matched', "
                "or 'conductance'"
            )
        if self.right_boundary_model == "fixed_voltage":
            if self.right_boundary_conductance_s is not None:
                raise ValueError(
                    "right_boundary_conductance_s is incompatible with a "
                    "fixed-voltage boundary"
                )
        elif self.right_boundary_model == "matched":
            if self.right_boundary_voltage_v != 0.0:
                raise ValueError(
                    "matched remote boundary requires zero boundary voltage"
                )
            if self.right_boundary_conductance_s is not None:
                raise ValueError(
                    "matched remote boundary determines conductance from V_A"
                )
        else:
            if self.right_boundary_voltage_v != 0.0:
                raise ValueError(
                    "conductance remote boundary requires zero boundary voltage"
                )
            if self.right_boundary_conductance_s is None or (
                not np.isfinite(float(self.right_boundary_conductance_s))
                or float(self.right_boundary_conductance_s) < 0.0
            ):
                raise ValueError(
                    "right_boundary_conductance_s must be non-negative and "
                    "finite for a conductance boundary"
                )
        if self.closure not in ("mimetic", "point_centered"):
            raise ValueError("closure must be 'mimetic' or 'point_centered'")
        if self.coupling_solver not in (
            "modal_woodbury",
            "direct_krylov",
            "fixed_point",
        ):
            raise ValueError("unsupported coupling_solver")
        if self.diagnostics_interval_steps < 1:
            raise ValueError("diagnostics_interval_steps must be positive")
        if not isinstance(self.adaptive_dt_halving, bool):
            raise ValueError("adaptive_dt_halving must be a boolean")
        if self.minimum_dt_s > self.dt_s:
            raise ValueError("minimum_dt_s must not exceed dt_s")
        if self.snapshot_interval_s is not None and (
            not np.isfinite(float(self.snapshot_interval_s))
            or float(self.snapshot_interval_s) <= 0.0
        ):
            raise ValueError("snapshot_interval_s must be positive when enabled")
        if self.initial_line_state not in ("zero", "li_potential_profile"):
            raise ValueError(
                "initial_line_state must be 'zero' or 'li_potential_profile'"
            )
        if not np.isfinite(self.line_cfl_limit) or not (
            0.0 < self.line_cfl_limit <= 1.0
        ):
            raise ValueError("line_cfl_limit must lie in (0, 1]")
        if self.fac_source_model not in (
            "boundary_gradient",
            "symmetric_flux_tube",
        ):
            raise ValueError("unsupported fac_source_model")
        if self.parallel_averaging_profile not in (
            "constant",
            "dipole_chapman_flux_tube",
        ):
            raise ValueError(
                "parallel_averaging_profile must be 'constant' or "
                "'dipole_chapman_flux_tube'"
            )
        if self.hemisphere_count < 1:
            raise ValueError("hemisphere_count must be positive")
        if self.fac_source_model == "symmetric_flux_tube":
            if self.parallel_averaging_profile == "constant":
                if self.parallel_averaging_length_km is None:
                    raise ValueError(
                        "parallel_averaging_length_km is required for the "
                        "constant symmetric_flux_tube profile"
                    )
                if (
                    not np.isfinite(float(self.parallel_averaging_length_km))
                    or float(self.parallel_averaging_length_km) <= 0.0
                ):
                    raise ValueError(
                        "parallel_averaging_length_km must be positive and finite"
                    )
            elif self.parallel_averaging_length_km is not None:
                raise ValueError(
                    "parallel_averaging_length_km must be null for the "
                    "dipole/Chapman profile"
                )
        elif self.parallel_averaging_profile != "constant":
            raise ValueError(
                "a variable parallel_averaging_profile requires "
                "symmetric_flux_tube FAC forcing"
            )
        if (
            not np.isfinite(float(self.dipole_earth_radius_km))
            or float(self.dipole_earth_radius_km) <= 0.0
        ):
            raise ValueError("dipole_earth_radius_km must be positive and finite")
        if (
            not np.isfinite(float(self.dipole_column_lower_altitude_km))
            or float(self.dipole_column_lower_altitude_km) < 0.0
        ):
            raise ValueError(
                "dipole_column_lower_altitude_km must be non-negative and finite"
            )
        if (
            isinstance(self.dipole_quadrature_order, bool)
            or int(self.dipole_quadrature_order) != self.dipole_quadrature_order
            or int(self.dipole_quadrature_order) < 16
        ):
            raise ValueError(
                "dipole_quadrature_order must be an integer of at least 16"
            )
        if self.growth_fit_start_s < 0.0 or self.growth_fit_start_s > self.t_end_s:
            raise ValueError("growth_fit_start_s must lie in [0, t_end_s]")
        if not np.isfinite(self.initial_characteristic_seed_v):
            raise ValueError("initial_characteristic_seed_v must be finite")
        if self.comparison_tracks not in ("legacy_pair", "local_column_em"):
            raise ValueError(
                "comparison_tracks must be 'legacy_pair' or 'local_column_em'"
            )
        if self.comparison_tracks == "local_column_em" and (
            self.fac_source_model != "symmetric_flux_tube"
            or self.parallel_averaging_profile != "dipole_chapman_flux_tube"
        ):
            raise ValueError(
                "local_column_em comparison requires symmetric_flux_tube FAC "
                "forcing with the dipole_chapman_flux_tube profile"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "EMGrowthConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown EM growth configuration keys: {unknown}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def resolved_remote_conductance_s(self) -> float | None:
        """Return the passive terminal conductance, when applicable."""

        if self.right_boundary_model == "fixed_voltage":
            return None
        sigma_a = 1.0 / (
            MU0 * float(self.alfven_speed_km_s) * 1000.0
        )
        if self.right_boundary_model == "matched":
            return float(sigma_a)
        return float(self.right_boundary_conductance_s)

    @property
    def remote_reflection_coefficient(self) -> float:
        """Return the terminal's small-signal voltage reflection factor."""

        if self.right_boundary_model == "fixed_voltage":
            return -1.0
        if self.right_boundary_model == "matched":
            return 0.0
        sigma_a = 1.0 / (
            MU0 * float(self.alfven_speed_km_s) * 1000.0
        )
        sigma_p = float(self.right_boundary_conductance_s)
        return float((sigma_a - sigma_p) / (sigma_a + sigma_p))


class MinimumTimeStepError(RuntimeError):
    """A retryable trial still failed at the configured minimum time step."""

    def __init__(
        self,
        time_s: float,
        dt_s: float,
        cause: Exception,
        *,
        component: str = "reduced-EM",
    ) -> None:
        self.time_s = float(time_s)
        self.dt_s = float(dt_s)
        self.cause = cause
        self.component = str(component)
        super().__init__(
            f"{self.component} step at t={time_s:.12g} s failed at minimum "
            f"dt={dt_s:.12g} s: {cause}"
        )


@dataclass(frozen=True)
class _BaselineAdvance:
    """One baseline advance over an EM macro step.

    ``substeps`` is one on the historical path.  A value greater than one
    means that only the electrostatic control trajectory was restarted from
    the macro-step input and advanced with uniform dyadic substeps.
    """

    state: DimensionalEMState
    diagnostics: Any
    substeps: int
    substep_dt_s: float
    maximum_substep_cfl: float


@dataclass(frozen=True)
class _ColumnControlAdvance:
    """One column-ES control advance over an EM macro step."""

    density_m3: np.ndarray
    diagnostics: Any
    substeps: int
    substep_dt_s: float
    maximum_substep_cfl: float


def _advance_baseline_with_subcycling(
    model: DimensionalBidirectionalModel,
    *,
    time_s: float,
    state: DimensionalEMState,
    macro_dt_s: float,
    adaptive: bool,
    minimum_dt_s: float,
) -> _BaselineAdvance:
    """Advance the ES baseline without reducing the expensive EM macro step.

    The first attempt is exactly the historical single SSPRK3 call.  Only a
    baseline density-CFL rejection activates dyadic subcycling.  Each finer
    attempt restarts from ``state`` so a rejected partial trajectory can never
    leak into the accepted result.
    """

    subdivisions = 1
    while True:
        substep_dt_s = float(macro_dt_s) / subdivisions
        candidate = state
        diagnostics: list[Any] = []
        try:
            for substep in range(subdivisions):
                candidate, diagnostic = model.ssprk3_step(
                    time_s + substep * substep_dt_s,
                    candidate,
                    substep_dt_s,
                )
                diagnostics.append(diagnostic)
        except _model_impl.CFLViolation as error:
            if not adaptive:
                raise
            next_subdivisions = 2 * subdivisions
            next_substep_dt_s = float(macro_dt_s) / next_subdivisions
            tolerance = 64.0 * np.finfo(float).eps * max(
                1.0, abs(float(macro_dt_s)), abs(float(minimum_dt_s))
            )
            if next_substep_dt_s < float(minimum_dt_s) - tolerance:
                raise MinimumTimeStepError(
                    time_s,
                    substep_dt_s,
                    error,
                    component="electrostatic baseline",
                ) from error
            subdivisions = next_subdivisions
            continue

        diagnostic = max(
            diagnostics,
            key=lambda item: getattr(item, "combined_cfl_rate_s_inv", 0.0),
        )
        maximum_substep_cfl = max(
            getattr(item, "combined_cfl_rate_s_inv", 0.0) * substep_dt_s
            for item in diagnostics
        )
        return _BaselineAdvance(
            state=candidate,
            diagnostics=diagnostic,
            substeps=subdivisions,
            substep_dt_s=substep_dt_s,
            maximum_substep_cfl=float(maximum_substep_cfl),
        )


def _column_control_ssprk3_step(
    model: DimensionalBidirectionalModel,
    time_s: float,
    density_m3: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, Any]:
    """Advance the zero-FAC column control with the runner's SSPRK3 scheme."""

    def evaluate(stage_time_s: float, density: np.ndarray):
        tendency, diagnostics = model.evaluate_column_es_rhs(density)
        cfl = float(diagnostics.cfl_rate) * float(dt_s)
        if cfl > model.config.cfl_limit:
            raise _model_impl.CFLViolation(
                f"column-ES density CFL={cfl:.6g} exceeds configured limit "
                f"{model.config.cfl_limit:.6g}"
            )
        return tendency, diagnostics

    density = np.asarray(density_m3, dtype=np.float64)
    k0, d0 = evaluate(time_s, density)
    u1 = density + dt_s * k0
    k1, d1 = evaluate(time_s + dt_s, u1)
    u2 = 0.75 * density + 0.25 * (u1 + dt_s * k1)
    k2, d2 = evaluate(time_s + 0.5 * dt_s, u2)
    result = (1.0 / 3.0) * density + (2.0 / 3.0) * (u2 + dt_s * k2)
    diagnostics = max((d0, d1, d2), key=lambda item: item.cfl_rate)
    return np.asarray(result, dtype=np.float64), diagnostics


def _advance_column_control_with_subcycling(
    model: DimensionalBidirectionalModel,
    *,
    time_s: float,
    density_m3: np.ndarray,
    macro_dt_s: float,
    adaptive: bool,
    minimum_dt_s: float,
) -> _ColumnControlAdvance:
    """Advance column-ES with independent dyadic density-CFL subcycling."""

    subdivisions = 1
    while True:
        substep_dt_s = float(macro_dt_s) / subdivisions
        candidate = np.asarray(density_m3, dtype=np.float64)
        diagnostics: list[Any] = []
        try:
            for substep in range(subdivisions):
                candidate, diagnostic = _column_control_ssprk3_step(
                    model,
                    time_s + substep * substep_dt_s,
                    candidate,
                    substep_dt_s,
                )
                diagnostics.append(diagnostic)
        except _model_impl.CFLViolation as error:
            if not adaptive:
                raise
            next_subdivisions = 2 * subdivisions
            next_substep_dt_s = float(macro_dt_s) / next_subdivisions
            tolerance = 64.0 * np.finfo(float).eps * max(
                1.0, abs(float(macro_dt_s)), abs(float(minimum_dt_s))
            )
            if next_substep_dt_s < float(minimum_dt_s) - tolerance:
                raise MinimumTimeStepError(
                    time_s,
                    substep_dt_s,
                    error,
                    component="column-ES control",
                ) from error
            subdivisions = next_subdivisions
            continue

        diagnostic = max(diagnostics, key=lambda item: item.cfl_rate)
        maximum_substep_cfl = max(
            float(item.cfl_rate) * substep_dt_s for item in diagnostics
        )
        return _ColumnControlAdvance(
            density_m3=candidate,
            diagnostics=diagnostic,
            substeps=subdivisions,
            substep_dt_s=substep_dt_s,
            maximum_substep_cfl=float(maximum_substep_cfl),
        )


def _configuration_hash(
    li: SimulationConfig, em: EMGrowthConfig, *, legacy_v1: bool = False
) -> str:
    em_payload = em.to_dict()
    # Zero drift preserves all historical trajectory hashes and code paths.
    if em.background_exb_y_m_s == 0.0:
        em_payload.pop("background_exb_y_m_s")
    else:
        em_payload["background_exb_y_m_s"] = float(em.background_exb_y_m_s)
    if em.periodic_x_layout == "legacy_stored_cells":
        em_payload.pop("periodic_x_layout")
    # ``comparison_tracks`` was introduced with schema v3.  Its default is
    # the historical paired runner, so omitting it preserves every legacy
    # schema-v1/v2 trajectory hash exactly.
    if em.comparison_tracks == "legacy_pair":
        em_payload.pop("comparison_tracks")
    # Before configurable remote loads existed, ``right_boundary_voltage_v``
    # unambiguously meant the fixed-voltage terminal.  Omit the two new fields
    # for that exact historical path so existing schema-v2 checkpoints and
    # snapshots remain resumable byte-for-byte.  Passive alternatives retain
    # their model/conductance in the protected trajectory hash.
    if em.right_boundary_model == "fixed_voltage":
        em_payload.pop("right_boundary_model")
        em_payload.pop("right_boundary_conductance_s")
    if em.parallel_averaging_profile == "constant":
        # These defaults did not exist when scalar-H schema-v2 checkpoints
        # were written.  The historical trajectory is fully specified by
        # parallel_averaging_length_km, so omit the inactive profile controls.
        for key in (
            "parallel_averaging_profile",
            "dipole_earth_radius_km",
            "dipole_column_lower_altitude_km",
            "dipole_quadrature_order",
        ):
            em_payload.pop(key)
    else:
        # The first experimental variable profile used one density-weighted
        # H for every term and stored FAC as J/H.  The present reduction has
        # independent particle/collision moments, surface FAC, and column
        # density continuity; protect it from resuming those incompatible
        # checkpoints without perturbing any scalar-H historical hash.
        em_payload["parallel_column_reduction_version"] = (
            "two_moment_surface_fac_column_continuity_v2"
        )
    # Operational/reporting controls may change when a stable run is extended;
    # they do not alter any already-computed time step.  Numerical/physical
    # controls, including dt and solver tolerance, remain hash protected.
    for key in (
        "t_end_s",
        "growth_fit_start_s",
        "diagnostics_interval_steps",
        "checkpoint_interval_s",
        "output_dir",
        "experiment_label",
    ):
        em_payload.pop(key)
    if legacy_v1:
        # Checkpoints written before adaptive stepping/snapshots existed did
        # not contain these keys.  Accept that exact historical hash once,
        # then upgrade the checkpoint to schema v2 at the next save.
        for key in (
            "adaptive_dt_halving",
            "minimum_dt_s",
            "snapshot_interval_s",
        ):
            em_payload.pop(key)
    payload = json.dumps(
        {"li": li.to_dict(), "em_trajectory": em_payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _snapshot_file(snapshot_dir: Path, index: int) -> Path:
    return snapshot_dir / f"snapshot_{int(index):06d}.npz"


def _save_density_snapshot(
    path: Path,
    *,
    index: int,
    time_s: float,
    step: int,
    baseline: DimensionalEMState,
    coupled: DimensionalEMState,
    configuration_hash: str,
    column_control_density_m3: np.ndarray | None = None,
) -> dict[str, Any]:
    """Atomically archive one exact-time paired or three-track snapshot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    if column_control_density_m3 is None:
        # Do not perturb the historical key set/order or schema number.
        np.savez_compressed(
            temporary,
            schema_version=np.int64(1),
            snapshot_index=np.int64(index),
            time_s=np.float64(time_s),
            step=np.int64(step),
            configuration_hash=np.asarray(configuration_hash),
            baseline_density_m3=np.asarray(
                baseline.density_m3, dtype=np.float64
            ),
            coupled_density_m3=np.asarray(coupled.density_m3, dtype=np.float64),
        )
    else:
        column = np.asarray(column_control_density_m3, dtype=np.float64)
        if column.shape != baseline.density_m3.shape:
            raise ValueError("column-control snapshot density shape mismatch")
        np.savez_compressed(
            temporary,
            schema_version=np.int64(2),
            snapshot_index=np.int64(index),
            time_s=np.float64(time_s),
            step=np.int64(step),
            configuration_hash=np.asarray(configuration_hash),
            baseline_density_m3=np.asarray(
                baseline.density_m3, dtype=np.float64
            ),
            column_control_density_m3=column,
            coupled_density_m3=np.asarray(coupled.density_m3, dtype=np.float64),
        )
    temporary.replace(path)
    return {
        "index": int(index),
        "time_s": float(time_s),
        "step": int(step),
        "file": str(path),
    }


def _write_snapshot_manifest(
    path: Path,
    *,
    configuration_hash: str,
    interval_s: float,
    records: list[dict[str, Any]],
    comparison_tracks: str = "legacy_pair",
) -> None:
    """Write a constant-size atomic manifest for deterministic snapshot names."""

    latest = records[-1] if records else None
    payload = {
            "format": (
                "esf_three_track_density_snapshots_v2"
                if comparison_tracks == "local_column_em"
                else "esf_paired_density_snapshots_v1"
            ),
            "configuration_hash": configuration_hash,
            "interval_s": float(interval_s),
            "file_pattern": "snapshots/snapshot_{index:06d}.npz",
            "count": len(records),
            "first_time_s": records[0]["time_s"] if records else None,
            "last_time_s": latest["time_s"] if latest else None,
            "last_step": latest["step"] if latest else None,
        }
    if comparison_tracks == "local_column_em":
        payload["comparison_tracks"] = ["local_es", "column_es", "column_em"]
        payload["snapshot_schema_version"] = 2
    _write_json(path, payload)


def _clear_snapshot_outputs(output: Path) -> None:
    snapshot_dir = output / "snapshots"
    if snapshot_dir.exists():
        for path in snapshot_dir.glob("snapshot_*.npz"):
            path.unlink()
    manifest = output / "snapshot_manifest.json"
    if manifest.exists():
        manifest.unlink()


def _reconcile_snapshots(
    output: Path,
    *,
    checkpoint_time_s: float,
    interval_s: float,
    configuration_hash: str,
    comparison_tracks: str = "legacy_pair",
) -> list[dict[str, Any]]:
    """Validate the exact prefix and discard snapshots newer than a checkpoint."""

    snapshot_dir = output / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(checkpoint_time_s), abs(interval_s)
    )
    maximum_index = int(np.floor((checkpoint_time_s + tolerance) / interval_s))
    for path in snapshot_dir.glob("snapshot_*.npz"):
        try:
            index = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid snapshot filename: {path.name}") from error
        if index > maximum_index:
            path.unlink()

    records: list[dict[str, Any]] = []
    for index in range(maximum_index + 1):
        path = _snapshot_file(snapshot_dir, index)
        if not path.exists():
            raise ValueError(
                f"missing paired density snapshot {path.name} before checkpoint"
            )
        with np.load(path, allow_pickle=False) as payload:
            expected_schema = 2 if comparison_tracks == "local_column_em" else 1
            if int(payload["schema_version"]) != expected_schema:
                raise ValueError(f"unsupported snapshot schema in {path.name}")
            if int(payload["snapshot_index"]) != index:
                raise ValueError(f"snapshot index mismatch in {path.name}")
            actual_hash = str(payload["configuration_hash"])
            if actual_hash != configuration_hash:
                raise ValueError(f"snapshot configuration hash mismatch in {path.name}")
            actual_time = float(payload["time_s"])
            expected_time = float(index * interval_s)
            if not np.isclose(actual_time, expected_time, rtol=0.0, atol=tolerance):
                raise ValueError(f"snapshot time mismatch in {path.name}")
            step = int(payload["step"])
            if comparison_tracks == "local_column_em":
                baseline_shape = np.asarray(payload["baseline_density_m3"]).shape
                if np.asarray(payload["column_control_density_m3"]).shape != baseline_shape:
                    raise ValueError(
                        f"column-control density shape mismatch in {path.name}"
                    )
                if np.asarray(payload["coupled_density_m3"]).shape != baseline_shape:
                    raise ValueError(f"coupled density shape mismatch in {path.name}")
        records.append(
            {
                "index": index,
                "time_s": expected_time,
                "step": step,
                "file": str(path),
            }
        )
    _write_snapshot_manifest(
        output / "snapshot_manifest.json",
        configuration_hash=configuration_hash,
        interval_s=interval_s,
        records=records,
        comparison_tracks=comparison_tracks,
    )
    return records


def _truncate_diagnostics_after_step(path: Path, maximum_step: int) -> None:
    """Drop records newer than a resumed stable checkpoint atomically."""

    if not path.exists():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("step", -1)) <= int(maximum_step):
            retained.append(json.dumps(record, ensure_ascii=False))
    temporary = path.with_suffix(".tmp.jsonl")
    temporary.write_text(
        "" if not retained else "\n".join(retained) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _baseline_subcycling_totals(path: Path) -> tuple[int, int]:
    """Recover cumulative subcycling counters from the auditable event log."""

    if not path.exists():
        return 0, 0
    events = 0
    extra_substeps = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "baseline_subcycling":
            continue
        substeps = int(record["baseline_substeps"])
        if substeps < 2:
            raise ValueError("baseline subcycling event must contain at least two steps")
        events += 1
        extra_substeps += substeps - 1
    return events, extra_substeps


def _column_subcycling_totals(path: Path) -> tuple[int, int]:
    """Recover column-control subcycling counters from the event log."""

    if not path.exists():
        return 0, 0
    events = 0
    extra_substeps = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "column_control_subcycling":
            continue
        substeps = int(record["column_control_substeps"])
        if substeps < 2:
            raise ValueError(
                "column-control subcycling event must contain at least two steps"
            )
        events += 1
        extra_substeps += substeps - 1
    return events, extra_substeps


def _save_checkpoint(
    path: Path,
    *,
    time_s: float,
    step: int,
    baseline: DimensionalEMState,
    coupled: DimensionalEMState,
    configuration_hash: str,
    times_s: list[float],
    baseline_amplitudes_m3: list[float],
    coupled_amplitudes_m3: list[float],
    current_dt_s: float,
    dt_reductions: int,
    column_control_density_m3: np.ndarray | None = None,
    column_control_amplitudes_m3: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    if (column_control_density_m3 is None) != (
        column_control_amplitudes_m3 is None
    ):
        raise ValueError("column-control checkpoint state/history must be paired")
    if column_control_density_m3 is None:
        # Historical schema-v2 byte/key path is intentionally untouched.
        np.savez_compressed(
            temporary,
            schema_version=np.int64(2),
            time_s=np.float64(time_s),
            step=np.int64(step),
            configuration_hash=np.asarray(configuration_hash),
            baseline_density_m3=baseline.density_m3,
            baseline_w_plus_v=baseline.alfven.w_plus_v,
            baseline_w_minus_v=baseline.alfven.w_minus_v,
            coupled_density_m3=coupled.density_m3,
            coupled_w_plus_v=coupled.alfven.w_plus_v,
            coupled_w_minus_v=coupled.alfven.w_minus_v,
            times_s=np.asarray(times_s, dtype=np.float64),
            baseline_amplitudes_m3=np.asarray(
                baseline_amplitudes_m3, dtype=np.float64
            ),
            coupled_amplitudes_m3=np.asarray(
                coupled_amplitudes_m3, dtype=np.float64
            ),
            current_dt_s=np.float64(current_dt_s),
            dt_reductions=np.int64(dt_reductions),
        )
    else:
        column = np.asarray(column_control_density_m3, dtype=np.float64)
        if column.shape != baseline.density_m3.shape:
            raise ValueError("column-control checkpoint density shape mismatch")
        history_length = len(times_s)
        if not (
            len(baseline_amplitudes_m3)
            == len(column_control_amplitudes_m3)
            == len(coupled_amplitudes_m3)
            == history_length
        ):
            raise ValueError("three-track checkpoint history lengths differ")
        np.savez_compressed(
            temporary,
            schema_version=np.int64(3),
            time_s=np.float64(time_s),
            step=np.int64(step),
            configuration_hash=np.asarray(configuration_hash),
            comparison_tracks=np.asarray("local_column_em"),
            baseline_density_m3=baseline.density_m3,
            baseline_w_plus_v=baseline.alfven.w_plus_v,
            baseline_w_minus_v=baseline.alfven.w_minus_v,
            column_control_density_m3=column,
            coupled_density_m3=coupled.density_m3,
            coupled_w_plus_v=coupled.alfven.w_plus_v,
            coupled_w_minus_v=coupled.alfven.w_minus_v,
            times_s=np.asarray(times_s, dtype=np.float64),
            baseline_amplitudes_m3=np.asarray(
                baseline_amplitudes_m3, dtype=np.float64
            ),
            column_control_amplitudes_m3=np.asarray(
                column_control_amplitudes_m3, dtype=np.float64
            ),
            coupled_amplitudes_m3=np.asarray(
                coupled_amplitudes_m3, dtype=np.float64
            ),
            current_dt_s=np.float64(current_dt_s),
            dt_reductions=np.int64(dt_reductions),
        )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    expected_hash: str,
    legacy_v1_hash: str | None = None,
    *,
    comparison_tracks: str = "legacy_pair",
) -> tuple[
    float,
    int,
    DimensionalEMState,
    DimensionalEMState,
    list[float],
    list[float],
    list[float],
    np.ndarray | None,
    list[float] | None,
    float | None,
    int,
]:
    with np.load(path, allow_pickle=False) as payload:
        schema_version = int(payload["schema_version"])
        expected_schemas = (3,) if comparison_tracks == "local_column_em" else (1, 2)
        if schema_version not in expected_schemas:
            raise ValueError("unsupported EM checkpoint schema")
        actual_hash = str(payload["configuration_hash"])
        hash_matches = actual_hash == expected_hash or (
            schema_version == 1
            and legacy_v1_hash is not None
            and actual_hash == legacy_v1_hash
        )
        if not hash_matches:
            raise ValueError("EM checkpoint configuration hash mismatch")
        baseline = DimensionalEMState(
            np.asarray(payload["baseline_density_m3"], dtype=np.float64),
            ModalAlfvenState(
                np.asarray(payload["baseline_w_plus_v"], dtype=np.float64),
                np.asarray(payload["baseline_w_minus_v"], dtype=np.float64),
            ),
        )
        coupled = DimensionalEMState(
            np.asarray(payload["coupled_density_m3"], dtype=np.float64),
            ModalAlfvenState(
                np.asarray(payload["coupled_w_plus_v"], dtype=np.float64),
                np.asarray(payload["coupled_w_minus_v"], dtype=np.float64),
            ),
        )
        column_control_density = None
        column_control_amplitudes = None
        if schema_version == 3:
            if str(payload["comparison_tracks"]) != "local_column_em":
                raise ValueError("invalid three-track checkpoint marker")
            column_control_density = np.asarray(
                payload["column_control_density_m3"], dtype=np.float64
            )
            column_control_amplitudes = np.asarray(
                payload["column_control_amplitudes_m3"], dtype=np.float64
            ).tolist()
            if column_control_density.shape != baseline.density_m3.shape:
                raise ValueError("column-control checkpoint density shape mismatch")
            history_length = np.asarray(payload["times_s"]).size
            if not (
                len(column_control_amplitudes)
                == np.asarray(payload["baseline_amplitudes_m3"]).size
                == np.asarray(payload["coupled_amplitudes_m3"]).size
                == history_length
            ):
                raise ValueError("three-track checkpoint history lengths differ")
        return (
            float(payload["time_s"]),
            int(payload["step"]),
            baseline,
            coupled,
            np.asarray(payload["times_s"], dtype=np.float64).tolist(),
            np.asarray(
                payload["baseline_amplitudes_m3"], dtype=np.float64
            ).tolist(),
            np.asarray(
                payload["coupled_amplitudes_m3"], dtype=np.float64
            ).tolist(),
            column_control_density,
            column_control_amplitudes,
            (
                float(payload["current_dt_s"])
                if schema_version >= 2
                else None
            ),
            int(payload["dt_reductions"]) if schema_version >= 2 else 0,
        )


def _save_failed_trial(
    path: Path,
    *,
    time_s: float,
    step: int,
    baseline: DimensionalEMState,
    coupled: DimensionalEMState,
    column_control_density_m3: np.ndarray | None = None,
) -> None:
    """Archive a rejected candidate without replacing the stable checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    if column_control_density_m3 is None:
        np.savez_compressed(
            temporary,
            time_s=np.float64(time_s),
            step=np.int64(step),
            baseline_density_m3=baseline.density_m3,
            baseline_w_plus_v=baseline.alfven.w_plus_v,
            baseline_w_minus_v=baseline.alfven.w_minus_v,
            coupled_density_m3=coupled.density_m3,
            coupled_w_plus_v=coupled.alfven.w_plus_v,
            coupled_w_minus_v=coupled.alfven.w_minus_v,
        )
    else:
        np.savez_compressed(
            temporary,
            time_s=np.float64(time_s),
            step=np.int64(step),
            baseline_density_m3=baseline.density_m3,
            baseline_w_plus_v=baseline.alfven.w_plus_v,
            baseline_w_minus_v=baseline.alfven.w_minus_v,
            column_control_density_m3=np.asarray(
                column_control_density_m3, dtype=np.float64
            ),
            coupled_density_m3=coupled.density_m3,
            coupled_w_plus_v=coupled.alfven.w_plus_v,
            coupled_w_minus_v=coupled.alfven.w_minus_v,
        )
    temporary.replace(path)


def _parallel_averaging_geometry(
    li: SimulationConfig, em: EMGrowthConfig
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    """Build independent particle/collision column weights and metadata."""

    if em.fac_source_model != "symmetric_flux_tube":
        return None, None, None
    if em.parallel_averaging_profile == "constant":
        length = float(em.parallel_averaging_length_km)
        return None, None, {
            "model": "constant",
            "minimum_length_km": length,
            "maximum_length_km": length,
            "first_length_km": length,
            "last_length_km": length,
        }
    profile = dipole_chapman_column_profile(
        li.y_coordinates_m,
        chapman_peak_altitude_m=li.chapman_peak_m,
        chapman_scale_height_m=li.chapman_scale_height_m,
        chapman_peak_density_m3=li.peak_density_m3,
        collision_nu0_s=li.collision_nu0_s,
        collision_frequency_scale_height_m=(
            li.collision_frequency_scale_height_m
        ),
        earth_radius_m=em.dipole_earth_radius_km * 1000.0,
        lower_altitude_m=em.dipole_column_lower_altitude_km * 1000.0,
        quadrature_order=em.dipole_quadrature_order,
    )
    metadata = profile.summary()
    metadata["altitude_km"] = (
        np.asarray(profile.apex_altitude_m) / 1000.0
    ).tolist()
    metadata["gravity_column_weight_km"] = (
        np.asarray(profile.gravity_column_weight_m) / 1000.0
    ).tolist()
    metadata["collision_column_weight_km"] = (
        np.asarray(profile.collision_column_weight_m) / 1000.0
    ).tolist()
    metadata["background_density_column_m2"] = np.asarray(
        profile.background_density_column_m2
    ).tolist()
    metadata["background_collision_column_m2_s"] = np.asarray(
        profile.background_collision_column_m2_s
    ).tolist()
    metadata["collision_moment_m_s"] = np.asarray(
        profile.collision_moment_m_s
    ).tolist()
    metadata["effective_collision_frequency_s"] = np.asarray(
        profile.effective_collision_frequency_s
    ).tolist()
    return (
        np.asarray(profile.gravity_column_weight_m, dtype=np.float64),
        np.asarray(profile.collision_column_weight_m, dtype=np.float64),
        metadata,
    )


def _build_models(li: SimulationConfig, em: EMGrowthConfig):
    if em.periodic_x_layout == "unique_periodic_cells":
        li.validate()
        # Bounds name the stored coordinates, not a duplicate closing node.
        # Require the deterministic cosine seed to fit the full periodic ring.
        periods = li.nx * li.dx_m / li.domain_width_m
        if not np.isfinite(periods) or periods < 1.0 or not np.isclose(
            periods, round(periods), rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("unique periodic grid requires integer seed wavelengths in nx * dx")
    if li.potential_boundary_y != BEST_BASELINE_POTENTIAL_BOUNDARY:
        raise ValueError("EM growth requires equal-extrapolated potential boundaries")
    if em.closure == "mimetic":
        li = li.with_updates(
            potential_operator=MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
            potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        )
        closure_type = DimensionalFACMimeticClosure
    else:
        li = li.with_updates(
            potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
            potential_boundary_y=BEST_BASELINE_POTENTIAL_BOUNDARY,
        )
        closure_type = DimensionalFACPointCenteredClosure

    (
        gravity_column_weight_m,
        collision_column_weight_m,
        _,
    ) = _parallel_averaging_geometry(li, em)
    ion_mass = em.ion_mass_amu * ATOMIC_MASS_KG
    if em.use_pardiso:
        legacy = ESFModel(li)
        closure = closure_type(
            li,
            background_field_t=em.background_field_t,
            ion_mass_kg=ion_mass,
            unforced_solver=legacy.potential_solver,
            gravity_column_weight_m=gravity_column_weight_m,
            collision_column_weight_m=collision_column_weight_m,
        )
    else:
        closure = closure_type(
            li,
            background_field_t=em.background_field_t,
            ion_mass_kg=ion_mass,
            gravity_column_weight_m=gravity_column_weight_m,
            collision_column_weight_m=collision_column_weight_m,
        )
        legacy = _model_impl.ESFModel.__new__(_model_impl.ESFModel)
        legacy.config = li
        legacy.potential_solver = closure

    basis = ModalBasis.low_wavenumber(
        nx=li.nx,
        ny=li.ny,
        dx_m=li.dx_m,
        dy_m=li.dy_m,
        max_x_wavenumber=em.max_x_wavenumber,
        max_y_wavenumber=em.max_y_wavenumber,
        include_sine=em.include_sine_modes,
    )
    line = ModalAlfvenLine(
        length_m=em.field_line_length_km * 1000.0,
        cells=em.field_line_cells,
        alfven_speed_m_s=em.alfven_speed_km_s * 1000.0,
        mode_count=basis.mode_count,
        spatial_order=em.line_spatial_order,
    )
    baseline = DimensionalBidirectionalModel(
        legacy,
        closure,
        basis,
        line,
        PhysicalFeedbackConfig(gain=0.0, background_exb_y_m_s=em.background_exb_y_m_s),
        right_boundary_voltage_v=em.right_boundary_voltage_v,
        right_boundary_model=em.right_boundary_model,
        right_boundary_conductance_s=em.right_boundary_conductance_s,
    )
    coupled = DimensionalBidirectionalModel(
        legacy,
        closure,
        basis,
        line,
        PhysicalFeedbackConfig(
            gain=em.feedback_gain,
            background_exb_y_m_s=em.background_exb_y_m_s,
            ramp_time_s=em.feedback_ramp_time_s,
            solver=em.coupling_solver,
            under_relaxation=em.coupling_under_relaxation,
            max_iterations=em.coupling_max_iterations,
            relative_tolerance=em.coupling_relative_tolerance,
            line_cfl_limit=em.line_cfl_limit,
            fac_source_model=em.fac_source_model,
            parallel_averaging_length_m=(
                None
                if (
                    em.parallel_averaging_length_km is None
                    or gravity_column_weight_m is not None
                )
                else em.parallel_averaging_length_km * 1000.0
            ),
            gravity_column_weight_m=gravity_column_weight_m,
            collision_column_weight_m=collision_column_weight_m,
            hemisphere_count=em.hemisphere_count,
        ),
        right_boundary_voltage_v=em.right_boundary_voltage_v,
        right_boundary_model=em.right_boundary_model,
        right_boundary_conductance_s=em.right_boundary_conductance_s,
    )
    return li, baseline, coupled, basis, line


def _growth_rate(times: list[float], amplitudes: list[float], start_s: float) -> float:
    time = np.asarray(times, dtype=np.float64)
    amplitude = np.asarray(amplitudes, dtype=np.float64)
    selected = (time >= start_s) & (amplitude > 0.0)
    if np.count_nonzero(selected) < 2:
        return float("nan")
    slope, _ = np.polyfit(time[selected], np.log(amplitude[selected]), 1)
    return float(slope)


def run_em_growth_experiment(
    li_config: SimulationConfig,
    em_config: EMGrowthConfig,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run paired trajectories and return an auditable growth report."""

    li, baseline_model, coupled_model, basis, line = _build_models(
        li_config, em_config
    )
    output = Path(em_config.output_dir)
    checkpoint_path = output / "checkpoint.npz"
    diagnostics_path = output / "diagnostics.jsonl"
    timestep_events_path = output / "timestep_events.jsonl"
    status_path = output / "status.json"
    config_hash = _configuration_hash(li, em_config)
    legacy_v1_hash = _configuration_hash(li, em_config, legacy_v1=True)

    three_track = em_config.comparison_tracks == "local_column_em"
    resuming_checkpoint = bool(resume and checkpoint_path.exists())
    if resuming_checkpoint:
        (
            time_s,
            step,
            baseline_state,
            coupled_state,
            times,
            baseline_amplitudes,
            coupled_amplitudes,
            column_control_density,
            column_control_amplitudes,
            loaded_current_dt_s,
            dt_reductions,
        ) = _load_checkpoint(
            checkpoint_path,
            config_hash,
            legacy_v1_hash,
            comparison_tracks=em_config.comparison_tracks,
        )
        _truncate_diagnostics_after_step(diagnostics_path, step)
        _truncate_diagnostics_after_step(timestep_events_path, step)
    else:
        # ``--no-resume`` is an explicit fresh start.  Remove only the runner's
        # own terminal artifacts so a failed replacement run cannot leave a
        # stale completed report or final state beside its new paused status.
        for stale in (
            checkpoint_path,
            output / "final_states.npz",
            output / "em_growth_report.json",
            output / "failed_trial.npz",
        ):
            if stale.exists():
                stale.unlink()
        density = initial_density(li)
        if em_config.initial_line_state == "zero":
            zeros = np.zeros((line.cells, basis.mode_count), dtype=np.float64)
            line_state = ModalAlfvenState(zeros.copy(), zeros.copy())
        else:
            line_state = coupled_model.initial_alfven_state_from_li_potential(
                density
            )
        if em_config.initial_characteristic_seed_v != 0.0:
            if not 0 <= em_config.initial_seed_mode < basis.mode_count:
                raise ValueError("initial_seed_mode is outside the retained basis")
            plus = line_state.w_plus_v.copy()
            plus[0, em_config.initial_seed_mode] += (
                em_config.initial_characteristic_seed_v
            )
            line_state = ModalAlfvenState(plus, line_state.w_minus_v.copy())
        baseline_state = DimensionalEMState(density.copy(), line_state)
        coupled_state = DimensionalEMState(density.copy(), line_state)
        column_control_density = density.copy() if three_track else None
        time_s = 0.0
        step = 0
        if diagnostics_path.exists():
            diagnostics_path.unlink()
        if timestep_events_path.exists():
            timestep_events_path.unlink()
        times = [time_s]
        baseline_amplitudes = [density_mode_amplitude(baseline_state.density_m3)]
        coupled_amplitudes = [density_mode_amplitude(coupled_state.density_m3)]
        column_control_amplitudes = (
            [density_mode_amplitude(column_control_density)]
            if column_control_density is not None
            else None
        )
        loaded_current_dt_s = None
        dt_reductions = 0

    (
        baseline_subcycling_events,
        baseline_extra_substeps,
    ) = _baseline_subcycling_totals(timestep_events_path)
    (
        column_control_subcycling_events,
        column_control_extra_substeps,
    ) = _column_subcycling_totals(timestep_events_path)
    line_dt_limit_s = em_config.line_cfl_limit * line.maximum_stable_dt_s
    current_dt_s = min(
        em_config.dt_s,
        line_dt_limit_s,
        (
            loaded_current_dt_s
            if loaded_current_dt_s is not None
            else em_config.dt_s
        ),
    )
    if not np.isfinite(current_dt_s) or current_dt_s <= 0.0:
        raise ValueError("checkpoint current_dt_s must be positive and finite")

    cell_area_m2 = li.dx_m * li.dy_m
    reference_density = initial_density(li)
    baseline_particle_reference = float(
        np.sum(reference_density, dtype=np.float64) * cell_area_m2
    )
    if getattr(coupled_model, "column_integrated", False):
        coupled_particle_weight = (
            coupled_model.gravity_column_weight_field_m
        )
    else:
        coupled_particle_weight = 1.0
    coupled_particle_reference = float(
        np.sum(
            coupled_particle_weight * reference_density,
            dtype=np.float64,
        )
        * cell_area_m2
    )
    column_control_particle_reference = coupled_particle_reference

    snapshot_records: list[dict[str, Any]] = []
    snapshot_interval = em_config.snapshot_interval_s
    if not resuming_checkpoint:
        _clear_snapshot_outputs(output)
    if snapshot_interval is not None:
        if resuming_checkpoint:
            snapshot_records = _reconcile_snapshots(
                output,
                checkpoint_time_s=time_s,
                interval_s=snapshot_interval,
                configuration_hash=config_hash,
                comparison_tracks=em_config.comparison_tracks,
            )
        else:
            snapshot_records.append(
                _save_density_snapshot(
                    _snapshot_file(output / "snapshots", 0),
                    index=0,
                    time_s=0.0,
                    step=0,
                    baseline=baseline_state,
                    coupled=coupled_state,
                    configuration_hash=config_hash,
                    column_control_density_m3=column_control_density,
                )
            )
            _write_snapshot_manifest(
                output / "snapshot_manifest.json",
                configuration_hash=config_hash,
                interval_s=snapshot_interval,
                records=snapshot_records,
                comparison_tracks=em_config.comparison_tracks,
            )
        next_snapshot_index = len(snapshot_records)
        next_snapshot_time_s = float(next_snapshot_index * snapshot_interval)
    else:
        next_snapshot_index = 0
        next_snapshot_time_s = float("inf")
    next_checkpoint = (
        (np.floor(time_s / em_config.checkpoint_interval_s) + 1.0)
        * em_config.checkpoint_interval_s
    )
    _write_json(
        status_path,
        {
            "status": "running",
            "time_s": time_s,
            "step": step,
            "configuration_hash": config_hash,
            "current_dt_s": current_dt_s,
            "dt_reductions": dt_reductions,
            "baseline_subcycling_events": baseline_subcycling_events,
            "baseline_extra_substeps": baseline_extra_substeps,
            **(
                {
                    "column_control_subcycling_events": (
                        column_control_subcycling_events
                    ),
                    "column_control_extra_substeps": (
                        column_control_extra_substeps
                    ),
                }
                if three_track
                else {}
            ),
            "snapshot_count": len(snapshot_records),
        },
    )
    def write_current_diagnostics(diagnostic_sampling: str) -> None:
        baseline_speed_m_s = max(
            baseline_diag.electrostatic.maximum_speed_x,
            baseline_diag.electrostatic.maximum_speed_y,
        )
        coupled_speed_m_s = max(
            coupled_diag.electrostatic.maximum_speed_x,
            coupled_diag.electrostatic.maximum_speed_y,
        )
        fac_to_gravity_ratio = (
            coupled_diag.fac_closure.fac_to_gravity_ratio
            if coupled_diag.fac_closure is not None
            else 0.0
        )
        baseline_particle_content = float(
            np.sum(baseline_state.density_m3, dtype=np.float64)
            * cell_area_m2
        )
        coupled_particle_content = float(
            np.sum(
                coupled_particle_weight * coupled_state.density_m3,
                dtype=np.float64,
            )
            * cell_area_m2
        )
        column_control_particle_content = None
        if column_control_density is not None:
            column_control_particle_content = float(
                np.sum(
                    coupled_particle_weight * column_control_density,
                    dtype=np.float64,
                )
                * cell_area_m2
            )
        _write_jsonl(
            diagnostics_path,
            {
                "step": step,
                "diagnostic_sampling": diagnostic_sampling,
                "time_s": time_s,
                "dt_s": dt,
                "current_dt_s": current_dt_s,
                "dt_reductions": dt_reductions,
                "baseline_amplitude_m3": baseline_amplitudes[-1],
                "coupled_amplitude_m3": coupled_amplitudes[-1],
                "minimum_baseline_density_m3": float(
                    np.min(baseline_state.density_m3)
                ),
                "minimum_coupled_density_m3": float(
                    np.min(coupled_state.density_m3)
                ),
                "baseline_cfl": baseline_advance.maximum_substep_cfl,
                "baseline_macro_cfl_without_subcycling": (
                    baseline_diag.combined_cfl_rate_s_inv * dt
                ),
                "baseline_substeps": baseline_advance.substeps,
                "baseline_substep_dt_s": baseline_advance.substep_dt_s,
                "coupled_cfl": coupled_diag.combined_cfl_rate_s_inv * dt,
                "maximum_baseline_speed_m_s": baseline_speed_m_s,
                "maximum_coupled_speed_m_s": coupled_speed_m_s,
                "baseline_potential_relative_residual": (
                    baseline_diag.electrostatic.potential.relative_residual
                ),
                "coupled_potential_relative_residual": (
                    coupled_diag.electrostatic.potential.relative_residual
                ),
                "fac_to_gravity_ratio": fac_to_gravity_ratio,
                "fac_forcing_representation": (
                    getattr(
                        coupled_diag,
                        "fac_forcing_representation",
                        "unknown",
                    )
                ),
                "fac_forcing_units": getattr(
                    coupled_diag, "fac_forcing_units", "unknown"
                ),
                "maximum_abs_fac_forcing": (
                    getattr(
                        coupled_diag, "maximum_abs_fac_forcing", 0.0
                    )
                ),
                "net_fac_forcing_integral": (
                    getattr(
                        coupled_diag, "net_fac_forcing_integral", 0.0
                    )
                ),
                "coupled_density_conservation_measure": (
                    getattr(
                        coupled_diag,
                        "density_conservation_measure",
                        "unknown",
                    )
                ),
                "baseline_local_density_content": (
                    baseline_particle_content
                ),
                "baseline_local_density_content_relative_change": (
                    baseline_particle_content
                    / baseline_particle_reference
                    - 1.0
                ),
                "coupled_conserved_density_content": (
                    coupled_particle_content
                ),
                "coupled_conserved_density_content_relative_change": (
                    coupled_particle_content
                    / coupled_particle_reference
                    - 1.0
                ),
                "effective_gain": coupled_diag.effective_gain,
                "activation_fraction": coupled_diag.activation_fraction,
                "coupling_iterations": coupled_diag.iterations,
                "coupling_relative_residual": (
                    coupled_diag.relative_coupling_residual
                ),
                "net_d_j_parallel_ds_a_m": (
                    coupled_diag.net_d_j_parallel_ds_a_m
                ),
                "relative_power_pairing_defect": (
                    coupled_diag.relative_power_pairing_defect
                ),
                "boundary_power_modal_w": coupled_diag.boundary_power_modal_w,
                "boundary_power_grid_w": coupled_diag.boundary_power_grid_w,
                "remote_boundary_power_modal_w": (
                    coupled_diag.remote_boundary_power_modal_w
                ),
                "remote_boundary_reflection_coefficient": (
                    coupled_diag.remote_boundary_reflection_coefficient
                ),
                **(
                    {
                        "column_control_amplitude_m3": (
                            column_control_amplitudes[-1]
                        ),
                        "minimum_column_control_density_m3": float(
                            np.min(column_control_density)
                        ),
                        "column_control_cfl": (
                            column_control_advance.maximum_substep_cfl
                        ),
                        "column_control_macro_cfl_without_subcycling": (
                            column_control_advance.diagnostics.cfl_rate * dt
                        ),
                        "column_control_substeps": (
                            column_control_advance.substeps
                        ),
                        "column_control_substep_dt_s": (
                            column_control_advance.substep_dt_s
                        ),
                        "maximum_column_control_speed_m_s": max(
                            column_control_advance.diagnostics.maximum_speed_x,
                            column_control_advance.diagnostics.maximum_speed_y,
                        ),
                        "column_control_potential_relative_residual": (
                            column_control_advance.diagnostics.potential.relative_residual
                        ),
                        "column_control_conserved_density_content": (
                            column_control_particle_content
                        ),
                        "column_control_conserved_density_content_relative_change": (
                            column_control_particle_content
                            / column_control_particle_reference
                            - 1.0
                        ),
                        "column_control_density_conservation_measure": (
                            "gravity_column_weight_times_density"
                        ),
                    }
                    if three_track
                    else {}
                ),
            },
        )

    while time_s < em_config.t_end_s - 1.0e-14:
        remaining_s = em_config.t_end_s - time_s
        dt = min(current_dt_s, remaining_s)
        if next_snapshot_time_s <= em_config.t_end_s + 1.0e-14:
            dt = min(dt, next_snapshot_time_s - time_s)
        try:
            baseline_advance = _advance_baseline_with_subcycling(
                baseline_model,
                time_s=time_s,
                state=baseline_state,
                macro_dt_s=dt,
                adaptive=em_config.adaptive_dt_halving,
                minimum_dt_s=em_config.minimum_dt_s,
            )
            next_baseline = baseline_advance.state
            baseline_diag = baseline_advance.diagnostics
            column_control_advance = None
            next_column_control_density = column_control_density
            if three_track:
                assert column_control_density is not None
                column_control_advance = _advance_column_control_with_subcycling(
                    coupled_model,
                    time_s=time_s,
                    density_m3=column_control_density,
                    macro_dt_s=dt,
                    adaptive=em_config.adaptive_dt_halving,
                    minimum_dt_s=em_config.minimum_dt_s,
                )
                next_column_control_density = (
                    column_control_advance.density_m3
                )
            next_coupled, coupled_diag = coupled_model.ssprk3_step(
                time_s, coupled_state, dt
            )
        except MinimumTimeStepError as error:
            _save_checkpoint(
                checkpoint_path,
                time_s=time_s,
                step=step,
                baseline=baseline_state,
                coupled=coupled_state,
                configuration_hash=config_hash,
                times_s=times,
                baseline_amplitudes_m3=baseline_amplitudes,
                coupled_amplitudes_m3=coupled_amplitudes,
                current_dt_s=current_dt_s,
                dt_reductions=dt_reductions,
                column_control_density_m3=column_control_density,
                column_control_amplitudes_m3=column_control_amplitudes,
            )
            _write_json(
                status_path,
                {
                    "status": "paused_minimum_dt",
                    "time_s": time_s,
                    "step": step,
                    "configuration_hash": config_hash,
                    "current_dt_s": current_dt_s,
                    "rejected_dt_s": error.dt_s,
                    "minimum_dt_s": em_config.minimum_dt_s,
                    "dt_reductions": dt_reductions,
                    "baseline_subcycling_events": baseline_subcycling_events,
                    "baseline_extra_substeps": baseline_extra_substeps,
                    **(
                        {
                            "column_control_subcycling_events": (
                                column_control_subcycling_events
                            ),
                            "column_control_extra_substeps": (
                                column_control_extra_substeps
                            ),
                        }
                        if three_track
                        else {}
                    ),
                    "failed_component": error.component,
                    "error_type": type(error.cause).__name__,
                    "error": str(error.cause),
                },
            )
            raise
        except (
            _model_impl.CFLViolation,
            PhysicalCouplingConvergenceError,
        ) as error:
            reduction_tolerance = 64.0 * np.finfo(float).eps * max(1.0, dt)
            can_reduce = (
                em_config.adaptive_dt_halving
                and dt > em_config.minimum_dt_s + reduction_tolerance
            )
            if can_reduce:
                previous_dt_s = current_dt_s
                current_dt_s = min(
                    current_dt_s,
                    max(em_config.minimum_dt_s, 0.5 * dt),
                )
                dt_reductions += 1
                _save_checkpoint(
                    checkpoint_path,
                    time_s=time_s,
                    step=step,
                    baseline=baseline_state,
                    coupled=coupled_state,
                    configuration_hash=config_hash,
                    times_s=times,
                    baseline_amplitudes_m3=baseline_amplitudes,
                    coupled_amplitudes_m3=coupled_amplitudes,
                    current_dt_s=current_dt_s,
                    dt_reductions=dt_reductions,
                    column_control_density_m3=column_control_density,
                    column_control_amplitudes_m3=column_control_amplitudes,
                )
                _write_jsonl(
                    timestep_events_path,
                    {
                        "event": "dt_halved_after_rejected_trial",
                        "step": step,
                        "time_s": time_s,
                        "rejected_dt_s": dt,
                        "previous_current_dt_s": previous_dt_s,
                        "new_current_dt_s": current_dt_s,
                        "dt_reductions": dt_reductions,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                _write_json(
                    status_path,
                    {
                        "status": "running",
                        "time_s": time_s,
                        "step": step,
                        "configuration_hash": config_hash,
                        "current_dt_s": current_dt_s,
                        "dt_reductions": dt_reductions,
                        "baseline_subcycling_events": baseline_subcycling_events,
                        "baseline_extra_substeps": baseline_extra_substeps,
                        **(
                            {
                                "column_control_subcycling_events": (
                                    column_control_subcycling_events
                                ),
                                "column_control_extra_substeps": (
                                    column_control_extra_substeps
                                ),
                            }
                            if three_track
                            else {}
                        ),
                        "snapshot_count": len(snapshot_records),
                        "last_rejected_error_type": type(error).__name__,
                    },
                )
                continue

            minimum_error = MinimumTimeStepError(time_s, dt, error)
            _save_checkpoint(
                checkpoint_path,
                time_s=time_s,
                step=step,
                baseline=baseline_state,
                coupled=coupled_state,
                configuration_hash=config_hash,
                times_s=times,
                baseline_amplitudes_m3=baseline_amplitudes,
                coupled_amplitudes_m3=coupled_amplitudes,
                current_dt_s=current_dt_s,
                dt_reductions=dt_reductions,
                column_control_density_m3=column_control_density,
                column_control_amplitudes_m3=column_control_amplitudes,
            )
            _write_json(
                status_path,
                {
                    "status": (
                        "paused_minimum_dt"
                        if em_config.adaptive_dt_halving
                        else "failed"
                    ),
                    "time_s": time_s,
                    "step": step,
                    "configuration_hash": config_hash,
                    "current_dt_s": current_dt_s,
                    "rejected_dt_s": dt,
                    "minimum_dt_s": em_config.minimum_dt_s,
                    "dt_reductions": dt_reductions,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if em_config.adaptive_dt_halving:
                raise minimum_error from error
            raise
        except Exception as error:
            _save_checkpoint(
                checkpoint_path,
                time_s=time_s,
                step=step,
                baseline=baseline_state,
                coupled=coupled_state,
                configuration_hash=config_hash,
                times_s=times,
                baseline_amplitudes_m3=baseline_amplitudes,
                coupled_amplitudes_m3=coupled_amplitudes,
                current_dt_s=current_dt_s,
                dt_reductions=dt_reductions,
                column_control_density_m3=column_control_density,
                column_control_amplitudes_m3=column_control_amplitudes,
            )
            _write_json(
                status_path,
                {
                    "status": "failed",
                    "time_s": time_s,
                    "step": step,
                    "configuration_hash": config_hash,
                    "current_dt_s": current_dt_s,
                    "dt_reductions": dt_reductions,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        if (
            not np.all(np.isfinite(next_baseline.density_m3))
            or not np.all(np.isfinite(next_coupled.density_m3))
            or np.min(next_baseline.density_m3) <= 0.0
            or np.min(next_coupled.density_m3) <= 0.0
            or (
                next_column_control_density is not None
                and (
                    not np.all(np.isfinite(next_column_control_density))
                    or np.min(next_column_control_density) <= 0.0
                )
            )
        ):
            # Keep checkpoint.npz at the last accepted state.  The rejected
            # candidate is useful for diagnosis but must never be resumed.
            _save_checkpoint(
                checkpoint_path,
                time_s=time_s,
                step=step,
                baseline=baseline_state,
                coupled=coupled_state,
                configuration_hash=config_hash,
                times_s=times,
                baseline_amplitudes_m3=baseline_amplitudes,
                coupled_amplitudes_m3=coupled_amplitudes,
                current_dt_s=current_dt_s,
                dt_reductions=dt_reductions,
                column_control_density_m3=column_control_density,
                column_control_amplitudes_m3=column_control_amplitudes,
            )
            _save_failed_trial(
                output / "failed_trial.npz",
                time_s=time_s + dt,
                step=step + 1,
                baseline=next_baseline,
                coupled=next_coupled,
                column_control_density_m3=next_column_control_density,
            )
            _write_json(
                status_path,
                {
                    "status": "failed",
                    "time_s": time_s,
                    "step": step,
                    "failed_trial_time_s": time_s + dt,
                    "failed_trial_step": step + 1,
                    "configuration_hash": config_hash,
                    "current_dt_s": current_dt_s,
                    "dt_reductions": dt_reductions,
                    "error_type": "DensityPositivityError",
                    "error": "non-positive or non-finite density in EM growth run",
                },
            )
            raise RuntimeError("non-positive or non-finite density in EM growth run")
        baseline_state = next_baseline
        coupled_state = next_coupled
        column_control_density = next_column_control_density
        time_s += dt
        if np.isclose(
            time_s,
            next_snapshot_time_s,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps * max(1.0, abs(time_s)),
        ):
            time_s = next_snapshot_time_s
        step += 1
        if baseline_advance.substeps > 1:
            baseline_subcycling_events += 1
            baseline_extra_substeps += baseline_advance.substeps - 1
            _write_jsonl(
                timestep_events_path,
                {
                    "event": "baseline_subcycling",
                    "step": step,
                    "time_s": time_s,
                    "macro_dt_s": dt,
                    "baseline_substeps": baseline_advance.substeps,
                    "baseline_substep_dt_s": baseline_advance.substep_dt_s,
                    "maximum_baseline_substep_cfl": (
                        baseline_advance.maximum_substep_cfl
                    ),
                    "baseline_subcycling_events": baseline_subcycling_events,
                    "baseline_extra_substeps": baseline_extra_substeps,
                },
            )
        if (
            column_control_advance is not None
            and column_control_advance.substeps > 1
        ):
            column_control_subcycling_events += 1
            column_control_extra_substeps += (
                column_control_advance.substeps - 1
            )
            _write_jsonl(
                timestep_events_path,
                {
                    "event": "column_control_subcycling",
                    "step": step,
                    "time_s": time_s,
                    "macro_dt_s": dt,
                    "column_control_substeps": (
                        column_control_advance.substeps
                    ),
                    "column_control_substep_dt_s": (
                        column_control_advance.substep_dt_s
                    ),
                    "maximum_column_control_substep_cfl": (
                        column_control_advance.maximum_substep_cfl
                    ),
                    "column_control_subcycling_events": (
                        column_control_subcycling_events
                    ),
                    "column_control_extra_substeps": (
                        column_control_extra_substeps
                    ),
                },
            )
        times.append(time_s)
        baseline_amplitudes.append(
            density_mode_amplitude(baseline_state.density_m3)
        )
        coupled_amplitudes.append(
            density_mode_amplitude(coupled_state.density_m3)
        )
        if column_control_amplitudes is not None:
            assert column_control_density is not None
            column_control_amplitudes.append(
                density_mode_amplitude(column_control_density)
            )
        if (
            step % em_config.diagnostics_interval_steps == 0
            or time_s >= em_config.t_end_s - 1.0e-14
        ):
            write_current_diagnostics("accepted_step_representative_stage")
        if (
            snapshot_interval is not None
            and time_s == next_snapshot_time_s
            and next_snapshot_time_s <= em_config.t_end_s + 1.0e-14
        ):
            snapshot_records.append(
                _save_density_snapshot(
                    _snapshot_file(output / "snapshots", next_snapshot_index),
                    index=next_snapshot_index,
                    time_s=time_s,
                    step=step,
                    baseline=baseline_state,
                    coupled=coupled_state,
                    configuration_hash=config_hash,
                    column_control_density_m3=column_control_density,
                )
            )
            _write_snapshot_manifest(
                output / "snapshot_manifest.json",
                configuration_hash=config_hash,
                interval_s=snapshot_interval,
                records=snapshot_records,
                comparison_tracks=em_config.comparison_tracks,
            )
            next_snapshot_index += 1
            next_snapshot_time_s = float(next_snapshot_index * snapshot_interval)
        if (
            time_s + 1.0e-12 >= next_checkpoint
            and time_s < em_config.t_end_s - 1.0e-12
        ):
            _save_checkpoint(
                checkpoint_path,
                time_s=time_s,
                step=step,
                baseline=baseline_state,
                coupled=coupled_state,
                configuration_hash=config_hash,
                times_s=times,
                baseline_amplitudes_m3=baseline_amplitudes,
                coupled_amplitudes_m3=coupled_amplitudes,
                current_dt_s=current_dt_s,
                dt_reductions=dt_reductions,
                column_control_density_m3=column_control_density,
                column_control_amplitudes_m3=column_control_amplitudes,
            )
            while time_s + 1.0e-12 >= next_checkpoint:
                next_checkpoint += em_config.checkpoint_interval_s

    # Older runners omitted terminal diagnostics when the final step was not
    # divisible by the diagnostic interval. Repair a completed checkpoint by
    # evaluating its RHS, never by advancing/replaying the accepted trajectory.
    # This endpoint sample is explicitly distinguished from historical RK-stage
    # samples; its CFL is instantaneous, not a recovered stage maximum.
    diagnostic_lines = (
        diagnostics_path.read_text(encoding="utf-8").splitlines()
        if diagnostics_path.exists() else []
    )
    diagnostic_records = [json.loads(line) for line in diagnostic_lines if line.strip()]
    if step > 0 and (
        not diagnostic_records or int(diagnostic_records[-1]["step"]) < step
    ):
        dt = float(times[-1] - times[-2])
        _, baseline_diag = baseline_model.evaluate_rhs(time_s, baseline_state)
        _, coupled_diag = coupled_model.evaluate_rhs(time_s, coupled_state)
        baseline_substeps = 1
        column_substeps = 1
        if timestep_events_path.exists():
            for event_line in timestep_events_path.read_text(encoding="utf-8").splitlines():
                if not event_line.strip():
                    continue
                event = json.loads(event_line)
                if int(event.get("step", -1)) != step:
                    continue
                if event.get("event") == "baseline_subcycling":
                    baseline_substeps = int(event["baseline_substeps"])
                elif event.get("event") == "column_control_subcycling":
                    column_substeps = int(event["column_control_substeps"])
        baseline_advance = _BaselineAdvance(
            state=baseline_state, diagnostics=baseline_diag,
            substeps=baseline_substeps, substep_dt_s=dt / baseline_substeps,
            maximum_substep_cfl=baseline_diag.combined_cfl_rate_s_inv * dt / baseline_substeps,
        )
        column_control_advance = None
        if column_control_density is not None:
            _, column_diag = coupled_model.evaluate_column_es_rhs(column_control_density)
            column_control_advance = _ColumnControlAdvance(
                density_m3=column_control_density, diagnostics=column_diag,
                substeps=column_substeps, substep_dt_s=dt / column_substeps,
                maximum_substep_cfl=column_diag.cfl_rate * dt / column_substeps,
            )
        write_current_diagnostics("reconstructed_terminal_state_rhs_not_stage_maximum")

    _save_checkpoint(
        checkpoint_path,
        time_s=time_s,
        step=step,
        baseline=baseline_state,
        coupled=coupled_state,
        configuration_hash=config_hash,
        times_s=times,
        baseline_amplitudes_m3=baseline_amplitudes,
        coupled_amplitudes_m3=coupled_amplitudes,
        current_dt_s=current_dt_s,
        dt_reductions=dt_reductions,
        column_control_density_m3=column_control_density,
        column_control_amplitudes_m3=column_control_amplitudes,
    )
    if three_track:
        assert column_control_density is not None
        np.savez_compressed(
            output / "final_states.npz",
            schema_version=np.int64(2),
            baseline_density_m3=baseline_state.density_m3,
            column_control_density_m3=column_control_density,
            coupled_density_m3=coupled_state.density_m3,
            coupled_w_plus_v=coupled_state.alfven.w_plus_v,
            coupled_w_minus_v=coupled_state.alfven.w_minus_v,
            time_s=time_s,
        )
    else:
        np.savez_compressed(
            output / "final_states.npz",
            baseline_density_m3=baseline_state.density_m3,
            coupled_density_m3=coupled_state.density_m3,
            coupled_w_plus_v=coupled_state.alfven.w_plus_v,
            coupled_w_minus_v=coupled_state.alfven.w_minus_v,
            time_s=time_s,
        )
    baseline_rate = _growth_rate(
        times, baseline_amplitudes, em_config.growth_fit_start_s
    )
    coupled_rate = _growth_rate(
        times, coupled_amplitudes, em_config.growth_fit_start_s
    )
    column_control_rate = (
        _growth_rate(
            times,
            column_control_amplitudes,
            em_config.growth_fit_start_s,
        )
        if column_control_amplitudes is not None
        else None
    )
    trajectory_difference = float(
        np.linalg.norm(coupled_state.density_m3 - baseline_state.density_m3)
        / max(np.linalg.norm(baseline_state.density_m3), np.finfo(float).tiny)
    )
    geometry_trajectory_difference = None
    em_trajectory_difference = None
    if column_control_density is not None:
        baseline_norm = max(
            float(np.linalg.norm(baseline_state.density_m3)),
            np.finfo(float).tiny,
        )
        column_norm = max(
            float(np.linalg.norm(column_control_density)),
            np.finfo(float).tiny,
        )
        geometry_trajectory_difference = float(
            np.linalg.norm(
                column_control_density - baseline_state.density_m3
            )
            / baseline_norm
        )
        em_trajectory_difference = float(
            np.linalg.norm(
                coupled_state.density_m3 - column_control_density
            )
            / column_norm
        )
    _, _, parallel_geometry = _parallel_averaging_geometry(li, em_config)
    archived_em_config = em_config.to_dict()
    if not three_track:
        archived_em_config.pop("comparison_tracks")
    report = {
        "format": (
            "esf_dimensional_em_growth_v2"
            if three_track
            else "esf_dimensional_em_growth_v1"
        ),
        "experiment_label": em_config.experiment_label,
        "configuration_hash": config_hash,
        "li_config": li.to_dict(),
        "em_config": archived_em_config,
        "scope": {
            "dimensional_fac_li_closure": True,
            "transverse_laplacian_fac": True,
            "two_way_nonlinear_density_feedback": True,
            "independent_es_baseline_subcycling": True,
            **(
                {"comparison_tracks": em_config.comparison_tracks}
                if three_track
                else {}
            ),
            "fac_source_model": em_config.fac_source_model,
            "parallel_averaging_length_km": (
                em_config.parallel_averaging_length_km
            ),
            "parallel_averaging_profile": (
                em_config.parallel_averaging_profile
            ),
            "parallel_averaging_geometry": parallel_geometry,
            "parallel_column_weighted_perpendicular_operator": (
                em_config.parallel_averaging_profile
                == "dipole_chapman_flux_tube"
            ),
            "parallel_column_coefficients_are_independent": (
                em_config.parallel_averaging_profile
                == "dipole_chapman_flux_tube"
            ),
            "parallel_column_reduction_version": (
                "two_moment_surface_fac_column_continuity_v2"
                if em_config.parallel_averaging_profile
                == "dipole_chapman_flux_tube"
                else "legacy_local_scalar_h"
            ),
            "fac_forcing_representation": (
                "surface_current"
                if em_config.parallel_averaging_profile
                == "dipole_chapman_flux_tube"
                else "volume_current_gradient"
            ),
            "coupled_density_conservation_measure": (
                "gravity_column_weight_times_density"
                if em_config.parallel_averaging_profile
                == "dipole_chapman_flux_tube"
                else "local_density"
            ),
            "baseline_density_conservation_measure": "local_density",
            **(
                {
                    "column_control_density_conservation_measure": (
                        "gravity_column_weight_times_density"
                    ),
                    "geometry_effect_isolated": True,
                    "electromagnetic_effect_isolated": True,
                }
                if three_track
                else {}
            ),
            "hemisphere_count": em_config.hemisphere_count,
            "remote_boundary_model": em_config.right_boundary_model,
            "remote_boundary_conductance_s": (
                em_config.resolved_remote_conductance_s
            ),
            "remote_boundary_reflection_coefficient": (
                em_config.remote_reflection_coefficient
            ),
            "modal_grid_power_adjoint_pairing": True,
            "coupled_energy_budget_verified": False,
            "background_exb_y_m_s": em_config.background_exb_y_m_s,
            "background_exb_model": "imposed_common_density_drift_no_background_current_closure",
            "common_mode_parallel_mass_transport": False,
            "two_hemisphere_metric_geometry": False,
        },
        "grid": [li.ny, li.nx, line.cells],
        "retained_transverse_modes": basis.mode_count,
        "completed_time_s": time_s,
        "steps": step,
        "final_current_dt_s": current_dt_s,
        "dt_reductions": dt_reductions,
        "baseline_subcycling": {
            "enabled": True,
            "activation": "baseline_density_cfl_only",
            "events": baseline_subcycling_events,
            "extra_baseline_steps": baseline_extra_substeps,
            "subcycled_macro_steps": baseline_subcycling_events,
        },
        **(
            {
                "column_control_subcycling": {
                    "enabled": True,
                    "activation": "column_control_density_cfl_only",
                    "events": column_control_subcycling_events,
                    "extra_column_control_steps": column_control_extra_substeps,
                    "subcycled_macro_steps": column_control_subcycling_events,
                }
            }
            if three_track
            else {}
        ),
        "initial_density_array_equal": True,
        "initial_line_state_array_equal": True,
        "initial_line_profile": em_config.initial_line_state,
        "baseline_growth_rate_s_inv": baseline_rate,
        "coupled_growth_rate_s_inv": coupled_rate,
        "growth_rate_difference_s_inv": coupled_rate - baseline_rate,
        **(
            {
                "column_control_growth_rate_s_inv": column_control_rate,
                "growth_effects_s_inv": {
                    "geometry_column_es_minus_local_es": (
                        column_control_rate - baseline_rate
                    ),
                    "electromagnetic_column_em_minus_column_es": (
                        coupled_rate - column_control_rate
                    ),
                    "total_column_em_minus_local_es": (
                        coupled_rate - baseline_rate
                    ),
                },
                "relative_final_density_effects": {
                    "geometry_column_es_minus_local_es": (
                        geometry_trajectory_difference
                    ),
                    "electromagnetic_column_em_minus_column_es": (
                        em_trajectory_difference
                    ),
                    "total_column_em_minus_local_es": trajectory_difference,
                },
            }
            if three_track
            else {}
        ),
        "relative_final_density_trajectory_difference": trajectory_difference,
        "minimum_baseline_density_m3": float(np.min(baseline_state.density_m3)),
        "minimum_coupled_density_m3": float(np.min(coupled_state.density_m3)),
        **(
            {
                "minimum_column_control_density_m3": float(
                    np.min(column_control_density)
                )
            }
            if three_track
            else {}
        ),
        "checkpoint": str(checkpoint_path),
        "final_states": str(output / "final_states.npz"),
        "diagnostics": str(diagnostics_path),
        "timestep_events": str(timestep_events_path),
        "snapshots": {
            "enabled": snapshot_interval is not None,
            "interval_s": snapshot_interval,
            "count": len(snapshot_records),
            "manifest": (
                str(output / "snapshot_manifest.json")
                if snapshot_interval is not None
                else None
            ),
        },
    }
    _write_json(output / "em_growth_report.json", report)
    _write_json(
        status_path,
        {
            "status": "completed",
            "time_s": time_s,
            "step": step,
            "configuration_hash": config_hash,
            "current_dt_s": current_dt_s,
            "dt_reductions": dt_reductions,
            "baseline_subcycling_events": baseline_subcycling_events,
            "baseline_extra_substeps": baseline_extra_substeps,
            **(
                {
                    "column_control_subcycling_events": (
                        column_control_subcycling_events
                    ),
                    "column_control_extra_substeps": (
                        column_control_extra_substeps
                    ),
                }
                if three_track
                else {}
            ),
            "snapshot_count": len(snapshot_records),
            "report": str(output / "em_growth_report.json"),
        },
    )
    return report


__all__ = [
    "EMGrowthConfig",
    "MinimumTimeStepError",
    "run_em_growth_experiment",
]
