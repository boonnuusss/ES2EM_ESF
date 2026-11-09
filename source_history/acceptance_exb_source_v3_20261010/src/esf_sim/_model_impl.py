"""Time integration and checkpointed execution of the electrostatic ESF model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    BEST_BASELINE_POTENTIAL_OPERATOR,
    DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
    CONSERVATIVE_HARMONIC_OPERATOR,
    MIMETIC_HARMONIC_LOG_RHS_OPERATOR,
    EXPONENTIAL_CONSERVATIVE_OPERATOR,
    SimulationConfig,
)
from .numerics import advection_rhs
from .potential import PotentialDiagnostics, PotentialSolver


@dataclass(frozen=True)
class RhsDiagnostics:
    """Diagnostics produced by one complete right-hand-side evaluation."""

    cfl_rate: float
    maximum_speed_x: float
    maximum_speed_y: float
    potential: PotentialDiagnostics


class CFLViolation(RuntimeError):
    """Raised when a trial step violates the configured CFL ceiling."""


def chapman_profile(y_m: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """Return the plan-prescribed Chapman layer in SI units."""

    z = (y_m - config.chapman_peak_m) / config.chapman_scale_height_m
    return config.peak_density_m3 * np.exp(1.0 - z - np.exp(-z))


def initial_density(config: SimulationConfig) -> np.ndarray:
    """Construct the Chapman layer with the 10% fundamental-mode perturbation."""

    x = config.x_coordinates_m
    y = config.y_coordinates_m
    background = chapman_profile(y, config)[:, None]
    modulation = 1.0 + config.perturbation_amplitude * np.cos(
        2.0 * np.pi * x / config.domain_width_m
    )
    density = background * modulation[None, :]
    if config.relative_noise:
        rng = np.random.default_rng(config.random_seed)
        density *= 1.0 + config.relative_noise * rng.standard_normal(density.shape)
    return density


class ESFModel:
    """Spatially discretized ESF equations for one reconstruction scheme."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.x = config.x_coordinates_m
        self.y = config.y_coordinates_m
        self.collision_frequency = config.collision_nu0_s * np.exp(
            -self.y / config.collision_frequency_scale_height_m
        )
        if config.potential_operator == CONSERVATIVE_HARMONIC_OPERATOR:
            if config.potential_boundary_y != DIRICHLET_ZERO_POTENTIAL_BOUNDARY:
                raise ValueError(
                    "conservative_harmonic currently supports only "
                    "potential_boundary_y=dirichlet_zero"
                )
            self.potential_solver = PotentialSolver(
                dx=config.dx_m,
                dy=config.dy_m,
                gravity=config.gravity_m_s2,
                collision_frequency=self.collision_frequency,
                density_floor=config.density_floor_m3,
                rtol=config.potential_rtol,
                maxiter=config.potential_maxiter,
                refresh_steps=config.preconditioner_refresh_steps,
                iteration_limit=config.preconditioner_iteration_limit,
            )
        elif config.potential_operator == BEST_BASELINE_POTENTIAL_OPERATOR:
            from .point_centered_potential import (
                EqualExtrapolatedLogPointCenteredPotentialSolver,
                LogPointCenteredPotentialSolver,
            )

            if config.potential_boundary_y == BEST_BASELINE_POTENTIAL_BOUNDARY:
                self.potential_solver = (
                    EqualExtrapolatedLogPointCenteredPotentialSolver(config)
                )
            else:
                self.potential_solver = LogPointCenteredPotentialSolver(config)
        elif config.potential_operator == EXPONENTIAL_CONSERVATIVE_OPERATOR:
            if config.potential_boundary_y != BEST_BASELINE_POTENTIAL_BOUNDARY:
                raise ValueError(
                    "exponential_conservative requires "
                    "potential_boundary_y=equal_extrapolation"
                )
            from .stabilized_potential import ExponentialConservativePotentialSolver

            self.potential_solver = ExponentialConservativePotentialSolver(config)
        elif config.potential_operator == MIMETIC_HARMONIC_LOG_RHS_OPERATOR:
            if config.potential_boundary_y != BEST_BASELINE_POTENTIAL_BOUNDARY:
                raise ValueError(
                    "mimetic_harmonic_log_rhs requires "
                    "potential_boundary_y=equal_extrapolation"
                )
            from .mimetic_potential import HarmonicMimeticLogPotentialSolver

            self.potential_solver = HarmonicMimeticLogPotentialSolver(config)
        else:  # guarded by SimulationConfig validation
            raise ValueError(f"unsupported potential operator {config.potential_operator!r}")

    def velocity_from_potential(self, potential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ExB velocity components v=(-psi_y, psi_x)."""

        dx = self.config.dx_m
        dy = self.config.dy_m
        dpsi_dx = (np.roll(potential, -1, axis=1) - np.roll(potential, 1, axis=1)) / (
            2.0 * dx
        )
        if self.config.potential_boundary_y == BEST_BASELINE_POTENTIAL_BOUNDARY:
            padded_y = np.pad(potential, ((1, 1), (0, 0)), mode="edge")
            dpsi_dy = (padded_y[2:] - padded_y[:-2]) / (2.0 * dy)
        else:
            dpsi_dy = np.empty_like(potential)
            dpsi_dy[1:-1] = (potential[2:] - potential[:-2]) / (2.0 * dy)
            dpsi_dy[0] = (potential[1] - potential[0]) / dy
            dpsi_dy[-1] = (potential[-1] - potential[-2]) / dy
        return -dpsi_dy, dpsi_dx

    def evaluate_rhs(self, density: np.ndarray) -> tuple[np.ndarray, RhsDiagnostics]:
        """Solve the elliptic equation and evaluate conservative density advection."""

        potential, potential_diagnostics = self.potential_solver.solve(density)
        velocity_x, velocity_y = self.velocity_from_potential(potential)
        rhs = advection_rhs(
            density,
            velocity_x,
            velocity_y,
            dx=self.config.dx_m,
            dy=self.config.dy_m,
            scheme=self.config.scheme,
            pdm_alpha=self.config.pdm_alpha,
            periodic_y=False,
        )
        cfl_rate = float(
            np.max(
                np.abs(velocity_x) / self.config.dx_m
                + np.abs(velocity_y) / self.config.dy_m
            )
        )
        diagnostics = RhsDiagnostics(
            cfl_rate=cfl_rate,
            maximum_speed_x=float(np.max(np.abs(velocity_x))),
            maximum_speed_y=float(np.max(np.abs(velocity_y))),
            potential=potential_diagnostics,
        )
        return rhs, diagnostics

    def ssprk3_step(
        self, density: np.ndarray, dt_s: float
    ) -> tuple[np.ndarray, RhsDiagnostics]:
        """Advance one third-order SSP Runge-Kutta start-up step."""

        rhs0, diag0 = self.evaluate_rhs(density)
        _check_cfl(diag0.cfl_rate, dt_s, self.config.cfl_limit)
        u1 = density + dt_s * rhs0
        rhs1, diag1 = self.evaluate_rhs(u1)
        _check_cfl(diag1.cfl_rate, dt_s, self.config.cfl_limit)
        u2 = 0.75 * density + 0.25 * (u1 + dt_s * rhs1)
        rhs2, diag2 = self.evaluate_rhs(u2)
        _check_cfl(diag2.cfl_rate, dt_s, self.config.cfl_limit)
        result = (density + 2.0 * (u2 + dt_s * rhs2)) / 3.0
        return result, max((diag0, diag1, diag2), key=lambda d: d.cfl_rate)

    def ab3_predictor_corrector_step(
        self,
        density: np.ndarray,
        density_previous: np.ndarray,
        density_previous2: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, RhsDiagnostics]:
        """Advance one AB3 predictor-corrector step used by the paper."""

        predictor = (
            15.0 / 8.0 * density
            - 5.0 / 4.0 * density_previous
            + 3.0 / 8.0 * density_previous2
        )
        rhs, diagnostics = self.evaluate_rhs(predictor)
        _check_cfl(diagnostics.cfl_rate, dt_s, self.config.cfl_limit)
        return density + dt_s * rhs, diagnostics


def _check_cfl(cfl_rate: float, dt_s: float, cfl_limit: float) -> None:
    if cfl_rate * dt_s > cfl_limit:
        raise CFLViolation(
            f"CFL={cfl_rate * dt_s:.6g} exceeds configured limit {cfl_limit:.6g}"
        )


def _checkpoint_path(config: SimulationConfig) -> Path:
    return config.output_directory / "checkpoints" / f"{config.scheme}.npz"


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_checkpoint(
    path: Path,
    density_history: list[np.ndarray],
    *,
    time_s: float,
    dt_s: float,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        density=density_history[0],
        density_previous=density_history[1] if len(density_history) > 1 else np.array([]),
        density_previous2=density_history[2] if len(density_history) > 2 else np.array([]),
        history_length=len(density_history),
        time_s=time_s,
        dt_s=dt_s,
        step=step,
    )
    temporary.replace(path)


def _archive_stable_checkpoint(path: Path, time_s: float) -> Path:
    archive = path.parent / "stable" / f"{path.stem}_t{time_s:010.3f}.npz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, archive)
    return archive


def _load_checkpoint(path: Path) -> tuple[list[np.ndarray], float, float, int]:
    with np.load(path) as saved:
        history = [saved["density"]]
        history_length = int(saved["history_length"])
        if history_length > 1:
            history.append(saved["density_previous"])
        if history_length > 2:
            history.append(saved["density_previous2"])
        return history, float(saved["time_s"]), float(saved["dt_s"]), int(saved["step"])


def run_simulation(
    config: SimulationConfig,
    *,
    resume: bool = True,
    maximum_steps: int | None = None,
    frame_archive_directory: str | Path | None = None,
    frame_interval_s: float = 1.0,
    frame_chunk_size: int = 100,
) -> Path:
    """Run one configured scheme and return its final density snapshot path."""

    config.validate()
    simulation_directory = config.output_directory / "simulations"
    simulation_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(config)
    diagnostics_path = simulation_directory / f"{config.scheme}_diagnostics.jsonl"
    log_path = simulation_directory / f"{config.scheme}.log"
    logger = logging.getLogger(f"esf_sim.{config.scheme}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    if resume and checkpoint_path.exists():
        density_history, time_s, dt_s, step = _load_checkpoint(checkpoint_path)
        logger.info("Resumed checkpoint at t=%g s, step=%d, dt=%g s", time_s, step, dt_s)
    else:
        density_history = [initial_density(config)]
        time_s = 0.0
        dt_s = config.dt_s
        step = 0
        diagnostics_path.unlink(missing_ok=True)

    model = ESFModel(config)
    mass_reference = float(np.sum(density_history[0]) * config.dx_m * config.dy_m)
    frame_archive = None
    if frame_archive_directory is not None:
        from .timeseries import ChunkedFrameArchive

        frame_archive = ChunkedFrameArchive(
            frame_archive_directory,
            config,
            interval_s=frame_interval_s,
            chunk_frames=frame_chunk_size,
        )
        frame_archive.prepare(
            resume_time_s=time_s,
            density=density_history[0],
            resume=resume,
        )
    next_checkpoint_time = (
        math.floor(time_s / config.checkpoint_interval_s) + 1
    ) * config.checkpoint_interval_s

    while time_s < config.t_end_s - 1.0e-12:
        if maximum_steps is not None and step >= maximum_steps:
            break
        dt_step = min(dt_s, config.t_end_s - time_s)
        if frame_archive is not None and np.isfinite(frame_archive.next_time_s):
            time_to_frame = frame_archive.next_time_s - time_s
            if time_to_frame > 1.0e-12:
                dt_step = min(dt_step, time_to_frame)
        current = density_history[0]
        try:
            if len(density_history) < 3:
                updated, diagnostics = model.ssprk3_step(current, dt_step)
            else:
                updated, diagnostics = model.ab3_predictor_corrector_step(
                    current, density_history[1], density_history[2], dt_step
                )
        except CFLViolation as error:
            dt_s *= 0.5
            density_history = [current]
            logger.warning("%s; reducing dt to %g s and rebuilding AB3 history", error, dt_s)
            _save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            continue
        except RuntimeError:
            _save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            logger.exception("Elliptic solve failed; checkpoint saved")
            if frame_archive is not None:
                frame_archive.finalize(completed=False)
            raise

        minimum_density = float(np.min(updated))
        if minimum_density < -config.positivity_tolerance_m3:
            _save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            if frame_archive is not None:
                frame_archive.finalize(completed=False)
            raise FloatingPointError(
                f"negative density {minimum_density:.6e} m^-3 exceeds tolerance"
            )
        if minimum_density < 0.0:
            updated = np.maximum(updated, 0.0)

        time_s += dt_step
        step += 1
        density_history.insert(0, updated)
        del density_history[3:]
        if frame_archive is not None and math.isclose(
            time_s,
            frame_archive.next_time_s,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            frame_archive.capture(time_s, updated)
        mass = float(np.sum(updated) * config.dx_m * config.dy_m)
        record = {
            "step": step,
            "time_s": time_s,
            "dt_s": dt_step,
            "cfl": diagnostics.cfl_rate * dt_step,
            "maximum_speed_x_m_s": diagnostics.maximum_speed_x,
            "maximum_speed_y_m_s": diagnostics.maximum_speed_y,
            "potential_iterations": diagnostics.potential.iterations,
            "potential_relative_residual": diagnostics.potential.relative_residual,
            "preconditioner_rebuilt": diagnostics.potential.rebuilt_preconditioner,
            "mass_relative_change": mass / mass_reference - 1.0,
            "minimum_density_m3": minimum_density,
        }
        if step % config.diagnostics_interval_steps == 0:
            _write_jsonl(diagnostics_path, record)

        checkpoint_due = time_s >= next_checkpoint_time
        if checkpoint_due or math.isclose(time_s, config.t_end_s):
            _save_checkpoint(
                checkpoint_path,
                density_history,
                time_s=time_s,
                dt_s=dt_s,
                step=step,
            )
            archive = _archive_stable_checkpoint(checkpoint_path, time_s)
            while next_checkpoint_time <= time_s:
                next_checkpoint_time += config.checkpoint_interval_s
            logger.info(
                "step=%d t=%g dt=%g CFL=%g residual=%g mass_change=%g "
                "stable_checkpoint=%s",
                step,
                time_s,
                dt_step,
                record["cfl"],
                record["potential_relative_residual"],
                record["mass_relative_change"],
                archive.name,
            )

    completed = math.isclose(
        time_s, config.t_end_s, rel_tol=0.0, abs_tol=1.0e-8
    )
    if frame_archive is not None:
        frame_archive.finalize(completed=completed)

    snapshot_path = simulation_directory / f"density_{config.scheme}.npz"
    np.savez_compressed(
        snapshot_path,
        density=density_history[0],
        x_km=config.x_coordinates_m / 1.0e3 - 125.0,
        y_km=config.y_coordinates_m / 1.0e3,
        time_s=time_s,
        dt_s=dt_s,
        scheme=config.scheme,
        potential_operator=config.potential_operator,
        potential_boundary_y=config.potential_boundary_y,
        initial_mass=mass_reference,
        final_mass=float(np.sum(density_history[0]) * config.dx_m * config.dy_m),
        frame_archive=(
            str(Path(frame_archive_directory))
            if frame_archive_directory is not None
            else ""
        ),
        frame_interval_s=(
            frame_interval_s if frame_archive_directory is not None else 0.0
        ),
    )
    return snapshot_path
