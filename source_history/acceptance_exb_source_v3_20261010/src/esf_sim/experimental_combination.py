"""Test the paper-oriented Figure 3 discretization combination.

The horizontal layout is inferred from the exact ghost-cell copies in the
ScienceDB arrays: a nominal 201-column field contains a 195-column active
periodic core and three display/halo columns on each side.  This inference is
kept isolated from the baseline implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix

from . import _model_impl
from .config import SimulationConfig
from .experimental import run_simulation_with_cfl_recovery as _unused_runner
from ._experimental_impl import _load_pardiso
from .metrics import compare_with_reference, morphology_metrics
from .model import initial_density
from .numerics import advection_rhs
from .potential import PotentialDiagnostics


STAGE_NAME = "arithmetic_extrapolate_author_core"
ACTIVE_HORIZONTAL_CELLS = 195
DISPLAY_HORIZONTAL_CELLS = 201
HORIZONTAL_HALO_CELLS = 3
SNAPSHOT_TIMES_S = (1900.0, 2250.0, 3050.0)


def active_horizontal_coordinates_m(config: SimulationConfig) -> np.ndarray:
    """Coordinates of the inferred active core within the nominal grid."""

    start = config.x_min_km * 1.0e3 + HORIZONTAL_HALO_CELLS * config.dx_m
    return start + np.arange(ACTIVE_HORIZONTAL_CELLS) * config.dx_m


def expand_author_horizontal(active: np.ndarray) -> np.ndarray:
    """Add the three inferred periodic halo columns used for display/storage."""

    values = np.asarray(active, dtype=np.float64)
    if values.shape[-1] != ACTIVE_HORIZONTAL_CELLS:
        raise ValueError(
            f"active horizontal size must be {ACTIVE_HORIZONTAL_CELLS}, "
            f"got {values.shape[-1]}"
        )
    return np.concatenate(
        (values[..., -HORIZONTAL_HALO_CELLS:], values, values[..., :HORIZONTAL_HALO_CELLS]),
        axis=-1,
    )


def author_storage_with_four_ghosts(active: np.ndarray) -> np.ndarray:
    """Return the 209-column layout observed in the public author arrays."""

    values = np.asarray(active, dtype=np.float64)
    display = expand_author_horizontal(values)
    return np.concatenate((values[..., -4:], display, values[..., :4]), axis=-1)


def build_arithmetic_neumann_matrix(
    coefficient: np.ndarray, dx: float, dy: float
) -> tuple[csr_matrix, float]:
    """Build ``-div(a grad)`` using arithmetic faces and extrapolated y ghosts."""

    raw = np.asarray(coefficient, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 3:
        raise ValueError("coefficient must be a two-dimensional array")
    scale = float(np.max(raw))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coefficient must contain positive finite values")
    a = raw / scale
    ny, nx = a.shape
    identifiers = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)

    east = 0.5 * (a + np.roll(a, -1, axis=1))
    west = 0.5 * (a + np.roll(a, 1, axis=1))
    north = np.zeros_like(a)
    north[:-1] = 0.5 * (a[:-1] + a[1:])
    south = np.zeros_like(a)
    south[1:] = north[:-1]
    diagonal = (east + west) / dx**2 + (north + south) / dy**2

    rows = [identifiers.ravel(), identifiers.ravel(), identifiers.ravel()]
    columns = [
        identifiers.ravel(),
        np.roll(identifiers, -1, axis=1).ravel(),
        np.roll(identifiers, 1, axis=1).ravel(),
    ]
    data = [
        diagonal.ravel(),
        (-east / dx**2).ravel(),
        (-west / dx**2).ravel(),
    ]
    rows.extend([identifiers[:-1].ravel(), identifiers[1:].ravel()])
    columns.extend([identifiers[1:].ravel(), identifiers[:-1].ravel()])
    data.extend([(-north[:-1] / dy**2).ravel(), (-south[1:] / dy**2).ravel()])
    matrix = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(ny * nx, ny * nx),
    ).tolil()
    # Pure extrapolation/Neumann conditions leave an arbitrary additive constant.
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    result = matrix.tocsr()
    result.sort_indices()
    return result, scale


class ArithmeticExtrapolatedPotentialSolver:
    """PARDISO solve with arithmetic faces and four-layer-equivalent y closure."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.collision_frequency = config.collision_nu0_s * np.exp(
            -config.y_coordinates_m / config.collision_frequency_scale_height_m
        )
        self.solver = _load_pardiso()(mtype=11)

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        coefficient = np.maximum(n, self.config.density_floor_m3)
        coefficient = coefficient * self.collision_frequency[:, None]
        matrix, scale = build_arithmetic_neumann_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        density_dx = (
            np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        rhs = (self.config.gravity_m_s2 * density_dx / scale).ravel()
        rhs[0] = 0.0
        solution = np.asarray(self.solver.solve(matrix, rhs), dtype=np.float64)
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            raise RuntimeError(
                f"PARDISO residual {relative_residual:.3e} exceeds 1e-8"
            )
        potential = solution.reshape(n.shape)
        potential -= float(np.mean(potential))
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=False,
            used_pyamg=False,
        )


class CombinationESFModel(_model_impl.ESFModel):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.potential_solver = ArithmeticExtrapolatedPotentialSolver(config)

    def velocity_from_potential(
        self, potential: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Four equal-value ghost layers reduce to these centered derivatives.
        padded_y = np.pad(potential, ((1, 1), (0, 0)), mode="edge")
        dpsi_dy = (padded_y[2:] - padded_y[:-2]) / (2.0 * self.config.dy_m)
        dpsi_dx = (
            np.roll(potential, -1, axis=1)
            - np.roll(potential, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        return -dpsi_dy, dpsi_dx

    def evaluate_rhs(self, density: np.ndarray):
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
        diagnostics = _model_impl.RhsDiagnostics(
            cfl_rate=float(
                np.max(
                    np.abs(velocity_x) / self.config.dx_m
                    + np.abs(velocity_y) / self.config.dy_m
                )
            ),
            maximum_speed_x=float(np.max(np.abs(velocity_x))),
            maximum_speed_y=float(np.max(np.abs(velocity_y))),
            potential=potential_diagnostics,
        )
        return rhs, diagnostics


def _active_config(base: SimulationConfig, *, alpha: float, output: Path) -> SimulationConfig:
    x = active_horizontal_coordinates_m(base) / 1.0e3
    return base.with_updates(
        scheme="pdm7",
        pdm_alpha=alpha,
        nx=ACTIVE_HORIZONTAL_CELLS,
        x_min_km=float(x[0]),
        x_max_km=float(x[-1]),
        t_end_s=3050.0,
        output_dir=str(output),
    )


def _checkpoint(path: Path, history, time_s: float, dt_s: float, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        states=np.stack(history),
        time_s=time_s,
        dt_s=dt_s,
        step=step,
    )
    temporary.replace(path)


def _load_checkpoint(path: Path):
    with np.load(path) as saved:
        return (
            [row.copy() for row in np.asarray(saved["states"])],
            float(saved["time_s"]),
            float(saved["dt_s"]),
            int(saved["step"]),
        )


def _write_snapshot(
    path: Path,
    active_density: np.ndarray,
    base: SimulationConfig,
    *,
    time_s: float,
    alpha: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = expand_author_horizontal(active_density)
    np.savez_compressed(
        path,
        density=display,
        active_density=active_density,
        x_km=base.x_coordinates_m / 1.0e3 - 125.0,
        active_x_km=active_horizontal_coordinates_m(base) / 1.0e3 - 125.0,
        y_km=base.y_coordinates_m / 1.0e3,
        time_s=time_s,
        pdm_alpha=alpha,
        stage=STAGE_NAME,
        inferred_active_horizontal_cells=ACTIVE_HORIZONTAL_CELLS,
    )


def run_alpha06(base: SimulationConfig, *, resume: bool = True) -> dict[float, Path]:
    root = base.output_directory / "figure3_stages" / STAGE_NAME
    run_root = root / "runs" / "alpha_0p6"
    config = _active_config(base, alpha=0.6, output=run_root)
    config.validate()
    archive = config.to_dict()
    archive["experimental_stage"] = {
        "potential_face_average": "arithmetic",
        "potential_y_boundary": "equal_extrapolation_neumann",
        "horizontal_layout": "inferred_195_active_plus_3_halo_each_side",
        "linear_solver": "Intel MKL PARDISO",
    }
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "alpha_0p6.json").write_text(
        json.dumps(archive, indent=2), encoding="utf-8"
    )

    snapshots = {
        time: root / "simulations" / f"density_alpha_0p6_t{int(time)}.npz"
        for time in SNAPSHOT_TIMES_S
    }
    if all(path.exists() for path in snapshots.values()):
        return snapshots

    checkpoint = run_root / "checkpoints" / "combo.npz"
    log_path = run_root / "simulations" / "combo.log"
    diagnostic_path = run_root / "simulations" / "combo_diagnostics.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"esf_sim.{STAGE_NAME}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    if resume and checkpoint.exists():
        history, time_s, dt_s, step = _load_checkpoint(checkpoint)
        logger.info("resumed t=%g dt=%g", time_s, dt_s)
    else:
        history = [initial_density(config)]
        time_s = 0.0
        dt_s = config.dt_s
        step = 0
        diagnostic_path.unlink(missing_ok=True)
    model = CombinationESFModel(config)
    mass_reference = float(np.sum(history[0]) * config.dx_m * config.dy_m)
    next_log = (math.floor(time_s / 100.0) + 1.0) * 100.0

    while time_s < config.t_end_s - 1.0e-12:
        future_targets = [value for value in SNAPSHOT_TIMES_S if value > time_s + 1.0e-10]
        next_target = min(future_targets) if future_targets else config.t_end_s
        dt_step = min(dt_s, next_target - time_s, config.t_end_s - time_s)
        current = history[0]
        try:
            if len(history) < 3:
                updated, diagnostics = model.ssprk3_step(current, dt_step)
            else:
                updated, diagnostics = model.ab3_predictor_corrector_step(
                    current, history[1], history[2], dt_step
                )
        except _model_impl.CFLViolation as error:
            dt_s *= 0.5
            history = [current]
            logger.warning("%s; reducing dt to %g", error, dt_s)
            _checkpoint(checkpoint, history, time_s, dt_s, step)
            continue
        except RuntimeError:
            _checkpoint(checkpoint, history, time_s, dt_s, step)
            logger.exception("elliptic solve failed")
            raise

        minimum = float(np.min(updated))
        if minimum < -config.positivity_tolerance_m3:
            _checkpoint(checkpoint, history, time_s, dt_s, step)
            raise FloatingPointError(f"negative density {minimum:.6e} m^-3")
        if minimum < 0.0:
            updated = np.maximum(updated, 0.0)
        time_s += dt_step
        step += 1
        history.insert(0, updated)
        del history[3:]
        mass = float(np.sum(updated) * config.dx_m * config.dy_m)
        record = {
            "step": step,
            "time_s": time_s,
            "dt_s": dt_step,
            "cfl": diagnostics.cfl_rate * dt_step,
            "maximum_speed_x_m_s": diagnostics.maximum_speed_x,
            "maximum_speed_y_m_s": diagnostics.maximum_speed_y,
            "potential_relative_residual": diagnostics.potential.relative_residual,
            "mass_relative_change": mass / mass_reference - 1.0,
            "minimum_density_m3": minimum,
        }
        with diagnostic_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

        for target, path in snapshots.items():
            if math.isclose(time_s, target, abs_tol=1.0e-9) and not path.exists():
                _write_snapshot(path, updated, base, time_s=time_s, alpha=0.6)
                _checkpoint(checkpoint, history, time_s, dt_s, step)
                logger.info(
                    "snapshot t=%g dt=%g CFL=%g residual=%g mass_change=%g",
                    time_s,
                    dt_step,
                    record["cfl"],
                    record["potential_relative_residual"],
                    record["mass_relative_change"],
                )
        if time_s >= next_log - 1.0e-9:
            logger.info(
                "step=%d t=%g dt=%g CFL=%g residual=%g mass_change=%g",
                step,
                time_s,
                dt_step,
                record["cfl"],
                record["potential_relative_residual"],
                record["mass_relative_change"],
            )
            while next_log <= time_s:
                next_log += 100.0
    return snapshots


def _load_density(path: Path):
    with np.load(path) as saved:
        return (
            np.asarray(saved["density"], dtype=np.float64),
            np.asarray(saved["x_km"], dtype=np.float64),
            np.asarray(saved["y_km"], dtype=np.float64),
        )


def render_alpha06(base: SimulationConfig, snapshots: dict[float, Path]) -> dict[str, object]:
    root = base.output_directory / "figure3_stages" / STAGE_NAME
    author_raw = loadmat(Path(base.reference_mat))["n2_06_3050"]
    author = np.asarray(author_raw[4:-4, 4:-4].T, dtype=np.float64)
    x = base.x_coordinates_m / 1.0e3 - 125.0
    y = base.y_coordinates_m / 1.0e3

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 7.8), sharex=True, sharey=True)
    panels = []
    for time in SNAPSHOT_TIMES_S:
        density, _, _ = _load_density(snapshots[time])
        panels.append((density, f"Simulation, t = {int(time):,} s"))
    panels.append((author, "Author, t = 3,050 s"))
    image = None
    for label, axis, (density, title) in zip("abcd", axes.ravel(), panels):
        image = axis.pcolormesh(
            x,
            y,
            density / 1.0e11,
            shading="auto",
            cmap="plasma",
            vmin=0.0,
            vmax=11.0,
            rasterized=True,
        )
        axis.text(
            0.035,
            0.965,
            f"{label})",
            transform=axis.transAxes,
            va="top",
            color="white",
            fontweight="bold",
            fontsize=9,
        )
        axis.set_title(title, fontsize=8)
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
    axes[1, 0].set_xlabel("East-West (km)")
    axes[1, 1].set_xlabel("East-West (km)")
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[1, 0].set_ylabel("Altitude (km)")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.035)
    colorbar.set_label(r"Plasma density ($10^5$ cm$^{-3}$)")
    fig.subplots_adjust(left=0.11, right=0.88, bottom=0.08, top=0.95, wspace=0.08, hspace=0.12)

    base_path = root / "combo_alpha06_diagnostic"
    outputs = {}
    for extension in ("svg", "pdf", "png", "tiff"):
        path = base_path.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight"}
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs[extension] = str(path)
    plt.close(fig)

    final_density, _, _ = _load_density(snapshots[3050.0])
    simulated_metric = morphology_metrics(final_density, x, y)
    author_metric = morphology_metrics(author, x, y)
    report = {
        "stage": STAGE_NAME,
        "horizontal_layout_is_inferred": True,
        "simulated_alpha_0p6": simulated_metric.to_dict(),
        "author_alpha_0p6": author_metric.to_dict(),
        "comparison": compare_with_reference(simulated_metric, author_metric),
        "outputs": outputs,
    }
    (root / "combo_alpha06_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    source = {"x_km": x, "y_km": y, "author_t3050": author}
    for time, path in snapshots.items():
        source[f"simulation_t{int(time)}"] = _load_density(path)[0]
    np.savez_compressed(root / "combo_alpha06_source_data.npz", **source)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m esf_sim.experimental_combination"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig.from_json(args.config)
    snapshots = run_alpha06(base, resume=not args.no_resume)
    if args.render:
        print(json.dumps(render_alpha06(base, snapshots), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

