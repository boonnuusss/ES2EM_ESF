"""Compare vertical potential boundaries for the paper-oriented ESF setup.

Both branches use arithmetic coefficient averages, the inferred 195-cell
periodic core, Intel MKL PARDISO, and equal-value extrapolation for density.
The zero-normal-current branch is the arithmetic/extrapolated experiment in
``experimental_combination``.  This module adds the strict ``psi=0`` branch
and produces an explicitly time-labelled comparison.
"""

from __future__ import annotations

import argparse
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
from ._experimental_impl import _load_pardiso
from .config import SimulationConfig
from .experimental_combination import (
    ACTIVE_HORIZONTAL_CELLS,
    SNAPSHOT_TIMES_S,
    active_horizontal_coordinates_m,
    expand_author_horizontal,
)
from .metrics import compare_with_reference, morphology_metrics
from .model import initial_density
from .numerics import advection_rhs
from .potential import PotentialDiagnostics


STAGE_NAME = "arithmetic_dirichlet_author_core"
COMPARISON_NAME = "potential_boundary_split_alpha06"
ZERO_CURRENT_STAGE_NAME = "arithmetic_extrapolate_author_core"


def build_arithmetic_dirichlet_matrix(
    coefficient: np.ndarray, dx: float, dy: float
) -> tuple[csr_matrix, float]:
    """Build ``-div(a grad)`` with arithmetic faces and ``psi=0`` at y edges.

    The first and last rows of *coefficient* are boundary grid points.  Only
    rows ``1:-1`` are unknowns.  Their south/north boundary-face terms remain
    on the diagonal, which imposes homogeneous Dirichlet values.
    """

    raw = np.asarray(coefficient, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 3:
        raise ValueError("coefficient must be a two-dimensional array")
    scale = float(np.max(raw))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coefficient must contain positive finite values")
    a = raw / scale
    interior = a[1:-1]
    nyi, nx = interior.shape
    identifiers = np.arange(nyi * nx, dtype=np.int64).reshape(nyi, nx)

    east = 0.5 * (interior + np.roll(interior, -1, axis=1))
    west = 0.5 * (interior + np.roll(interior, 1, axis=1))
    north = 0.5 * (interior + a[2:])
    south = 0.5 * (interior + a[:-2])
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
    if nyi > 1:
        rows.extend([identifiers[:-1].ravel(), identifiers[1:].ravel()])
        columns.extend([identifiers[1:].ravel(), identifiers[:-1].ravel()])
        data.extend(
            [
                (-north[:-1] / dy**2).ravel(),
                (-south[1:] / dy**2).ravel(),
            ]
        )
    matrix = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(identifiers.size, identifiers.size),
    )
    matrix.sort_indices()
    return matrix, scale


def zero_normal_current_boundary_flux(
    coefficient: np.ndarray, potential: np.ndarray, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate boundary normal-current flux for equal-potential ghost cells."""

    a = np.asarray(coefficient, dtype=np.float64)
    psi = np.asarray(potential, dtype=np.float64)
    if a.shape != psi.shape:
        raise ValueError("coefficient and potential must have the same shape")
    padded = np.pad(psi, ((1, 1), (0, 0)), mode="edge")
    bottom = a[0] * (psi[0] - padded[0]) / dy
    top = a[-1] * (padded[-1] - psi[-1]) / dy
    return bottom, top


class ArithmeticDirichletPotentialSolver:
    """PARDISO solve with arithmetic faces and zero vertical potential."""

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
        matrix, scale = build_arithmetic_dirichlet_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        density_dx = (
            np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        rhs = (self.config.gravity_m_s2 * density_dx[1:-1] / scale).ravel()
        solution = np.asarray(self.solver.solve(matrix, rhs), dtype=np.float64)
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            raise RuntimeError(
                f"PARDISO residual {relative_residual:.3e} exceeds 1e-8"
            )
        potential = np.zeros_like(n)
        potential[1:-1] = solution.reshape(n.shape[0] - 2, n.shape[1])
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=False,
            used_pyamg=False,
        )


class DirichletCombinationESFModel(_model_impl.ESFModel):
    """ESF model differing from the prior combination only in psi boundary."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.potential_solver = ArithmeticDirichletPotentialSolver(config)

    def evaluate_rhs(self, density: np.ndarray):
        potential, potential_diagnostics = self.potential_solver.solve(density)
        velocity_x, velocity_y = self.velocity_from_potential(potential)
        # periodic_y=False uses four equal-value density ghosts in numerics.py.
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


def _active_config(base: SimulationConfig, output: Path) -> SimulationConfig:
    x = active_horizontal_coordinates_m(base) / 1.0e3
    return base.with_updates(
        scheme="pdm7",
        pdm_alpha=0.6,
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        density=expand_author_horizontal(active_density),
        active_density=active_density,
        x_km=base.x_coordinates_m / 1.0e3 - 125.0,
        active_x_km=active_horizontal_coordinates_m(base) / 1.0e3 - 125.0,
        y_km=base.y_coordinates_m / 1.0e3,
        time_s=time_s,
        pdm_alpha=0.6,
        stage=STAGE_NAME,
        potential_y_boundary="dirichlet_zero",
        density_y_boundary="four_equal_value_ghosts",
    )


def run_dirichlet_alpha06(
    base: SimulationConfig, *, resume: bool = True
) -> dict[float, Path]:
    """Run the strict-zero-potential branch through t=3050 s."""

    root = base.output_directory / "figure3_stages" / STAGE_NAME
    run_root = root / "runs" / "alpha_0p6"
    config = _active_config(base, run_root)
    config.validate()
    archive = config.to_dict()
    archive["experimental_stage"] = {
        "potential_face_average": "arithmetic",
        "potential_y_boundary": "dirichlet_zero_at_first_and_last_grid_rows",
        "density_y_boundary": "four_equal_value_ghost_cells",
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

    checkpoint = run_root / "checkpoints" / "dirichlet.npz"
    log_path = run_root / "simulations" / "dirichlet.log"
    diagnostic_path = run_root / "simulations" / "dirichlet_diagnostics.jsonl"
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
    model = DirichletCombinationESFModel(config)
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
                _write_snapshot(path, updated, base, time_s=time_s)
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
    _checkpoint(checkpoint, history, time_s, dt_s, step)
    return snapshots


def _load_density(path: Path) -> np.ndarray:
    with np.load(path) as saved:
        return np.asarray(saved["density"], dtype=np.float64)


def render_boundary_comparison(
    base: SimulationConfig, dirichlet_snapshots: dict[float, Path]
) -> dict[str, object]:
    """Render both boundary branches and the author alpha=0.6 result."""

    output_root = base.output_directory / "figure3_stages" / COMPARISON_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    zero_current_root = (
        base.output_directory / "figure3_stages" / ZERO_CURRENT_STAGE_NAME
    )
    dirichlet = {
        time: _load_density(path) for time, path in dirichlet_snapshots.items()
    }
    zero_current = {
        1900.0: _load_density(
            zero_current_root / "simulations" / "density_alpha_0p6_t1900.npz"
        ),
        2250.0: _load_density(
            zero_current_root / "simulations" / "density_alpha_0p6_t2250.npz"
        ),
    }
    stopped_path = zero_current_root / "simulations" / "density_alpha_0p6_stopped.npz"
    with np.load(stopped_path) as saved:
        stopped_time = float(saved["time_s"])
        zero_current[stopped_time] = np.asarray(saved["density"], dtype=np.float64)
    author = np.asarray(
        loadmat(Path(base.reference_mat))["n2_06_3050"][4:-4, 4:-4].T,
        dtype=np.float64,
    )
    x = base.x_coordinates_m / 1.0e3 - 125.0
    y = base.y_coordinates_m / 1.0e3

    metric_fields = {
        "dirichlet": {
            str(time): morphology_metrics(value, x, y).to_dict()
            for time, value in dirichlet.items()
        },
        "zero_normal_current": {
            str(time): morphology_metrics(value, x, y).to_dict()
            for time, value in zero_current.items()
        },
        "author_3050": morphology_metrics(author, x, y).to_dict(),
    }

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
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 5.9))
    density_panels = [
        (dirichlet[1900.0], r"$\psi=0$, t = 1,900 s"),
        (dirichlet[2250.0], r"$\psi=0$, t = 2,250 s"),
        (dirichlet[3050.0], r"$\psi=0$, t = 3,050 s"),
        (author, "Author, t = 3,050 s"),
        (zero_current[1900.0], r"$J_y=0$, t = 1,900 s"),
        (zero_current[2250.0], r"$J_y=0$, t = 2,250 s"),
        (
            zero_current[stopped_time],
            rf"$J_y=0$, stopped t = {stopped_time:,.1f} s",
        ),
    ]
    image = None
    for label, axis, (density, title) in zip("abcdefg", axes.ravel()[:7], density_panels):
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
        axis.set_title(title, fontsize=7.5)
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, 0, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
    for axis in axes[0, 1:]:
        axis.set_yticklabels([])
    for axis in axes[1, 1:3]:
        axis.set_yticklabels([])
    for axis in axes[0]:
        axis.set_xticklabels([])
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[1, 0].set_ylabel("Altitude (km)")
    for axis in axes[1, :3]:
        axis.set_xlabel("East-West (km)")

    metric_axis = axes[1, 3]
    metric_axis.text(
        0.035,
        0.965,
        "h)",
        transform=metric_axis.transAxes,
        va="top",
        fontweight="bold",
        fontsize=9,
    )
    dirichlet_times = np.array(sorted(dirichlet))
    current_times = np.array(sorted(zero_current))
    metric_axis.plot(
        dirichlet_times,
        [metric_fields["dirichlet"][str(t)]["bubble_top_km"] for t in dirichlet_times],
        "o-",
        color="#0072B2",
        label=r"$\psi=0$",
    )
    metric_axis.plot(
        current_times,
        [
            metric_fields["zero_normal_current"][str(t)]["bubble_top_km"]
            for t in current_times
        ],
        "s--",
        color="#D55E00",
        label=r"$J_y=0$",
    )
    metric_axis.plot(
        [3050.0],
        [metric_fields["author_3050"]["bubble_top_km"]],
        marker="*",
        markersize=8,
        linestyle="none",
        color="black",
        label="Author",
    )
    metric_axis.set_title("Depletion-top diagnostic", fontsize=7.5)
    metric_axis.set_xlabel("Time (s)")
    metric_axis.set_ylabel("Bubble top (km)")
    metric_axis.set_xlim(1800.0, 3125.0)
    metric_axis.set_ylim(300.0, 670.0)
    metric_axis.grid(alpha=0.25, linewidth=0.5)
    metric_axis.legend(frameon=False, fontsize=6.5, loc="best")

    colorbar = fig.colorbar(
        image, ax=axes[:, :3], fraction=0.018, pad=0.015, aspect=35
    )
    colorbar.set_label(r"Plasma density ($10^5$ cm$^{-3}$)")
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.10, top=0.94, wspace=0.13, hspace=0.12
    )

    output_base = output_root / "boundary_split_alpha06_comparison"
    outputs = {}
    for extension in ("svg", "pdf", "png", "tiff"):
        path = output_base.with_suffix(f".{extension}")
        options = {"bbox_inches": "tight"}
        if extension in {"png", "tiff"}:
            options["dpi"] = 600
        fig.savefig(path, **options)
        outputs[extension] = str(path)
    plt.close(fig)

    dirichlet_final_metric = morphology_metrics(dirichlet[3050.0], x, y)
    author_metric = morphology_metrics(author, x, y)
    report = {
        "experiment": COMPARISON_NAME,
        "controlled_settings": {
            "potential_face_average": "arithmetic",
            "density_y_boundary": "four_equal_value_ghost_cells",
            "horizontal_layout": "inferred_195_active_plus_3_halo_each_side",
            "linear_solver": "Intel MKL PARDISO",
            "pdm_alpha": 0.6,
        },
        "boundary_variants": {
            "dirichlet": "psi=0 at y=250 and 650 km",
            "zero_normal_current": (
                "n*nu_in*dpsi/dy=0; identical to the prior equal-potential "
                "ghost experiment after background-potential removal"
            ),
        },
        "zero_normal_current_status": {
            "status": "stopped_after_cfl_step_collapse",
            "time_s": stopped_time,
            "not_a_target_time_snapshot": True,
        },
        "metrics": metric_fields,
        "dirichlet_vs_author": compare_with_reference(
            dirichlet_final_metric, author_metric
        ),
        "outputs": outputs,
    }
    (output_root / "boundary_split_alpha06_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    source = {
        "x_km": x,
        "y_km": y,
        "author_t3050": author,
        "zero_current_stopped_time_s": stopped_time,
    }
    source.update({f"dirichlet_t{int(t)}": value for t, value in dirichlet.items()})
    source.update(
        {
            f"zero_current_t{int(t) if float(t).is_integer() else str(t).replace('.', 'p')}": value
            for t, value in zero_current.items()
        }
    )
    np.savez_compressed(output_root / "boundary_split_alpha06_source_data.npz", **source)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m esf_sim.experimental_boundary_split"
    )
    parser.add_argument("--config", default="configs/figure2.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig.from_json(args.config)
    snapshots = run_dirichlet_alpha06(base, resume=not args.no_resume)
    if args.render:
        print(json.dumps(render_boundary_comparison(base, snapshots), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
