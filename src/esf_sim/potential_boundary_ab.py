"""A/B calibration of the vertical boundary used by the potential equation.

Variant A is the promoted point-centered equation (2.21) solver with the
first and last altitude rows fixed to zero.  Variant B keeps the same
point-centered/log-coefficient stencil and PARDISO solver, but applies the
paper's equal-value extrapolation to the potential as well as the density.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from . import _model_impl
from .arithmetic_mean_ab import (
    PDM_ALPHA,
    REFERENCE_VARIABLE,
    SNAPSHOT_TIME_S,
    WIDTH_ALTITUDES_KM,
    _author_density,
    _diagnostics_summary,
    _field_agreement,
    _load_snapshot,
    _snapshot_complete,
    _width_table,
    arm_width_profile,
)
from .config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    BEST_BASELINE_POTENTIAL_OPERATOR,
    SimulationConfig,
)
from .metrics import compare_with_reference, morphology_metrics
from .model import run_simulation
from .point_centered_potential import (
    LogPointCenteredPotentialSolver,
    _load_pardiso,
)
from .potential import PotentialDiagnostics
from .reference import md5sum


STAGE_NAME = "potential_vertical_boundary_ab"
DIRICHLET_BOUNDARY = "psi_zero_at_first_and_last_active_rows"
EXTRAPOLATED_BOUNDARY = BEST_BASELINE_POTENTIAL_BOUNDARY


def build_equal_extrapolated_log_point_centered_matrix(
    coefficient: np.ndarray,
    dx: float,
    dy: float,
    *,
    impose_gauge: bool = True,
) -> csr_matrix:
    """Return the equation (2.21) five-point matrix with extrapolated y ghosts.

    The four identical ghost layers reduce to one identical ghost value for a
    five-point stencil.  Consequently all vertical grid rows remain unknowns,
    the ghost contribution is folded into the boundary-row diagonal, and the
    ungauged operator annihilates constants.  A single identity row fixes the
    otherwise arbitrary additive constant in ``psi``.
    """

    raw = np.asarray(coefficient, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) < 2:
        raise ValueError("coefficient must be two-dimensional with at least 2 rows")
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("coefficient must be positive and finite")

    log_a = np.log(raw)
    ny, nx = log_a.shape
    identifiers = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    drift_x = (
        np.roll(log_a, -1, axis=1) - np.roll(log_a, 1, axis=1)
    ) / (2.0 * dx)
    padded_y = np.pad(log_a, ((1, 1), (0, 0)), mode="edge")
    drift_y = (padded_y[2:] - padded_y[:-2]) / (2.0 * dy)

    diagonal = np.full((ny, nx), 2.0 / dx**2 + 2.0 / dy**2)
    diagonal[0] = 2.0 / dx**2 + 1.0 / dy**2 + drift_y[0] / (2.0 * dy)
    diagonal[-1] = 2.0 / dx**2 + 1.0 / dy**2 - drift_y[-1] / (2.0 * dy)

    rows = [identifiers.ravel(), identifiers.ravel(), identifiers.ravel()]
    columns = [
        identifiers.ravel(),
        np.roll(identifiers, -1, axis=1).ravel(),
        np.roll(identifiers, 1, axis=1).ravel(),
    ]
    data = [
        diagonal.ravel(),
        (-1.0 / dx**2 - drift_x / (2.0 * dx)).ravel(),
        (-1.0 / dx**2 + drift_x / (2.0 * dx)).ravel(),
    ]
    rows.extend([identifiers[:-1].ravel(), identifiers[1:].ravel()])
    columns.extend([identifiers[1:].ravel(), identifiers[:-1].ravel()])
    data.extend(
        [
            (-1.0 / dy**2 - drift_y[:-1] / (2.0 * dy)).ravel(),
            (-1.0 / dy**2 + drift_y[1:] / (2.0 * dy)).ravel(),
        ]
    )
    matrix = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(ny * nx, ny * nx),
    )
    matrix.sort_indices()
    if impose_gauge:
        start, stop = matrix.indptr[0], matrix.indptr[1]
        matrix.data[start:stop] = 0.0
        local_diagonal = np.flatnonzero(matrix.indices[start:stop] == 0)
        if local_diagonal.size != 1:
            raise RuntimeError("failed to locate the potential gauge diagonal")
        matrix.data[start + int(local_diagonal[0])] = 1.0
    return matrix


class EqualExtrapolatedLogPointCenteredPotentialSolver(
    LogPointCenteredPotentialSolver
):
    """PARDISO solution with equal-value potential ghost cells in altitude."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.collision_frequency = config.collision_nu0_s * np.exp(
            -config.y_coordinates_m / config.collision_frequency_scale_height_m
        )
        self.solver = _load_pardiso()(mtype=11)
        self._pattern_indptr: np.ndarray | None = None
        self._pattern_indices: np.ndarray | None = None

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        positive_n = np.maximum(n, self.config.density_floor_m3)
        coefficient = positive_n * self.collision_frequency[:, None]
        matrix = build_equal_extrapolated_log_point_centered_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        log_n = np.log(positive_n)
        log_density_dx = (
            np.roll(log_n, -1, axis=1) - np.roll(log_n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        rhs = (
            self.config.gravity_m_s2
            * log_density_dx
            / self.collision_frequency[:, None]
        ).ravel()
        rhs[0] = 0.0
        solution, rebuilt = self._solve(matrix, rhs)
        residual = matrix @ solution - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            self._pattern_indptr = None
            self._pattern_indices = None
            self.solver.set_phase(13)
            solution = np.asarray(self.solver.solve(matrix, rhs), dtype=np.float64)
            residual = matrix @ solution - rhs
            relative_residual = float(np.linalg.norm(residual) / denominator)
            rebuilt = True
        if not np.isfinite(relative_residual) or relative_residual > 1.0e-8:
            raise RuntimeError(
                f"PARDISO residual {relative_residual:.3e} exceeds 1e-8"
            )
        potential = solution.reshape(n.shape)
        potential -= float(np.mean(potential))
        return potential, PotentialDiagnostics(
            iterations=1,
            relative_residual=relative_residual,
            rebuilt_preconditioner=rebuilt,
            used_pyamg=False,
        )


class EqualExtrapolatedPotentialBoundaryESFModel(_model_impl.ESFModel):
    """Best-baseline model with only the potential y boundary changed."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.potential_solver = EqualExtrapolatedLogPointCenteredPotentialSolver(config)

    def velocity_from_potential(
        self, potential: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dx = self.config.dx_m
        dy = self.config.dy_m
        dpsi_dx = (
            np.roll(potential, -1, axis=1) - np.roll(potential, 1, axis=1)
        ) / (2.0 * dx)
        padded_y = np.pad(potential, ((1, 1), (0, 0)), mode="edge")
        dpsi_dy = (padded_y[2:] - padded_y[:-2]) / (2.0 * dy)
        return -dpsi_dy, dpsi_dx


def _stage_root(base: SimulationConfig) -> Path:
    return base.output_directory / "figure3_stages" / STAGE_NAME


def _baseline_snapshot(base: SimulationConfig) -> Path:
    candidates = [
        Path("output/legacy_point_centered_dirichlet/figure3/simulations/density_alpha_0p4.npz"),
        base.output_directory / "figure3" / "simulations" / "density_alpha_0p4.npz",
        base.output_directory / "figure3" / "runs" / "alpha_0p4" / "simulations" / "density_pdm7.npz",
    ]
    for path in candidates:
        if _snapshot_complete(path, (base.ny, base.nx)):
            return path
    raise FileNotFoundError("the completed alpha=0.4 best-baseline snapshot is missing")


def prepare_extrapolated_snapshot(
    base: SimulationConfig, *, resume: bool = True
) -> tuple[Path, SimulationConfig]:
    root = _stage_root(base)
    stable = root / "simulations" / "density_equal_extrapolated_alpha_0p4.npz"
    run_root = root / "runs" / "b_equal_extrapolated"
    config = base.with_updates(
        scheme="pdm7",
        pdm_alpha=PDM_ALPHA,
        t_end_s=SNAPSHOT_TIME_S,
        potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
        output_dir=str(run_root),
    )
    config.validate()
    if _snapshot_complete(stable, (base.ny, base.nx)):
        return stable, config

    archive = config.to_dict()
    archive["ab_calibration"] = {
        "variant": "B",
        "changed_factor_only": "vertical potential boundary implementation",
        "A_boundary": DIRICHLET_BOUNDARY,
        "B_boundary": EXTRAPOLATED_BOUNDARY,
        "B_active_vertical_unknowns": base.ny,
        "B_gauge": "psi[0,0]=0 followed by zero-mean normalization",
        "unchanged": [
            "point-centered five-point equation (2.21)",
            "centered logarithmic coefficient gradients",
            "Intel MKL PARDISO",
            "201x401 grid and horizontal periodic indexing",
            "density equal-value extrapolation",
            "PDM7 alpha=0.4",
            "AB3/SSPRK3 and CFL policy",
            "initial condition and physical parameters",
        ],
    }
    config_path = root / "configs" / "b_equal_extrapolated_alpha_0p4.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    original_model = _model_impl.ESFModel
    try:
        _model_impl.ESFModel = EqualExtrapolatedPotentialBoundaryESFModel
        source = run_simulation(config, resume=resume)
    finally:
        _model_impl.ESFModel = original_model

    with np.load(source, allow_pickle=False) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload["potential_boundary_y"] = np.array(EXTRAPOLATED_BOUNDARY)
    payload["potential_boundary_implementation"] = np.array(
        "four_equal_value_ghost_layers_zero_normal_gradient"
    )
    payload["active_vertical_unknowns"] = np.array(base.ny)
    stable.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stable, **payload)
    return stable, config


def _metric_bundle(
    density: np.ndarray, x: np.ndarray, y: np.ndarray, author: np.ndarray
) -> dict[str, Any]:
    metric = morphology_metrics(density, x, y)
    return {
        "morphology": metric.to_dict(),
        "reference_comparison": compare_with_reference(
            metric, morphology_metrics(author, x, y)
        ),
        "field_agreement_diagnostic": _field_agreement(density, author),
        "main_arm_widths_km": _width_table(density, x, y),
    }


def write_report(
    base: SimulationConfig, a_path: Path, b_path: Path
) -> tuple[Path, Path, dict[str, Any]]:
    root = _stage_root(base)
    a, x, y, time_a = _load_snapshot(a_path)
    b, xb, yb, time_b = _load_snapshot(b_path)
    if not (np.array_equal(x, xb) and np.array_equal(y, yb)):
        raise ValueError("A and B grids differ")
    author = _author_density(base.reference_mat)
    bundles = {
        "A_dirichlet_zero": _metric_bundle(a, x, y, author),
        "B_equal_extrapolation": _metric_bundle(b, x, y, author),
        "author": {
            "morphology": morphology_metrics(author, x, y).to_dict(),
            "main_arm_widths_km": _width_table(author, x, y),
        },
    }
    a_agree = bundles["A_dirichlet_zero"]["field_agreement_diagnostic"]
    b_agree = bundles["B_equal_extrapolation"]["field_agreement_diagnostic"]
    a_ref = bundles["A_dirichlet_zero"]["reference_comparison"]
    b_ref = bundles["B_equal_extrapolation"]["reference_comparison"]
    report: dict[str, Any] = {
        "experiment": "vertical potential boundary A/B on the point-centered best baseline",
        "grid": [base.nx, base.ny],
        "pdm_alpha": PDM_ALPHA,
        "time_s": SNAPSHOT_TIME_S,
        "reference_variable": REFERENCE_VARIABLE,
        "reference_mat_md5": md5sum(base.reference_mat),
        "paper_evidence": {
            "JGR_2021_page_4": "altitudinal direction with equivalent extrapolation boundary conditions",
            "Chinese_thesis_pages_28_29": "y方向采用等值外推边界条件",
            "ambiguity": "Neither source limits that statement to density alone.",
        },
        "variants": {
            "A": DIRICHLET_BOUNDARY,
            "B": EXTRAPOLATED_BOUNDARY,
        },
        "controlled_scope": {
            "changed_only": "potential y boundary and its ghost-cell-consistent boundary derivative",
            "A_active_vertical_unknowns": base.ny - 2,
            "B_active_vertical_unknowns": base.ny,
            "unchanged_operator": BEST_BASELINE_POTENTIAL_OPERATOR,
        },
        "times_verified_s": {"A": time_a, "B": time_b},
        "solver_diagnostics": {
            "A": _diagnostics_summary(
                base.output_directory / "figure3" / "runs" / "alpha_0p4" / "simulations" / "pdm7_diagnostics.jsonl"
            ),
            "B": _diagnostics_summary(
                root / "runs" / "b_equal_extrapolated" / "simulations" / "pdm7_diagnostics.jsonl"
            ),
        },
        "results": bundles,
        "calibration_outcome": {
            "B_reaches_target_time": bool(np.isclose(time_b, SNAPSHOT_TIME_S)),
            "B_improves_spatial_correlation": bool(
                b_agree["pearson_spatial_correlation"]
                > a_agree["pearson_spatial_correlation"]
            ),
            "B_improves_normalized_rmse": bool(
                b_agree["normalized_rmse_by_reference_range"]
                < a_agree["normalized_rmse_by_reference_range"]
            ),
            "B_improves_bubble_top_error": bool(
                b_ref["bubble_top_absolute_error_km"]
                < a_ref["bubble_top_absolute_error_km"]
            ),
            "B_improves_depleted_area_error": bool(
                b_ref["depleted_area_relative_error"]
                < a_ref["depleted_area_relative_error"]
            ),
        },
    }
    report_path = root / "potential_vertical_boundary_ab_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_path = root / "potential_vertical_boundary_ab_source_data.npz"
    np.savez_compressed(
        source_path,
        x_km=x,
        y_km=y,
        density_A_dirichlet_m3=a,
        density_B_equal_extrapolation_m3=b,
        density_author_m3=author,
        difference_B_minus_A_m3=b - a,
        arm_width_A_km=arm_width_profile(a, x, y)["mean_km"],
        arm_width_B_km=arm_width_profile(b, x, y)["mean_km"],
        arm_width_author_km=arm_width_profile(author, x, y)["mean_km"],
        time_s=SNAPSHOT_TIME_S,
        pdm_alpha=PDM_ALPHA,
    )
    return report_path, source_path, report


def create_figure(base: SimulationConfig, a_path: Path, b_path: Path) -> dict[str, Path]:
    root = _stage_root(base)
    a, x, y, _ = _load_snapshot(a_path)
    b, _, _, _ = _load_snapshot(b_path)
    author = _author_density(base.reference_mat)
    profiles = {
        "A: psi=0": arm_width_profile(a, x, y)["mean_km"],
        "B: equal extrapolation": arm_width_profile(b, x, y)["mean_km"],
        "Author": arm_width_profile(author, x, y)["mean_km"],
    }
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(9.0, 7.0), constrained_layout=True)
    image = None
    for axis, field, title in zip(
        axes[0],
        (a, b, author),
        ("a) A: psi=0 boundary", "b) B: equal extrapolation", "c) Author reference"),
    ):
        image = axis.pcolormesh(
            x, y, field / 1.0e11, shading="nearest", cmap="plasma",
            vmin=0.0, vmax=11.0, rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
        axis.set_xlabel("East-West (km)")
    axes[0, 0].set_ylabel("Altitude (km)")
    assert image is not None
    figure.colorbar(image, ax=axes[0], label=r"Density ($10^5$ cm$^{-3}$)", shrink=0.88)

    difference = (b - a) / 1.0e11
    limit = max(float(np.percentile(np.abs(difference), 99.0)), 0.1)
    diff_image = axes[1, 0].pcolormesh(
        x, y, difference, shading="nearest", cmap="RdBu_r",
        vmin=-limit, vmax=limit, rasterized=True,
    )
    axes[1, 0].set_title("d) B - A")
    axes[1, 0].set_xlabel("East-West (km)")
    axes[1, 0].set_ylabel("Altitude (km)")
    axes[1, 0].set_xlim(-125.0, 125.0)
    axes[1, 0].set_ylim(250.0, 650.0)
    figure.colorbar(diff_image, ax=axes[1, 0], label=r"$10^5$ cm$^{-3}$", shrink=0.88)

    for label, profile in profiles.items():
        axes[1, 1].plot(profile, y, linewidth=1.25, label=label)
    axes[1, 1].set_title("e) Main-arm width")
    axes[1, 1].set_xlabel("Mean width (km)")
    axes[1, 1].set_ylabel("Altitude (km)")
    axes[1, 1].set_xlim(0.0, 65.0)
    axes[1, 1].set_ylim(350.0, 520.0)
    axes[1, 1].legend(fontsize=7.0)
    axes[1, 1].grid(alpha=0.2)

    metrics = {
        "A": _field_agreement(a, author),
        "B": _field_agreement(b, author),
    }
    axes[1, 2].bar(
        [0, 1],
        [metrics["A"]["pearson_spatial_correlation"], metrics["B"]["pearson_spatial_correlation"]],
        color=["#5470c6", "#d95f02"],
    )
    axes[1, 2].set_xticks([0, 1], ["A: psi=0", "B: extrapolated"])
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_ylabel("Spatial correlation")
    axes[1, 2].set_title("f) Agreement with author")
    axes[1, 2].grid(axis="y", alpha=0.2)

    outputs = {
        suffix: root / f"potential_vertical_boundary_ab_comparison.{suffix}"
        for suffix in ("svg", "pdf", "png", "tiff")
    }
    title = "ESF vertical potential boundary A/B calibration"
    figure.savefig(outputs["svg"], metadata={"Title": title})
    figure.savefig(
        outputs["pdf"],
        metadata={
            "Title": title,
            "Subject": "201x401, PDM alpha=0.4, t=3150 s",
        },
    )
    figure.savefig(outputs["png"], dpi=600)
    figure.savefig(outputs["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    return outputs


def run_ab(
    base: SimulationConfig, *, simulate: bool = True, resume: bool = True
) -> dict[str, Any]:
    a_path = _baseline_snapshot(base)
    b_path = _stage_root(base) / "simulations" / "density_equal_extrapolated_alpha_0p4.npz"
    if simulate:
        b_path, _ = prepare_extrapolated_snapshot(base, resume=resume)
    elif not _snapshot_complete(b_path, (base.ny, base.nx)):
        raise FileNotFoundError(f"complete B snapshot is missing: {b_path}")
    report_path, source_path, report = write_report(base, a_path, b_path)
    outputs = create_figure(base, a_path, b_path)
    return {
        "A_snapshot": str(a_path),
        "B_snapshot": str(b_path),
        "report": str(report_path),
        "source_data": str(source_path),
        "figure_outputs": {key: str(path) for key, path in outputs.items()},
        "calibration_outcome": report["calibration_outcome"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.potential_boundary_ab")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_ab(
        SimulationConfig.from_json(args.config),
        simulate=not args.skip_simulation,
        resume=not args.no_resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EqualExtrapolatedLogPointCenteredPotentialSolver",
    "EqualExtrapolatedPotentialBoundaryESFModel",
    "build_equal_extrapolated_log_point_centered_matrix",
    "run_ab",
]
