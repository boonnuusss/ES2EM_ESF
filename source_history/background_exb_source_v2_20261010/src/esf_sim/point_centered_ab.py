"""A/B calibration of the point-centered five-point form of equation (2.21)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from . import _model_impl
from .arithmetic_mean_ab import (
    REFERENCE_VARIABLE,
    SNAPSHOT_TIME_S,
    PDM_ALPHA,
    WIDTH_ALTITUDES_KM,
    WIDTH_THRESHOLD,
    _author_density,
    _diagnostics_summary,
    _field_agreement,
    _load_snapshot,
    _snapshot_complete,
    _width_table,
    arm_width_profile,
)
from .config import (
    BEST_BASELINE_POTENTIAL_OPERATOR,
    DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
    SimulationConfig,
)
from .point_centered_potential import (
    LogPointCenteredPotentialSolver,
    build_log_point_centered_matrix,
)
from .metrics import compare_with_reference, morphology_metrics
from .model import run_simulation
from .reference import md5sum


STAGE_NAME = "point_centered_equation221_ab"


class PointCenteredESFModel(_model_impl.ESFModel):
    """Compatibility wrapper selecting the promoted production operator."""

    def __init__(self, config: SimulationConfig):
        super().__init__(
            config.with_updates(
                potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
                potential_boundary_y=DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
            )
        )


def prepare_b_snapshot(
    base: SimulationConfig,
    *,
    resume: bool = True,
) -> tuple[Path, SimulationConfig]:
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    run_root = stage_root / "runs" / "b_point_centered"
    stable = stage_root / "simulations" / "density_point_centered_alpha_0p4.npz"
    config = base.with_updates(
        scheme="pdm7",
        pdm_alpha=PDM_ALPHA,
        t_end_s=SNAPSHOT_TIME_S,
        potential_operator=BEST_BASELINE_POTENTIAL_OPERATOR,
        potential_boundary_y=DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
        output_dir=str(run_root),
    )
    config.validate()
    if _snapshot_complete(stable, (base.ny, base.nx)):
        return stable, config

    archive = config.to_dict()
    archive["ab_calibration"] = {
        "variant": "B",
        "changed_factor": "potential discretization",
        "A_operator": "conservative -div(a grad psi), harmonic face mean",
        "B_operator": (
            "point-centered five-point equation (2.21), with grad(a)/a and "
            "grad(n)/n evaluated through centered grad(log(a)) and grad(log(n))"
        ),
        "B_linear_solver": "Intel MKL PARDISO for the nonsymmetric matrix",
        "interpretation_note": (
            "The paper states five-point centered differencing but does not publish "
            "the coefficient-evaluation code. The logarithmic form is continuum-"
            "equivalent and avoids the strong-jump amplification of a centered "
            "numerator divided by the local depleted value."
        ),
        "unchanged_from_A": [
            "201_by_401_grid",
            "all_201_horizontal_values_active_and_periodic",
            "zero_vertical_potential_boundary",
            "equal_value_vertical_density_extrapolation",
            "PDM7_reconstruction_and_alpha_0.4",
            "AB3_predictor_corrector_with_SSPRK3_startup",
            "dt_and_CFL_policy",
            "initial_condition_and_physics",
        ],
    }
    config_path = stage_root / "configs" / "b_point_centered_alpha_0p4.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(archive, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    source = run_root / "simulations" / "density_pdm7.npz"
    if not _snapshot_complete(source, (base.ny, base.nx)):
        source = run_simulation(config, resume=resume)
    with np.load(source) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload["potential_operator"] = np.array(BEST_BASELINE_POTENTIAL_OPERATOR)
    stable.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stable, **payload)
    return stable, config


def _literal_failure_evidence(base: SimulationConfig) -> dict[str, Any] | None:
    path = (
        base.output_directory
        / "figure3_stages"
        / "centered_five_point_ab"
        / "centered_five_point_ab_failure_report.json"
    )
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": raw.get("status"),
        "last_persisted_time_s": raw.get("variant_b", {}).get("last_persisted_time_s"),
        "minimum_accepted_dt_s": raw.get("numerical_health_b_accepted_step_envelope", {}).get("minimum_accepted_dt_s"),
        "maximum_trial_speed_m_s": raw.get("numerical_health_b_accepted_step_envelope", {}).get("maximum_trial_speed_m_s"),
        "note": "Earlier literal centered-numerator/local-denominator form; non-equitemporal and excluded from A/B acceptance.",
    }


def write_report_and_source(
    base: SimulationConfig,
    a_path: str | Path,
    b_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    a, x, y, time_a = _load_snapshot(a_path)
    b, xb, yb, time_b = _load_snapshot(b_path)
    if not (np.array_equal(x, xb) and np.array_equal(y, yb)):
        raise ValueError("A and B grids differ")
    author = _author_density(base.reference_mat)
    metrics_a = morphology_metrics(a, x, y)
    metrics_b = morphology_metrics(b, x, y)
    metrics_author = morphology_metrics(author, x, y)
    widths = {
        "A_harmonic_conservative": _width_table(a, x, y),
        "B_point_centered": _width_table(b, x, y),
        "author_reference": _width_table(author, x, y),
    }
    width_summary: dict[str, Any] = {}
    for altitude in map(lambda value: f"{value:.0f}", WIDTH_ALTITUDES_KM):
        a_width = widths["A_harmonic_conservative"][altitude]["mean_km"]
        b_width = widths["B_point_centered"][altitude]["mean_km"]
        ref_width = widths["author_reference"][altitude]["mean_km"]
        width_summary[altitude] = {
            "B_minus_A_km": b_width - a_width,
            "B_over_A": b_width / max(a_width, np.finfo(float).tiny),
            "A_absolute_error_vs_author_km": abs(a_width - ref_width),
            "B_absolute_error_vs_author_km": abs(b_width - ref_width),
            "B_improves_width_error": bool(abs(b_width - ref_width) < abs(a_width - ref_width)),
        }
    report: dict[str, Any] = {
        "experiment": "point-centered equation (2.21) potential-operator A/B",
        "grid": [base.nx, base.ny],
        "pdm_alpha": PDM_ALPHA,
        "time_s": SNAPSHOT_TIME_S,
        "reference_variable": REFERENCE_VARIABLE,
        "reference_mat_md5": md5sum(base.reference_mat),
        "variants": {
            "A": "conservative harmonic-face potential operator; CG plus PyAMG",
            "B": "point-centered five-point equation (2.21); centered logarithmic coefficients; PARDISO",
        },
        "discretization_scope": {
            "changed": [
                "potential differential operator",
                "linear solver required by the nonsymmetric B matrix",
            ],
            "unchanged": [
                "physics and initial condition",
                "grid and periodic indexing",
                "vertical boundaries",
                "PDM7 alpha=0.4",
                "AB3/SSPRK3 and CFL policy",
            ],
            "paper_ambiguity": (
                "The paper/thesis specify five-point centered differencing but not "
                "whether coefficient quotients were evaluated before or after "
                "the continuum logarithmic identity."
            ),
        },
        "times_verified_s": {"A": time_a, "B": time_b},
        "solver_diagnostics": {
            "A": _diagnostics_summary(
                base.output_directory / "figure3" / "runs" / "alpha_0p4" / "simulations" / "pdm7_diagnostics.jsonl"
            ),
            "B": _diagnostics_summary(
                stage_root / "runs" / "b_point_centered" / "simulations" / "pdm7_diagnostics.jsonl"
            ),
        },
        "literal_quotient_prior_attempt": _literal_failure_evidence(base),
        "morphology_metrics": {
            "A": metrics_a.to_dict(),
            "B": metrics_b.to_dict(),
            "author_reference": metrics_author.to_dict(),
        },
        "reference_comparison": {
            "A": compare_with_reference(metrics_a, metrics_author),
            "B": compare_with_reference(metrics_b, metrics_author),
        },
        "field_agreement_diagnostic_only": {
            "A": _field_agreement(a, author),
            "B": _field_agreement(b, author),
        },
        "main_arm_width_definition": {
            "threshold": "n / row_p90 < 0.5",
            "reported_value": "mean widest depleted segment on the left and right sides",
        },
        "main_arm_widths_km": widths,
        "width_change_summary": width_summary,
    }
    improvements = [entry["B_improves_width_error"] for entry in width_summary.values()]
    report["calibration_outcome"] = {
        "B_reaches_target_time": bool(np.isclose(time_b, SNAPSHOT_TIME_S)),
        "B_widens_majority_of_sampled_rows": bool(
            sum(entry["B_minus_A_km"] > 0.0 for entry in width_summary.values()) >= 3
        ),
        "B_improves_width_error_majority": bool(sum(improvements) >= 3),
        "B_improves_bubble_top_error": bool(
            report["reference_comparison"]["B"]["bubble_top_absolute_error_km"]
            < report["reference_comparison"]["A"]["bubble_top_absolute_error_km"]
        ),
        "B_improves_spatial_correlation": bool(
            report["field_agreement_diagnostic_only"]["B"]["pearson_spatial_correlation"]
            > report["field_agreement_diagnostic_only"]["A"]["pearson_spatial_correlation"]
        ),
    }
    report_path = stage_root / "point_centered_ab_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    profiles = {
        label: arm_width_profile(field, x, y)["mean_km"]
        for label, field in (("A", a), ("B", b), ("author", author))
    }
    source_path = stage_root / "point_centered_ab_source_data.npz"
    np.savez_compressed(
        source_path,
        x_km=x,
        y_km=y,
        density_A_harmonic_conservative_m3=a,
        density_B_point_centered_m3=b,
        density_author_m3=author,
        difference_B_minus_A_m3=b - a,
        arm_width_A_km=profiles["A"],
        arm_width_B_km=profiles["B"],
        arm_width_author_km=profiles["author"],
        width_threshold=WIDTH_THRESHOLD,
        time_s=SNAPSHOT_TIME_S,
        pdm_alpha=PDM_ALPHA,
    )
    return report_path, source_path, report


def create_ab_figure(
    base: SimulationConfig,
    a_path: str | Path,
    b_path: str | Path,
) -> dict[str, Path]:
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    a, x, y, _ = _load_snapshot(a_path)
    b, _, _, _ = _load_snapshot(b_path)
    author = _author_density(base.reference_mat)
    profiles = {
        label: arm_width_profile(field, x, y)["mean_km"]
        for label, field in (("A", a), ("B", b), ("author", author))
    }
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )
    figure = plt.figure(figsize=(9.0, 6.8))
    grid = figure.add_gridspec(
        2, 3, left=0.07, right=0.92, bottom=0.09, top=0.95,
        wspace=0.24, hspace=0.28, height_ratios=(1.08, 0.92)
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    lower = [figure.add_subplot(grid[1, index]) for index in range(3)]
    density_image = None
    for axis, field, title in zip(
        axes,
        (a, b, author),
        (
            "a) A: harmonic conservative",
            "b) B: point-centered Eq. (2.21)",
            "c) Author reference",
        ),
    ):
        density_image = axis.pcolormesh(
            x, y, field / 1.0e11, shading="nearest", cmap="plasma",
            vmin=0.0, vmax=11.0, rasterized=True,
        )
        axis.set_title(title, fontsize=8.2, pad=3.0)
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
        axis.set_xlabel("East-West (km)")
    axes[0].set_ylabel("Altitude (km)")
    assert density_image is not None
    density_axis = figure.add_axes((0.935, 0.555, 0.018, 0.365))
    density_bar = figure.colorbar(density_image, cax=density_axis)
    density_bar.set_label(r"Density ($10^5$ cm$^{-3}$)")

    difference = (b - a) / 1.0e11
    difference_limit = max(float(np.percentile(np.abs(difference), 99.0)), 0.25)
    difference_image = lower[0].pcolormesh(
        x, y, difference, shading="nearest", cmap="RdBu_r",
        vmin=-difference_limit, vmax=difference_limit, rasterized=True,
    )
    lower[0].set_title("d) B - A", fontsize=8.2, pad=3.0)
    lower[0].set_xlim(-125.0, 125.0)
    lower[0].set_ylim(250.0, 650.0)
    lower[0].set_xticks([-100, -50, 0, 50, 100])
    lower[0].set_yticks([250, 350, 450, 550, 650])
    lower[0].set_xlabel("East-West (km)")
    lower[0].set_ylabel("Altitude (km)")
    difference_axis = figure.add_axes((0.318, 0.125, 0.012, 0.255))
    difference_bar = figure.colorbar(difference_image, cax=difference_axis)
    difference_bar.set_label(r"B - A ($10^5$ cm$^{-3}$)")

    row = int(np.argmin(np.abs(y - 460.0)))
    colors = {"A": "#4C78A8", "B": "#E45756", "author": "#333333"}
    labels = {"A": "A harmonic", "B": "B point-centered", "author": "Author"}
    for label, field in (("A", a), ("B", b), ("author", author)):
        background = float(np.percentile(field[row], 90.0))
        lower[1].plot(x, field[row] / background, lw=1.25, color=colors[label], label=labels[label])
    lower[1].axhline(WIDTH_THRESHOLD, color="#777777", lw=0.8, ls="--")
    lower[1].set_xlim(-125.0, 125.0)
    lower[1].set_ylim(0.0, 1.15)
    lower[1].set_xlabel("East-West (km)")
    lower[1].set_ylabel(r"$n/n_{90}$")
    lower[1].set_title("e) Cross-section at 460 km", fontsize=8.2, pad=3.0)
    lower[1].legend(loc="lower right", fontsize=6.6)

    altitude_mask = (y >= 420.0) & (y <= 480.0)
    for label in ("A", "B", "author"):
        lower[2].plot(
            profiles[label][altitude_mask], y[altitude_mask],
            lw=1.4, color=colors[label], label=labels[label]
        )
    lower[2].set_xlim(left=0.0)
    lower[2].set_ylim(420.0, 480.0)
    lower[2].set_xlabel("Mean main-arm width (km)")
    lower[2].set_ylabel("Altitude (km)")
    lower[2].set_title("f) Main-arm width", fontsize=8.2, pad=3.0)
    lower[2].legend(loc="best", fontsize=6.6)

    base_path = stage_root / "point_centered_ab_comparison"
    outputs = {
        extension: base_path.with_suffix(f".{extension}")
        for extension in ("svg", "pdf", "png", "tiff")
    }
    metadata = {"Creator": "esf_sim point-centered equation 2.21 A/B"}
    figure.savefig(outputs["svg"], metadata=metadata)
    figure.savefig(outputs["pdf"], metadata=metadata)
    figure.savefig(outputs["png"], dpi=600)
    figure.savefig(outputs["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    return outputs


def run_ab(
    base: SimulationConfig,
    *,
    simulate: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    a_path = base.output_directory / "figure3" / "simulations" / "density_alpha_0p4.npz"
    if not _snapshot_complete(a_path, (base.ny, base.nx)):
        raise FileNotFoundError(f"complete A baseline is missing: {a_path}")
    stage_root = base.output_directory / "figure3_stages" / STAGE_NAME
    b_path = stage_root / "simulations" / "density_point_centered_alpha_0p4.npz"
    if simulate:
        b_path, _ = prepare_b_snapshot(base, resume=resume)
    elif not _snapshot_complete(b_path, (base.ny, base.nx)):
        raise FileNotFoundError(f"complete B snapshot is missing: {b_path}")
    report_path, source_path, report = write_report_and_source(base, a_path, b_path)
    outputs = create_ab_figure(base, a_path, b_path)
    return {
        "A_snapshot": str(a_path),
        "B_snapshot": str(b_path),
        "report": str(report_path),
        "source_data": str(source_path),
        "figure_outputs": {key: str(value) for key, value in outputs.items()},
        "calibration_outcome": report["calibration_outcome"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.point_centered_ab")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_ab(
        SimulationConfig.from_json(args.config),
        simulate=not args.skip_simulation,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
