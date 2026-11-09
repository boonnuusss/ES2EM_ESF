"""Independent numerical reproduction workflow for Li et al. (2021) Figure 6."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks

from .config import (
    CONSERVATIVE_HARMONIC_OPERATOR,
    DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
    SimulationConfig,
)
from .metrics import compare_with_reference, morphology_metrics
from .model import run_simulation
from .reference import verify_reference_files


@dataclass(frozen=True)
class Figure6Case:
    key: str
    nx: int
    ny: int
    time_s: float
    reference_variable: str
    panel_label: str

    @property
    def dx_km(self) -> float:
        return 250.0 / (self.nx - 1)

    @property
    def dy_km(self) -> float:
        return 400.0 / (self.ny - 1)

    @property
    def interval_label(self) -> str:
        return f"{self.nx - 1}×{self.ny - 1} cells"


FIGURE6_CASES = (
    Figure6Case("grid_101x201", 101, 201, 3330.0, "n1_04_3330", "a)"),
    Figure6Case("grid_201x401", 201, 401, 3150.0, "n2_04_3150", "b)"),
    Figure6Case("grid_301x601", 301, 601, 3000.0, "n3_04_3000", "c)"),
    Figure6Case("grid_401x801", 401, 801, 2900.0, "n4_04_2900", "d)"),
)


def case_by_key(key: str) -> Figure6Case:
    for case in FIGURE6_CASES:
        if case.key == key:
            return case
    raise KeyError(f"unknown Figure 6 case {key!r}")


def case_config(
    base: SimulationConfig, case: Figure6Case, output_directory: Path
) -> SimulationConfig:
    """Return the exact endpoint-inclusive grid used by one Figure 6 case."""

    return base.with_updates(
        scheme="pdm7",
        pdm_alpha=0.4,
        nx=case.nx,
        ny=case.ny,
        x_min_km=0.0,
        x_max_km=250.0,
        y_min_km=250.0,
        y_max_km=650.0,
        dx_km=case.dx_km,
        dy_km=case.dy_km,
        t_end_s=case.time_s,
        output_dir=str(output_directory),
        relative_noise=0.0,
    )


def _load_snapshot(path: str | Path):
    with np.load(path) as saved:
        return (
            np.asarray(saved["density"], dtype=np.float64),
            np.asarray(saved["x_km"], dtype=np.float64),
            np.asarray(saved["y_km"], dtype=np.float64),
            float(saved["time_s"]),
        )


def _snapshot_matches(
    path: Path, case: Figure6Case, expected_operator: str,
    expected_boundary: str = DIRICHLET_ZERO_POTENTIAL_BOUNDARY,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as saved:
            stored_operator = (
                str(saved["potential_operator"])
                if "potential_operator" in saved.files
                else CONSERVATIVE_HARMONIC_OPERATOR
            )
            stored_boundary = (
                str(saved["potential_boundary_y"])
                if "potential_boundary_y" in saved.files
                else DIRICHLET_ZERO_POTENTIAL_BOUNDARY
            )
            return bool(
                saved["density"].shape == (case.ny, case.nx)
                and np.isclose(float(saved["time_s"]), case.time_s)
                and stored_operator == expected_operator
                and stored_boundary == expected_boundary
            )
    except (OSError, KeyError, ValueError):
        return False


def _archive_config(config: SimulationConfig, path: Path, case: Figure6Case) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload["figure6_case"] = {
        "grid_points": [case.nx, case.ny],
        "finite_volume_intervals": [case.nx - 1, case.ny - 1],
        "dx_km": case.dx_km,
        "dy_km": case.dy_km,
        "snapshot_time_s": case.time_s,
        "pdm_alpha": 0.4,
        "reference_variable": case.reference_variable,
    }
    payload["implementation_note"] = (
        "The paper caption calls the endpoint-inclusive arrays cells; the "
        "panel artwork labels the corresponding interval counts."
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_snapshot(source: Path, destination: Path, case: Figure6Case) -> Path:
    density, x_km, y_km, time_s = _load_snapshot(source)
    with np.load(source) as saved:
        potential_operator = (
            str(saved["potential_operator"])
            if "potential_operator" in saved.files
            else CONSERVATIVE_HARMONIC_OPERATOR
        )
        potential_boundary_y = (
            str(saved["potential_boundary_y"])
            if "potential_boundary_y" in saved.files
            else DIRICHLET_ZERO_POTENTIAL_BOUNDARY
        )
    if density.shape != (case.ny, case.nx) or not np.isclose(time_s, case.time_s):
        raise ValueError(f"{source} does not match {case.key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        density=density,
        x_km=x_km,
        y_km=y_km,
        time_s=time_s,
        scheme="pdm7",
        pdm_alpha=0.4,
        potential_operator=potential_operator,
        potential_boundary_y=potential_boundary_y,
        nx=case.nx,
        ny=case.ny,
        dx_km=case.dx_km,
        dy_km=case.dy_km,
        provenance="independent_esf_simulation",
    )
    return destination


def prepare_figure6_snapshots(
    base: SimulationConfig,
    *,
    selected_cases: Iterable[str] | None = None,
    simulate: bool = True,
    resume: bool = True,
) -> dict[str, Path]:
    """Prepare selected independent snapshots, reusing the identical Figure 3 b run."""

    selected = set(selected_cases or (case.key for case in FIGURE6_CASES))
    unknown = selected - {case.key for case in FIGURE6_CASES}
    if unknown:
        raise KeyError(f"unknown Figure 6 case keys: {sorted(unknown)}")
    root = base.output_directory / "figure6"
    snapshots: dict[str, Path] = {}
    for case in FIGURE6_CASES:
        if case.key not in selected:
            continue
        stable = root / "simulations" / f"density_{case.key}.npz"
        run_root = root / "runs" / case.key
        config = case_config(base, case, run_root)
        config.validate()
        _archive_config(config, root / "configs" / f"{case.key}.json", case)
        if _snapshot_matches(
            stable, case, base.potential_operator, base.potential_boundary_y
        ):
            snapshots[case.key] = stable
            continue

        if case.key == "grid_201x401":
            source = (
                base.output_directory
                / "figure3"
                / "simulations"
                / "density_alpha_0p4.npz"
            )
        else:
            source = run_root / "simulations" / "density_pdm7.npz"
        if simulate and not _snapshot_matches(
            source, case, base.potential_operator, base.potential_boundary_y
        ):
            print(
                f"[esf_sim] running Figure 6 {case.nx}x{case.ny}, "
                f"alpha=0.4 to t={case.time_s:g} s",
                flush=True,
            )
            source = run_simulation(config, resume=resume)
        if not _snapshot_matches(
            source, case, base.potential_operator, base.potential_boundary_y
        ):
            raise FileNotFoundError(
                f"missing complete Figure 6 snapshot {source} for {case.key}"
            )
        snapshots[case.key] = _normalize_snapshot(source, stable, case)
    return snapshots


def all_stable_snapshots(base: SimulationConfig) -> dict[str, Path]:
    root = base.output_directory / "figure6" / "simulations"
    result = {}
    for case in FIGURE6_CASES:
        path = root / f"density_{case.key}.npz"
        if not _snapshot_matches(
            path, case, base.potential_operator, base.potential_boundary_y
        ):
            raise FileNotFoundError(f"missing complete snapshot {path}")
        result[case.key] = path
    return result


def depletion_finger_diagnostic(
    density: np.ndarray, x_km: np.ndarray, y_km: np.ndarray
) -> dict[str, float | int]:
    """Count row-wise depletion fingers in the 350-520 km branch region."""

    n = np.asarray(density, dtype=np.float64)
    x = np.asarray(x_km, dtype=np.float64)
    y = np.asarray(y_km, dtype=np.float64)
    dx = float(np.median(np.diff(x)))
    counts: list[int] = []
    spacings: list[float] = []
    sampled_altitudes: list[float] = []
    for row in np.where((y >= 350.0) & (y <= 520.0))[0]:
        background = float(np.percentile(n[row], 90.0))
        deficit = np.clip(1.0 - n[row] / max(background, np.finfo(float).tiny), 0.0, None)
        peaks, _ = find_peaks(
            deficit,
            height=0.35,
            prominence=0.05,
            distance=max(1, int(round(1.5 / dx))),
        )
        counts.append(int(peaks.size))
        sampled_altitudes.append(float(y[row]))
        if peaks.size > 1:
            spacings.append(float(np.median(np.diff(x[peaks]))))
    maximum = max(counts, default=0)
    maximum_index = counts.index(maximum) if counts else 0
    return {
        "maximum_depletion_fingers_per_row": int(maximum),
        "altitude_of_maximum_fingers_km": (
            sampled_altitudes[maximum_index] if sampled_altitudes else float("nan")
        ),
        "finger_count_p90": float(np.percentile(counts, 90.0)) if counts else 0.0,
        "median_detected_spacing_km": (
            float(np.median(spacings)) if spacings else float("nan")
        ),
    }


def _load_references(base: SimulationConfig):
    raw = loadmat(Path(base.reference_mat))
    result = {}
    for case in FIGURE6_CASES:
        values = np.asarray(raw[case.reference_variable], dtype=np.float64)
        expected = (case.nx + 8, case.ny + 8)
        if values.shape != expected:
            raise ValueError(
                f"{case.reference_variable} has shape {values.shape}, expected {expected}"
            )
        result[case.key] = values[4:-4, 4:-4].T.copy()
    return result


def write_figure6_source_data(
    snapshots: Mapping[str, str | Path], output_path: str | Path
) -> Path:
    source: dict[str, np.ndarray | float] = {"pdm_alpha": 0.4}
    for case in FIGURE6_CASES:
        density, x_km, y_km, time_s = _load_snapshot(snapshots[case.key])
        source[f"density_{case.key}_m3"] = density
        source[f"x_{case.key}_km"] = x_km
        source[f"y_{case.key}_km"] = y_km
        source[f"time_{case.key}_s"] = time_s
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **source)
    return path


def create_figure6(
    snapshots: Mapping[str, str | Path], output_directory: str | Path
) -> dict[str, Path]:
    """Render the paper-like two-by-two resolution comparison."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(6.35, 8.0), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.845,
        bottom=0.075,
        top=0.975,
        wspace=0.08,
        hspace=0.045,
    )
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    image = None
    for index, (axis, case) in enumerate(zip(axes, FIGURE6_CASES)):
        density, x_km, y_km, _ = _load_snapshot(snapshots[case.key])
        image = axis.pcolormesh(
            x_km,
            y_km,
            density / 1.0e11,
            cmap="plasma",
            vmin=0.0,
            vmax=11.0,
            shading="nearest",
            rasterized=True,
        )
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_xticks([-100, -50, 0, 50, 100])
        axis.set_yticks([250, 350, 450, 550, 650])
        axis.tick_params(top=True, right=True, length=4.0)
        axis.text(
            0.05,
            0.965,
            f"{case.panel_label} {case.interval_label}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10.0,
            fontweight="bold",
        )
        row, column = divmod(index, 2)
        if row == 0:
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel("East-West (km)", fontsize=10.0)
        if column == 1:
            axis.tick_params(labelleft=False)
        else:
            axis.set_ylabel("Altitude (km)", fontsize=10.0)
    assert image is not None
    colorbar_axis = figure.add_axes((0.885, 0.245, 0.044, 0.535))
    colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=np.arange(0, 12, 1))
    colorbar.ax.tick_params(direction="out", length=3.0, labelsize=8.5)
    colorbar.ax.set_title(r"$10^5$/cm$^3$", fontsize=9.0, pad=8.0)

    paths = {
        "svg": output / "figure6_reproduction.svg",
        "pdf": output / "figure6_reproduction.pdf",
        "png": output / "figure6_reproduction.png",
        "tiff": output / "figure6_reproduction.tiff",
    }
    metadata = {"Creator": "esf_sim independent Python model"}
    figure.savefig(paths["svg"], metadata=metadata)
    figure.savefig(paths["pdf"], metadata=metadata)
    figure.savefig(paths["png"], dpi=600)
    figure.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    return paths


def write_figure6_metrics(
    base: SimulationConfig,
    snapshots: Mapping[str, str | Path],
    output_path: str | Path,
) -> dict[str, object]:
    references = _load_references(base)
    simulated: dict[str, object] = {}
    author: dict[str, object] = {}
    comparisons: dict[str, object] = {}
    for case in FIGURE6_CASES:
        density, x_km, y_km, time_s = _load_snapshot(snapshots[case.key])
        simulated_metric = morphology_metrics(density, x_km, y_km)
        reference_density = references[case.key]
        reference_x = np.linspace(-125.0, 125.0, case.nx)
        reference_y = np.linspace(250.0, 650.0, case.ny)
        reference_metric = morphology_metrics(reference_density, reference_x, reference_y)
        simulated[case.key] = {
            "grid_points": [case.nx, case.ny],
            "cell_intervals": [case.nx - 1, case.ny - 1],
            "dx_km": case.dx_km,
            "dy_km": case.dy_km,
            "time_s": time_s,
            "metrics": simulated_metric.to_dict(),
            "secondary_instability": depletion_finger_diagnostic(
                density, x_km, y_km
            ),
        }
        author[case.key] = {
            "reference_variable": case.reference_variable,
            "metrics": reference_metric.to_dict(),
            "secondary_instability": depletion_finger_diagnostic(
                reference_density, reference_x, reference_y
            ),
        }
        comparisons[case.key] = compare_with_reference(
            simulated_metric, reference_metric
        )

    report: dict[str, object] = {
        "provenance": {
            "simulation": (
                "independent Python solves; author arrays are used only after "
                "solving for acceptance metrics"
            ),
            "reference_dataset_version": "ScienceDB V2",
            "reference_file_integrity": verify_reference_files(
                Path(base.reference_mat).parent
            ),
            "pdm_alpha": 0.4,
            "paper_cell_label_note": (
                "101x201 endpoint-inclusive grid points correspond to the "
                "100x200 interval labels drawn in the published panel, and so on"
            ),
        },
        "simulated": simulated,
        "author_reference": author,
        "comparison": comparisons,
        "all_reference_thresholds_pass": bool(
            all(
                all(item["criteria"].values())
                for item in comparisons.values()
            )
        ),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def reproduce_figure6(
    base: SimulationConfig, *, simulate: bool = True, resume: bool = True
):
    root = base.output_directory / "figure6"
    prepare_figure6_snapshots(base, simulate=simulate, resume=resume)
    snapshots = all_stable_snapshots(base)
    source = write_figure6_source_data(snapshots, root / "figure6_source_data.npz")
    figures = create_figure6(snapshots, root)
    report = write_figure6_metrics(base, snapshots, root / "figure6_metrics.json")
    shutil.copyfile(root / "figure6_metrics.json", root / "figure6_acceptance_report.json")
    return figures, source, report


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m esf_sim.figure6")
    parser.add_argument("--config", default="configs/figure6.json")
    parser.add_argument(
        "--cases",
        default=",".join(case.key for case in FIGURE6_CASES),
        help="comma-separated case keys for staged simulation",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig.from_json(args.config)
    selected = [value.strip() for value in args.cases.split(",") if value.strip()]
    prepare_figure6_snapshots(
        base,
        selected_cases=selected,
        simulate=not args.skip_simulation,
        resume=not args.no_resume,
    )
    if args.render:
        figures, source, report = reproduce_figure6(
            base, simulate=False, resume=not args.no_resume
        )
        print(
            json.dumps(
                {
                    "figures": {key: str(value) for key, value in figures.items()},
                    "source_data": str(source),
                    "all_reference_thresholds_pass": report[
                        "all_reference_thresholds_pass"
                    ],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
