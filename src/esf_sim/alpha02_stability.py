"""Calibrate and recover the unstable Figure 3 alpha=0.2 experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import _model_impl
from .config import SimulationConfig
from .model import run_simulation
from .stabilized_potential import (
    ExponentialConservativePotentialSolver,
    point_centered_peclet_diagnostics,
)
from .timeseries import load_frame


ANCHOR_TIME_S = 2300.0
TEST_END_TIME_S = 2350.0
OVERLAP_TIME_S = 2324.0
_BASE_ESF_MODEL = _model_impl.ESFModel


class ExponentialConservativeESFModel(_BASE_ESF_MODEL):
    """Use the fitted positive-face potential solve and unchanged transport."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.potential_solver = ExponentialConservativePotentialSolver(config)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def calibration_paths(base: SimulationConfig) -> dict[str, Path]:
    formal = base.output_directory / "figure3_v2_timeseries"
    root = formal / "alpha02_stability_calibration"
    candidate = root / "exponential_conservative"
    return {
        "formal": formal,
        "root": root,
        "control": formal / "runs" / "alpha_0p2",
        "control_frames": formal / "time_series" / "alpha_0p2",
        "anchor": formal
        / "runs"
        / "alpha_0p2"
        / "checkpoints"
        / "stable"
        / "pdm7_t002300.000.npz",
        "candidate": candidate,
        "candidate_checkpoint": candidate / "checkpoints" / "pdm7.npz",
        "candidate_overlap": candidate
        / "checkpoints"
        / "stable"
        / "pdm7_t002324.000.npz",
        "status": candidate / "status.json",
        "report": root / "alpha02_stability_report.json",
        "figure": root / "alpha02_stability_comparison",
    }


def _candidate_config(base: SimulationConfig) -> SimulationConfig:
    paths = calibration_paths(base)
    return base.with_updates(
        pdm_alpha=0.2,
        potential_operator="exponential_conservative",
        t_end_s=TEST_END_TIME_S,
        output_dir=str(paths["candidate"]),
        checkpoint_interval_s=1.0,
    )


def _checkpoint_state(path: Path) -> tuple[np.ndarray, float, float, int]:
    with np.load(path, allow_pickle=False) as saved:
        return (
            np.asarray(saved["density"], dtype=np.float64),
            float(saved["time_s"]),
            float(saved["dt_s"]),
            int(saved["step"]),
        )


def prepare_candidate_anchor(base: SimulationConfig) -> Path:
    """Copy only the stable density and deliberately restart AB3 history."""

    paths = calibration_paths(base)
    source = paths["anchor"]
    if not source.exists():
        raise FileNotFoundError(f"missing stable alpha=0.2 anchor: {source}")
    destination = paths["candidate_checkpoint"]
    if destination.exists():
        _, time_s, _, _ = _checkpoint_state(destination)
        if time_s >= ANCHOR_TIME_S:
            return destination
    density, time_s, dt_s, step = _checkpoint_state(source)
    if not math.isclose(time_s, ANCHOR_TIME_S, abs_tol=1.0e-9):
        raise RuntimeError(f"anchor time is {time_s:g}, expected {ANCHOR_TIME_S:g}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        density=density,
        density_previous=np.array([]),
        density_previous2=np.array([]),
        history_length=1,
        time_s=time_s,
        dt_s=dt_s,
        step=step,
    )
    temporary.replace(destination)
    return destination


def run_candidate(base: SimulationConfig) -> dict[str, Any]:
    paths = calibration_paths(base)
    if paths["status"].exists():
        saved = json.loads(paths["status"].read_text(encoding="utf-8"))
        if saved.get("status") == "completed":
            return saved
    prepare_candidate_anchor(base)
    config = _candidate_config(base)
    archived = config.to_dict()
    archived.update(
        {
            "calibration_role": "alpha02_exponential_conservative_recovery",
            "anchor_time_s": ANCHOR_TIME_S,
            "changed_factor_only": (
                "point-centred non-monotone potential matrix and divided log RHS "
                "replaced by a positive exponential-face conservative matrix and "
                "the compatible conservative RHS g*Dx(n)"
            ),
            "transport_and_time_integrator": "unchanged PDM7 alpha=0.2 and AB3/SSPRK3",
        }
    )
    _write_json(paths["candidate"] / "config.json", archived)

    original = _model_impl.ESFModel
    _model_impl.ESFModel = ExponentialConservativeESFModel
    try:
        try:
            snapshot = run_simulation(config, resume=True)
            status: dict[str, Any] = {
                "status": "completed",
                "snapshot": str(snapshot),
                "target_time_s": TEST_END_TIME_S,
            }
        except (RuntimeError, FloatingPointError) as error:
            _, time_s, dt_s, step = _checkpoint_state(paths["candidate_checkpoint"])
            status = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "last_time_s": time_s,
                "last_dt_s": dt_s,
                "last_step": step,
            }
    finally:
        _model_impl.ESFModel = original
    _write_json(paths["status"], status)
    return status


def _read_diagnostics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _diagnostic_summary(path: Path) -> dict[str, Any]:
    rows = _read_diagnostics(path)
    if not rows:
        return {"record_count": 0}
    return {
        "record_count": len(rows),
        "first_time_s": float(rows[0]["time_s"]),
        "last_time_s": float(rows[-1]["time_s"]),
        "minimum_dt_s": float(min(row["dt_s"] for row in rows)),
        "maximum_cfl": float(max(row["cfl"] for row in rows)),
        "maximum_speed_m_s": float(
            max(
                max(row["maximum_speed_x_m_s"], row["maximum_speed_y_m_s"])
                for row in rows
            )
        ),
        "maximum_potential_relative_residual": float(
            max(row["potential_relative_residual"] for row in rows)
        ),
        "minimum_density_m3": float(min(row["minimum_density_m3"] for row in rows)),
        "last_mass_relative_change": float(rows[-1]["mass_relative_change"]),
    }


def _agreement(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    return {
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "normalized_rmse": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        "maximum_relative_difference": float(
            np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), np.finfo(float).tiny)
        ),
    }


def _matrix_health(density: np.ndarray, config: SimulationConfig) -> dict[str, Any]:
    collision = config.collision_nu0_s * np.exp(
        -config.y_coordinates_m / config.collision_frequency_scale_height_m
    )
    coefficient = np.maximum(density, config.density_floor_m3) * collision[:, None]
    return point_centered_peclet_diagnostics(
        coefficient, config.dx_m, config.dy_m
    )


def _render(base: SimulationConfig) -> list[str]:
    paths = calibration_paths(base)
    control = np.asarray(load_frame(paths["control_frames"], OVERLAP_TIME_S), dtype=float)
    overlap, _, _, _ = _checkpoint_state(paths["candidate_overlap"])
    final_path = paths["candidate"] / "simulations" / "density_pdm7.npz"
    with np.load(final_path, allow_pickle=False) as saved:
        final = np.asarray(saved["density"], dtype=np.float64)
    x = base.x_coordinates_m / 1.0e3 - 125.0
    y = base.y_coordinates_m / 1.0e3
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.7), sharex=True, sharey=True)
    titles = (
        "a) Original point-centred\nt = 2324 s",
        "b) Exponential conservative\nt = 2324 s",
        f"c) Exponential conservative\nt = {TEST_END_TIME_S:g} s",
    )
    image = None
    for axis, density, title in zip(axes, (control, overlap, final), titles):
        image = axis.pcolormesh(
            x,
            y,
            density / 1.0e11,
            cmap="plasma",
            vmin=0.0,
            vmax=11.0,
            shading="auto",
            rasterized=True,
        )
        axis.set_title(title, loc="left")
        axis.set_xlabel("East-West (km)")
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
    axes[0].set_ylabel("Altitude (km)")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label(r"Plasma density ($10^5$ cm$^{-3}$)")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.14, top=0.84, wspace=0.08)
    outputs: list[str] = []
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        destination = paths["figure"].with_suffix(f".{suffix}")
        fig.savefig(destination, bbox_inches="tight", **kwargs)
        outputs.append(str(destination))
    plt.close(fig)
    return outputs


def write_report(base: SimulationConfig, status: dict[str, Any]) -> Path:
    paths = calibration_paths(base)
    config = _candidate_config(base)
    control_overlap = np.asarray(
        load_frame(paths["control_frames"], OVERLAP_TIME_S), dtype=np.float64
    )
    candidate_overlap, _, _, _ = _checkpoint_state(paths["candidate_overlap"])
    anchor, _, _, _ = _checkpoint_state(paths["anchor"])
    final, _, _, _ = _checkpoint_state(paths["candidate_checkpoint"])
    report = {
        "annotation": 1,
        "root_cause": (
            "the nonconservative point-centred potential stencil loses its "
            "M-matrix property and develops a non-physical near-zero mode"
        ),
        "control_failure_time_s": 2324.2163696289062,
        "candidate_status": status,
        "anchor_time_s": ANCHOR_TIME_S,
        "test_end_time_s": TEST_END_TIME_S,
        "overlap_time_s": OVERLAP_TIME_S,
        "overlap_agreement_with_original": _agreement(
            control_overlap, candidate_overlap
        ),
        "point_centered_health": {
            "anchor": _matrix_health(anchor, config),
            "candidate_final_density": _matrix_health(final, config),
        },
        "control_diagnostics": _diagnostic_summary(
            paths["control"] / "simulations" / "pdm7_diagnostics.jsonl"
        ),
        "candidate_diagnostics": _diagnostic_summary(
            paths["candidate"] / "simulations" / "pdm7_diagnostics.jsonl"
        ),
        "unchanged": [
            "Chapman initial state through the t=2300 stable anchor",
            "201x401 grid and equal-extrapolation vertical boundary",
            "PDM7 alpha=0.2 transport reconstruction and Rusanov flux",
            "AB3 predictor-corrector with SSPRK3 restart",
            "CFL limit 0.25 and PARDISO residual tolerance 1e-8",
        ],
        "changed": [
            "positive exponential-profile face conductances on the potential LHS",
            "compatible conservative RHS g*Dx(n)",
            "one eliminated gauge degree of freedom followed by zero-mean normalization",
        ],
    }
    if status.get("status") == "completed":
        report["figures"] = _render(base)
    _write_json(paths["report"], report)
    return paths["report"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/figure3.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    base = SimulationConfig.from_json(args.config).with_updates(pdm_alpha=0.2)
    paths = calibration_paths(base)
    status = (
        json.loads(paths["status"].read_text(encoding="utf-8"))
        if args.report_only
        else run_candidate(base)
    )
    report = write_report(base, status)
    print(f"[alpha02-stability] candidate={status['status']}", flush=True)
    print(f"[alpha02-stability] report={report}", flush=True)


if __name__ == "__main__":
    main()
