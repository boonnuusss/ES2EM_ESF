"""Calibrate Figure 6 high-resolution time stepping from a stable state."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .experimental_time import (
    convert_fixed_history_checkpoint,
    run_variable_history_simulation,
)
from .figure6 import case_by_key, case_config
from .model import run_simulation


ANCHOR_TIME_S = 2300.0
TEST_END_TIME_S = 2330.0
CASE_KEY = "grid_301x601"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _fixed_checkpoint_state(path: Path) -> tuple[np.ndarray, float, float, int]:
    with np.load(path) as saved:
        return (
            np.asarray(saved["density"], dtype=np.float64),
            float(saved["time_s"]),
            float(saved["dt_s"]),
            int(saved["step"]),
        )


def _variable_checkpoint_state(path: Path) -> tuple[np.ndarray, float, float, int]:
    with np.load(path) as saved:
        states = np.asarray(saved["states"], dtype=np.float64)
        times = np.asarray(saved["times"], dtype=np.float64)
        return states[0], float(times[0]), float(saved["dt_s"]), int(saved["step"])


def _checkpoint_matches(path: Path, expected_time_s: float, *, variable: bool) -> bool:
    if not path.exists():
        return False
    try:
        loader = _variable_checkpoint_state if variable else _fixed_checkpoint_state
        _, time_s, _, _ = loader(path)
        return bool(np.isclose(time_s, expected_time_s))
    except (OSError, KeyError, ValueError):
        return False


def _diagnostic_summary(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        return {"record_count": 0}
    last = rows[-1]
    return {
        "record_count": len(rows),
        "first_time_s": float(rows[0]["time_s"]),
        "last_time_s": float(last["time_s"]),
        "last_dt_s": float(last["dt_s"]),
        "last_cfl": float(last["cfl"]),
        "maximum_cfl": float(max(row["cfl"] for row in rows)),
        "maximum_speed_x_m_s": float(max(row["maximum_speed_x_m_s"] for row in rows)),
        "maximum_speed_y_m_s": float(max(row["maximum_speed_y_m_s"] for row in rows)),
        "maximum_potential_relative_residual": float(
            max(row["potential_relative_residual"] for row in rows)
        ),
        "last_minimum_density_m3": float(last["minimum_density_m3"]),
        "last_mass_relative_change": float(last["mass_relative_change"]),
    }


def calibration_paths(base: SimulationConfig) -> dict[str, Path]:
    root = base.output_directory / "figure6_stability_calibration"
    return {
        "root": root,
        "anchor": root / "anchor_301x601_t2300",
        "control": root / "control_restart_ssprk3",
        "variable": root / "variable_history_ab3",
        "report": root / "stability_ab_report.json",
    }


def _calibration_config(
    base: SimulationConfig, output: Path, t_end_s: float
) -> SimulationConfig:
    case = case_by_key(CASE_KEY)
    return case_config(base, case, output).with_updates(t_end_s=t_end_s)


def _archive_config(path: Path, config: SimulationConfig, role: str) -> None:
    payload = config.to_dict()
    payload["calibration_role"] = role
    payload["calibration_case"] = CASE_KEY
    _write_json(path, payload)


def prepare_anchor(
    base: SimulationConfig,
    *,
    anchor_time_s: float = ANCHOR_TIME_S,
) -> Path:
    paths = calibration_paths(base)
    root = paths["anchor"]
    checkpoint = root / "checkpoints" / "pdm7.npz"
    config = _calibration_config(base, root, anchor_time_s)
    _archive_config(root / "config.json", config, "stable_anchor")
    if _checkpoint_matches(checkpoint, anchor_time_s, variable=False):
        return checkpoint
    run_simulation(config, resume=True)
    if not _checkpoint_matches(checkpoint, anchor_time_s, variable=False):
        raise RuntimeError("stable anchor did not reach the requested time")
    return checkpoint


def _initialize_branches(base: SimulationConfig, anchor: Path) -> None:
    paths = calibration_paths(base)
    control_checkpoint = paths["control"] / "checkpoints" / "pdm7.npz"
    if not control_checkpoint.exists():
        control_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(anchor, control_checkpoint)
    variable_checkpoint = paths["variable"] / "checkpoints" / "variable_history.npz"
    if not variable_checkpoint.exists():
        convert_fixed_history_checkpoint(anchor, variable_checkpoint)


def _run_control(base: SimulationConfig, test_end_time_s: float) -> dict[str, Any]:
    paths = calibration_paths(base)
    root = paths["control"]
    status_path = root / "status.json"
    if status_path.exists():
        saved = json.loads(status_path.read_text(encoding="utf-8"))
        if saved.get("status") in {"completed", "failed"}:
            return saved
    config = _calibration_config(base, root, test_end_time_s)
    _archive_config(root / "config.json", config, "control_ssprk3_history_restart")
    status: dict[str, Any]
    try:
        snapshot = run_simulation(config, resume=True)
        status = {"status": "completed", "snapshot": str(snapshot)}
    except (RuntimeError, FloatingPointError) as error:
        checkpoint = root / "checkpoints" / "pdm7.npz"
        _, time_s, dt_s, step = _fixed_checkpoint_state(checkpoint)
        status = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "last_time_s": time_s,
            "last_dt_s": dt_s,
            "last_step": step,
        }
    _write_json(status_path, status)
    return status


def _run_variable(base: SimulationConfig, test_end_time_s: float) -> dict[str, Any]:
    paths = calibration_paths(base)
    root = paths["variable"]
    status_path = root / "status.json"
    if status_path.exists():
        saved = json.loads(status_path.read_text(encoding="utf-8"))
        if saved.get("status") == "completed":
            return saved
    config = _calibration_config(base, root, test_end_time_s)
    _archive_config(root / "config.json", config, "variable_history_no_ssprk_restart")
    try:
        snapshot = run_variable_history_simulation(config, resume=True)
        status: dict[str, Any] = {"status": "completed", "snapshot": str(snapshot)}
    except (RuntimeError, FloatingPointError) as error:
        checkpoint = root / "checkpoints" / "variable_history.npz"
        _, time_s, dt_s, step = _variable_checkpoint_state(checkpoint)
        status = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "last_time_s": time_s,
            "last_dt_s": dt_s,
            "last_step": step,
        }
    _write_json(status_path, status)
    return status


def _state_for_panel(root: Path, *, variable: bool) -> tuple[np.ndarray, float]:
    snapshot = root / "simulations" / "density_pdm7.npz"
    if snapshot.exists():
        with np.load(snapshot) as saved:
            return np.asarray(saved["density"], dtype=np.float64), float(saved["time_s"])
    checkpoint = root / "checkpoints" / ("variable_history.npz" if variable else "pdm7.npz")
    loader = _variable_checkpoint_state if variable else _fixed_checkpoint_state
    density, time_s, _, _ = loader(checkpoint)
    return density, time_s


def _render_ab(base: SimulationConfig) -> list[str]:
    paths = calibration_paths(base)
    control, time_control = _state_for_panel(paths["control"], variable=False)
    variable, time_variable = _state_for_panel(paths["variable"], variable=True)
    case = case_by_key(CASE_KEY)
    x = np.linspace(-125.0, 125.0, case.nx)
    y = np.linspace(250.0, 650.0, case.ny)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)
    titles = (
        f"a) SSPRK3 history restart\nfailed/control, t = {time_control:.3f} s",
        f"b) Variable-step AB3 history\nt = {time_variable:.3f} s",
    )
    image = None
    for axis, density, title in zip(axes, (control, variable), titles):
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
        axis.set_xlabel("East?West (km)")
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
    axes[0].set_ylabel("Altitude (km)")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.045, pad=0.04)
    colorbar.set_label(r"Plasma density ($10^5$ cm$^{-3}$)")
    fig.subplots_adjust(left=0.10, right=0.87, bottom=0.13, top=0.86, wspace=0.08)
    stem = paths["root"] / "stability_ab"
    outputs: list[str] = []
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600}),
    ):
        output = stem.with_suffix(f".{suffix}")
        fig.savefig(output, bbox_inches="tight", **kwargs)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def write_report(
    base: SimulationConfig,
    control_status: dict[str, Any],
    variable_status: dict[str, Any],
) -> Path:
    paths = calibration_paths(base)
    control_diag = _diagnostic_summary(
        paths["control"] / "simulations" / "pdm7_diagnostics.jsonl"
    )
    variable_diag = _diagnostic_summary(
        paths["variable"] / "simulations" / "pdm7_diagnostics.jsonl"
    )
    selected = None
    if variable_status.get("status") == "completed":
        selected = "variable_step_lagrange_half_step_ab3"
    payload = {
        "case": CASE_KEY,
        "anchor_time_s": ANCHOR_TIME_S,
        "test_end_time_s": TEST_END_TIME_S,
        "fixed_scientific_settings": {
            "potential_operator": base.potential_operator,
            "potential_boundary_y": base.potential_boundary_y,
            "linear_solver": "Intel MKL PARDISO",
            "spatial_scheme": "seventh-order upwind reconstruction with PDM alpha=0.4",
        },
        "control": {"status": control_status, "diagnostics": control_diag},
        "variable_history": {"status": variable_status, "diagnostics": variable_diag},
        "selected_time_integration": selected,
        "selection_rule": (
            "Select variable-step history only if it crosses the control instability "
            "window with positive density, CFL <= 0.25, bounded velocity, and an "
            "elliptic residual <= 1e-8."
        ),
    }
    if paths["control"].exists() and paths["variable"].exists():
        payload["figures"] = _render_ab(base)
    _write_json(paths["report"], payload)
    return paths["report"]


def run_calibration(base: SimulationConfig) -> Path:
    anchor = prepare_anchor(base)
    _initialize_branches(base, anchor)
    control = _run_control(base, TEST_END_TIME_S)
    variable = _run_variable(base, TEST_END_TIME_S)
    return write_report(base, control, variable)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/figure6.json")
    parser.add_argument(
        "--stage",
        choices=("anchor", "ab", "report", "all"),
        default="all",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    base = SimulationConfig.from_json(args.config)
    paths = calibration_paths(base)
    if args.stage in {"anchor", "all"}:
        anchor = prepare_anchor(base)
        print(f"[stability] stable anchor: {anchor}", flush=True)
    else:
        anchor = paths["anchor"] / "checkpoints" / "pdm7.npz"
    if args.stage in {"ab", "all"}:
        if not _checkpoint_matches(anchor, ANCHOR_TIME_S, variable=False):
            raise FileNotFoundError("run the anchor stage first")
        _initialize_branches(base, anchor)
        control = _run_control(base, TEST_END_TIME_S)
        print(f"[stability] control: {control['status']}", flush=True)
        variable = _run_variable(base, TEST_END_TIME_S)
        print(f"[stability] variable: {variable['status']}", flush=True)
        report = write_report(base, control, variable)
        print(f"[stability] report: {report}", flush=True)
    elif args.stage == "report":
        control = json.loads((paths["control"] / "status.json").read_text(encoding="utf-8"))
        variable = json.loads((paths["variable"] / "status.json").read_text(encoding="utf-8"))
        report = write_report(base, control, variable)
        print(f"[stability] report: {report}", flush=True)


if __name__ == "__main__":
    main()
