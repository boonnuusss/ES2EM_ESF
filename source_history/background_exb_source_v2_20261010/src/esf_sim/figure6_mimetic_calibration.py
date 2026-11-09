"""Short-window calibration of a mimetic Figure 6 potential operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import _model_impl
from .config import SimulationConfig
from .experimental_boundary import build_harmonic_neumann_matrix
from .figure6 import case_by_key, case_config
from .figure6_stability import (
    ANCHOR_TIME_S,
    CASE_KEY,
    _diagnostic_summary,
    _fixed_checkpoint_state,
    _write_json,
    calibration_paths,
    prepare_anchor,
)
from .model import run_simulation
from .point_centered_potential import LogPointCenteredPotentialSolver
from .potential import PotentialDiagnostics


TEST_END_TIME_S = 2400.0
_BASE_ESF_MODEL = _model_impl.ESFModel


class HarmonicMimeticLogPotentialSolver(LogPointCenteredPotentialSolver):
    """Positive conservative left operator with the calibrated log RHS."""

    def solve(self, density: np.ndarray):
        n = np.asarray(density, dtype=np.float64)
        positive_n = np.maximum(n, self.config.density_floor_m3)
        coefficient = positive_n * self.collision_frequency[:, None]
        matrix, scale = build_harmonic_neumann_matrix(
            coefficient, self.config.dx_m, self.config.dy_m
        )
        log_n = np.log(positive_n)
        log_density_dx = (
            np.roll(log_n, -1, axis=1) - np.roll(log_n, 1, axis=1)
        ) / (2.0 * self.config.dx_m)
        divided_rhs = (
            self.config.gravity_m_s2
            * log_density_dx
            / self.collision_frequency[:, None]
        )
        rhs = (coefficient / scale * divided_rhs).ravel()
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


class MimeticESFModel(_BASE_ESF_MODEL):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.potential_solver = HarmonicMimeticLogPotentialSolver(config)


def _paths(base: SimulationConfig) -> dict[str, Path]:
    common = calibration_paths(base)
    root = common["root"] / "mimetic_harmonic_log_rhs"
    return {
        "root": root,
        "checkpoint": root / "checkpoints" / "pdm7.npz",
        "status": root / "status.json",
        "report": common["root"] / "mimetic_calibration_report.json",
    }


def _config(base: SimulationConfig) -> SimulationConfig:
    case = case_by_key(CASE_KEY)
    return case_config(base, case, _paths(base)["root"]).with_updates(
        t_end_s=TEST_END_TIME_S
    )


def run_candidate(base: SimulationConfig) -> dict[str, Any]:
    paths = _paths(base)
    if paths["status"].exists():
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            return status
    anchor = prepare_anchor(base)
    if not paths["checkpoint"].exists():
        paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(anchor, paths["checkpoint"])
    config = _config(base)
    payload = config.to_dict()
    payload["calibration_role"] = "mimetic_harmonic_left_logarithmic_rhs"
    payload["changed_factor_only"] = (
        "replace the nonconservative centered drift matrix by the conservative "
        "harmonic-face five-point matrix"
    )
    _write_json(paths["root"] / "config.json", payload)
    original = _model_impl.ESFModel
    _model_impl.ESFModel = MimeticESFModel
    try:
        try:
            snapshot = run_simulation(config, resume=True)
            status: dict[str, Any] = {
                "status": "completed",
                "snapshot": str(snapshot),
            }
        except (RuntimeError, FloatingPointError) as error:
            _, time_s, dt_s, step = _fixed_checkpoint_state(paths["checkpoint"])
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


def _load_density(root: Path) -> tuple[np.ndarray, float]:
    snapshot = root / "simulations" / "density_pdm7.npz"
    if snapshot.exists():
        with np.load(snapshot) as saved:
            return np.asarray(saved["density"], dtype=np.float64), float(saved["time_s"])
    density, time_s, _, _ = _fixed_checkpoint_state(root / "checkpoints" / "pdm7.npz")
    return density, time_s


def _render(base: SimulationConfig) -> list[str]:
    common = calibration_paths(base)
    paths = _paths(base)
    control, control_time = _load_density(common["control"])
    candidate, candidate_time = _load_density(paths["root"])
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
        f"a) Nonconservative log-centered control\nfailed, t = {control_time:.3f} s",
        f"b) Conservative harmonic/log-RHS candidate\nt = {candidate_time:.3f} s",
    )
    image = None
    for axis, density, title in zip(axes, (control, candidate), titles):
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
    colorbar = fig.colorbar(image, ax=axes, fraction=0.045, pad=0.04)
    colorbar.set_label(r"Plasma density ($10^5$ cm$^{-3}$)")
    fig.subplots_adjust(left=0.10, right=0.87, bottom=0.13, top=0.86, wspace=0.08)
    stem = paths["report"].with_name("mimetic_calibration")
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


def write_report(base: SimulationConfig, status: dict[str, Any]) -> Path:
    paths = _paths(base)
    common = calibration_paths(base)
    report = {
        "case": CASE_KEY,
        "anchor_time_s": ANCHOR_TIME_S,
        "test_end_time_s": TEST_END_TIME_S,
        "candidate_status": status,
        "changed_factor_only": (
            "left-hand potential matrix: nonconservative log-centered drift "
            "to conservative harmonic-face five-point"
        ),
        "unchanged": [
            "PDM7 alpha=0.4",
            "AB3 predictor-corrector and SSPRK3 restart policy",
            "equal-extrapolation vertical potential boundary",
            "all physical parameters and initial condition",
            "Intel MKL PARDISO and 1e-8 residual threshold",
            "logarithmic centered right-hand-side density gradient",
        ],
        "control": {
            "status": json.loads(
                (common["control"] / "status.json").read_text(encoding="utf-8")
            ),
            "diagnostics": _diagnostic_summary(
                common["control"] / "simulations" / "pdm7_diagnostics.jsonl"
            ),
        },
        "candidate": {
            "status": status,
            "diagnostics": _diagnostic_summary(
                paths["root"] / "simulations" / "pdm7_diagnostics.jsonl"
            ),
        },
    }
    report["figures"] = _render(base)
    _write_json(paths["report"], report)
    return paths["report"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/figure6.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    base = SimulationConfig.from_json(args.config)
    status = (
        json.loads(_paths(base)["status"].read_text(encoding="utf-8"))
        if args.report_only
        else run_candidate(base)
    )
    print(f"[mimetic] candidate: {status['status']}", flush=True)
    report = write_report(base, status)
    print(f"[mimetic] report: {report}", flush=True)


if __name__ == "__main__":
    main()
