"""Best-baseline Figure 3 reproduction with one-second movie archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .config import (
    BEST_BASELINE_POTENTIAL_BOUNDARY,
    BEST_BASELINE_POTENTIAL_OPERATOR,
    SimulationConfig,
)
from .figure3 import (
    FIGURE3_CASES,
    create_figure3,
    write_figure3_metrics,
    write_figure3_source_data,
)
from . import _figure3_impl as figure3_impl
from .model import run_simulation
from .timeseries import validate_frame_archive


OUTPUT_NAME = "figure3_v2_timeseries"
FRAME_INTERVAL_S = 1.0
FRAME_CHUNK_SIZE = 100


def validate_v2_baseline(config: SimulationConfig) -> None:
    """Reject any accidental use of the later mimetic stability branch."""

    required = {
        "scheme": "pdm7",
        "nx": 201,
        "ny": 401,
        "potential_operator": BEST_BASELINE_POTENTIAL_OPERATOR,
        "potential_boundary_y": BEST_BASELINE_POTENTIAL_BOUNDARY,
    }
    mismatches = {
        key: {"expected": expected, "actual": getattr(config, key)}
        for key, expected in required.items()
        if getattr(config, key) != expected
    }
    if mismatches:
        raise ValueError(f"configuration is not the formal v2 baseline: {mismatches}")
    config.validate()


def _case_config(base: SimulationConfig, case, root: Path) -> SimulationConfig:
    return base.with_updates(
        scheme="pdm7",
        pdm_alpha=case.alpha,
        t_end_s=case.time_s,
        output_dir=str(root / "runs" / case.key),
    )


def _archive_config(config: SimulationConfig, case, root: Path) -> Path:
    destination = root / "configs" / f"{case.key}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload["figure3_case"] = {
        "alpha": case.alpha,
        "snapshot_time_s": case.time_s,
        "reference_variable": case.reference_variable,
    }
    payload["formal_best_baseline"] = {
        "version": "v2_equal_extrapolation_B",
        "potential_operator": BEST_BASELINE_POTENTIAL_OPERATOR,
        "potential_boundary_y": BEST_BASELINE_POTENTIAL_BOUNDARY,
        "linear_solver": "Intel MKL PARDISO",
    }
    payload["movie_archive"] = {
        "frame_interval_s": FRAME_INTERVAL_S,
        "storage_dtype": "float32",
        "solver_and_checkpoint_dtype": "float64",
        "chunk_frames": FRAME_CHUNK_SIZE,
        "includes_t0_and_target": True,
    }
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def _fast_series_complete(directory: Path, target_time_s: float) -> bool:
    manifest = directory / "manifest.json"
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return bool(
            payload["completed"]
            and payload["frame_count"] == int(round(target_time_s)) + 1
            and np.isclose(payload["target_time_s"], target_time_s)
            and payload["potential_operator"] == BEST_BASELINE_POTENTIAL_OPERATOR
            and payload["potential_boundary_y"] == BEST_BASELINE_POTENTIAL_BOUNDARY
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prepare_timeseries_snapshots(
    base: SimulationConfig,
    *,
    simulate: bool = True,
    resume: bool = True,
    case_keys: tuple[str, ...] | None = None,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Run selected cases from their own Chapman initial states."""

    validate_v2_baseline(base)
    available = {case.key for case in FIGURE3_CASES}
    selected = set(case_keys) if case_keys is not None else available
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(f"unknown Figure 3 cases: {unknown}")
    if not selected:
        raise ValueError("at least one Figure 3 case must be selected")
    root = base.output_directory / OUTPUT_NAME
    snapshots: dict[str, Path] = {}
    frame_reports: dict[str, dict[str, Any]] = {}
    for case in FIGURE3_CASES:
        if case.key not in selected:
            continue
        config = _case_config(base, case, root)
        _archive_config(config, case, root)
        run_snapshot = config.output_directory / "simulations" / "density_pdm7.npz"
        stable_snapshot = root / "simulations" / f"density_{case.key}.npz"
        frame_directory = root / "time_series" / case.key
        snapshot_complete = figure3_impl._snapshot_matches(
            run_snapshot,
            case.time_s,
            BEST_BASELINE_POTENTIAL_OPERATOR,
            BEST_BASELINE_POTENTIAL_BOUNDARY,
        )
        series_complete = _fast_series_complete(frame_directory, case.time_s)
        if simulate and not (snapshot_complete and series_complete):
            print(
                f"[esf_sim] Figure 3 v2 alpha={case.alpha:g}: "
                f"t=0..{case.time_s:g} s with one-second frames",
                flush=True,
            )
            run_snapshot = run_simulation(
                config,
                resume=resume,
                frame_archive_directory=frame_directory,
                frame_interval_s=FRAME_INTERVAL_S,
                frame_chunk_size=FRAME_CHUNK_SIZE,
            )
        if not figure3_impl._snapshot_matches(
            run_snapshot,
            case.time_s,
            BEST_BASELINE_POTENTIAL_OPERATOR,
            BEST_BASELINE_POTENTIAL_BOUNDARY,
        ):
            raise FileNotFoundError(f"missing complete v2 snapshot: {run_snapshot}")
        expected_times = np.arange(int(round(case.time_s)) + 1, dtype=np.float64)
        frame_reports[case.key] = validate_frame_archive(
            frame_directory,
            expected_times=expected_times,
            expected_shape=(base.ny, base.nx),
            full=True,
        )
        normalized = figure3_impl._normalize_snapshot(
            run_snapshot, stable_snapshot, case
        )
        snapshots[case.key] = normalized
        if case.key == "alpha_0p4":
            equivalence = _alpha04_v2_equivalence(root, normalized)
            if equivalence.get("available") and not equivalence.get("array_equal"):
                raise RuntimeError(
                    "alpha=0.4 does not reproduce the formal v2 baseline: "
                    f"{equivalence}"
                )
    return snapshots, frame_reports


def _alpha04_v2_equivalence(root: Path, snapshot: Path) -> dict[str, Any]:
    canonical = (
        root.parent
        / "figure3_stages"
        / "potential_vertical_boundary_ab"
        / "simulations"
        / "density_equal_extrapolated_alpha_0p4.npz"
    )
    result: dict[str, Any] = {
        "canonical_snapshot": str(canonical),
        "available": canonical.exists(),
    }
    if not canonical.exists():
        return result
    with np.load(canonical, allow_pickle=False) as left, np.load(
        snapshot, allow_pickle=False
    ) as right:
        canonical_density = np.asarray(left["density"], dtype=np.float64)
        reproduced_density = np.asarray(right["density"], dtype=np.float64)
    result.update(
        {
            "array_equal": bool(np.array_equal(canonical_density, reproduced_density)),
            "maximum_absolute_difference_m3": float(
                np.max(np.abs(canonical_density - reproduced_density))
            ),
            "rmse_m3": float(
                np.sqrt(np.mean((canonical_density - reproduced_density) ** 2))
            ),
            "canonical_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
            "reproduced_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        }
    )
    return result


def _write_movie_readme(root: Path) -> Path:
    path = root / "MOVIE_DATA_README.md"
    path.write_text(
        """# Figure 3 one-second density archives

Each `time_series/alpha_*` directory contains atomically written compressed NPZ
chunks. Every integer second from `t=0` through the panel target is present.
Movie frames are stored as `float32`; solver checkpoints and final snapshots are
`float64`.

```python
from esf_sim.timeseries import load_frame

density_m3 = load_frame(
    "output/best_baseline/figure3_v2_timeseries/time_series/alpha_0p4",
    1200,
)
```

Use `manifest.json` in each case directory for chronological coverage and units.
""",
        encoding="utf-8",
    )
    return path


def run_selected_timeseries_cases(
    base: SimulationConfig,
    case_keys: tuple[str, ...],
    *,
    simulate: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Run and validate a subset without requiring the unfinished panels."""

    validate_v2_baseline(base)
    root = base.output_directory / OUTPUT_NAME
    snapshots, frame_reports = prepare_timeseries_snapshots(
        base,
        simulate=simulate,
        resume=resume,
        case_keys=case_keys,
    )
    report: dict[str, Any] = {
        "formal_best_baseline": "v2_equal_extrapolation_B",
        "potential_operator": BEST_BASELINE_POTENTIAL_OPERATOR,
        "potential_boundary_y": BEST_BASELINE_POTENTIAL_BOUNDARY,
        "linear_solver": "Intel MKL PARDISO",
        "selected_cases": list(case_keys),
        "frame_interval_s": FRAME_INTERVAL_S,
        "total_frame_count": int(
            sum(item["frame_count"] for item in frame_reports.values())
        ),
        "cases": frame_reports,
    }
    if "alpha_0p4" in snapshots:
        report["alpha_0p4_v2_equivalence"] = _alpha04_v2_equivalence(
            root, snapshots["alpha_0p4"]
        )
    path = root / "selected_cases_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_movie_readme(root)
    return report


def reproduce_figure3_timeseries(
    base: SimulationConfig,
    *,
    simulate: bool = True,
    resume: bool = True,
) -> tuple[dict[str, Path], dict[str, Any]]:
    validate_v2_baseline(base)
    root = base.output_directory / OUTPUT_NAME
    snapshots, frame_reports = prepare_timeseries_snapshots(
        base, simulate=simulate, resume=resume
    )
    source_data = write_figure3_source_data(
        snapshots, root / "figure3_source_data.npz"
    )
    figures = create_figure3(snapshots, root)
    metrics = write_figure3_metrics(
        snapshots,
        reference_directory=Path(base.reference_mat).parent,
        output_path=root / "figure3_metrics.json",
    )
    shutil.copyfile(
        root / "figure3_metrics.json", root / "figure3_acceptance_report.json"
    )
    report = {
        "formal_best_baseline": "v2_equal_extrapolation_B",
        "potential_operator": BEST_BASELINE_POTENTIAL_OPERATOR,
        "potential_boundary_y": BEST_BASELINE_POTENTIAL_BOUNDARY,
        "linear_solver": "Intel MKL PARDISO",
        "frame_interval_s": FRAME_INTERVAL_S,
        "total_frame_count": int(sum(item["frame_count"] for item in frame_reports.values())),
        "cases": frame_reports,
        "alpha_0p4_v2_equivalence": _alpha04_v2_equivalence(
            root, snapshots["alpha_0p4"]
        ),
        "figures": {key: str(value) for key, value in figures.items()},
        "source_data": str(source_data),
        "reference_thresholds_pass": metrics["all_reference_thresholds_pass"],
        "alpha_trend_pass": metrics["all_scientific_alpha_trend_criteria_pass"],
    }
    report_path = root / "figure3_timeseries_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_movie_readme(root)
    return figures, report


__all__ = [
    "FRAME_CHUNK_SIZE",
    "FRAME_INTERVAL_S",
    "OUTPUT_NAME",
    "prepare_timeseries_snapshots",
    "reproduce_figure3_timeseries",
    "run_selected_timeseries_cases",
    "validate_v2_baseline",
]
