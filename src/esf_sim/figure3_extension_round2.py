"""Second 1000-s continuation for the alpha=0.2 and alpha=0.4 Figure 3 cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Install tolerant archived-config loading before importing the continuation driver.
from . import figure3_extension_runner as _config_compatibility  # noqa: F401
from .figure3_extension import ExtensionCase, _atomic_json, run_case


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FIGURE3_ROOT = WORKSPACE_ROOT / "output" / "best_baseline" / "figure3_v2_timeseries"
ROUND1_ROOT = FIGURE3_ROOT / "extended_1000s"
DEFAULT_OUTPUT_ROOT = FIGURE3_ROOT / "extended_1000s_round2"


def round2_cases() -> tuple[ExtensionCase, ...]:
    return (
        ExtensionCase(
            "alpha_0p2",
            0.2,
            4150.0,
            5150.0,
            ROUND1_ROOT / "runs" / "alpha_0p2" / "config.json",
            ROUND1_ROOT / "runs" / "alpha_0p2" / "checkpoints" / "pdm7.npz",
        ),
        ExtensionCase(
            "alpha_0p4",
            0.4,
            4150.0,
            5150.0,
            ROUND1_ROOT / "runs" / "alpha_0p4" / "config.json",
            ROUND1_ROOT / "runs" / "alpha_0p4" / "checkpoints" / "pdm7.npz",
        ),
    )


def run_round2(
    selected: list[str],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    minimum_dt_s: float = 1.0 / 64.0,
    frame_chunk_size: int = 100,
) -> dict[str, Any]:
    definitions = {case.name: case for case in round2_cases()}
    output_root.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, Any] = {}
    for name in selected:
        try:
            statuses[name] = run_case(
                definitions[name],
                output_root,
                minimum_dt_s=minimum_dt_s,
                frame_chunk_size=frame_chunk_size,
            )
        except Exception as error:
            statuses[name] = {
                "case": name,
                "state": "launcher_failed",
                "reason": f"{type(error).__name__}: {error}",
            }
            _atomic_json(output_root / "runs" / name / "status.json", statuses[name])
        _atomic_json(
            output_root / "extension_round2_report.json",
            {
                "format": "esf_figure3_extension_round2_v1",
                "requested_additional_seconds": 1000.0,
                "cumulative_target_time_s": 5150.0,
                "minimum_dt_s": minimum_dt_s,
                "cases": statuses,
            },
        )
    return statuses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=["alpha_0p2", "alpha_0p4"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--minimum-dt", type=float, default=1.0 / 64.0)
    parser.add_argument("--frame-chunk-size", type=int, default=100)
    args = parser.parse_args(argv)
    selected = args.case or ["alpha_0p2", "alpha_0p4"]
    statuses = run_round2(
        selected,
        output_root=args.output_root,
        minimum_dt_s=args.minimum_dt,
        frame_chunk_size=args.frame_chunk_size,
    )
    print(json.dumps(statuses, indent=2, ensure_ascii=False))
    return 1 if any(x["state"] in {"failed", "launcher_failed"} for x in statuses.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
