"""Corrected optimized alpha=0.2 run from the paper Chapman state to 5050 s."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import initial_density as paper_exact_initial_density
from . import optimized_alpha04_run as workflow


DEFAULT_OUTPUT = Path("output/optimization/alpha02_to_5050_corrected")


def run(config_path="configs/figure3.json", output=DEFAULT_OUTPUT, *, resume=True):
    original_with_updates = workflow.SimulationConfig.with_updates

    def alpha02_updates(self, **updates):
        # Reuse the audited restart/archive driver while replacing only the
        # hard-coded case definition.
        if updates.get("pdm_alpha") == 0.4 and updates.get("t_end_s") == 5011.0:
            updates["pdm_alpha"] = 0.2
            updates["t_end_s"] = 5050.0
        return original_with_updates(self, **updates)

    workflow.initial_density = paper_exact_initial_density
    workflow.SimulationConfig.with_updates = alpha02_updates
    try:
        report = workflow.run(config_path, output, resume=resume)
    finally:
        workflow.SimulationConfig.with_updates = original_with_updates

    output = Path(output)
    old_snapshot = output/"simulations"/"density_optimized_alpha_0p4_t5011.npz"
    snapshot = output/"simulations"/"density_optimized_alpha_0p2_t5050.npz"
    if old_snapshot.exists():
        old_snapshot.replace(snapshot)
    report["snapshot"] = str(snapshot.resolve())
    report["case"] = {"pdm_alpha": 0.2, "target_time_s": 5050.0}
    report["initial_condition"] = {
        "source": "esf_sim.model.initial_density",
        "chapman_exponent": "1 - z - exp(-z)",
        "matches_formal_v2_baseline": True,
    }
    (output/"optimized_run_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/figure3.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, resume=not args.no_resume), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
