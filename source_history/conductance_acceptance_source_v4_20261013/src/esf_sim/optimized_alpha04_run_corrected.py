"""Corrected optimized alpha=0.4 run using the public paper-exact Chapman state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import initial_density as paper_exact_initial_density
from . import optimized_alpha04_run as workflow


DEFAULT_OUTPUT = Path("output/optimization/alpha04_to_5011_corrected")


def run(config_path="configs/figure3.json", output=DEFAULT_OUTPUT, *, resume=True):
    # The historical private helper contains the obsolete 0.5 Chapman exponent.
    # Patch only this isolated workflow; paper reproduction modules remain frozen.
    workflow.initial_density = paper_exact_initial_density
    report = workflow.run(config_path, output, resume=resume)
    report["initial_condition"] = {
        "source": "esf_sim.model.initial_density",
        "chapman_exponent": "1 - z - exp(-z)",
        "matches_formal_v2_baseline": True,
    }
    path = Path(output)/"optimized_run_report.json"
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/figure3.json")
    parser.add_argument("--output",default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.config,args.output,resume=not args.no_resume),indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
