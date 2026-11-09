"""Quantitative comparison accompanying the alpha=0.4 movie."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import morphology_metrics
from .optimized_alpha04_movie import ArchiveReader, BaselineTimeline


def _compare(a,b,x,y):
    difference=np.asarray(b,dtype=np.float64)-np.asarray(a,dtype=np.float64)
    ma=morphology_metrics(a,x,y); mb=morphology_metrics(b,x,y)
    return {
        "spatial_correlation":float(np.corrcoef(a.ravel(),b.ravel())[0,1]),
        "relative_l2_difference":float(np.linalg.norm(difference)/np.linalg.norm(a)),
        "rmse_m3":float(np.sqrt(np.mean(difference*difference))),
        "maximum_absolute_difference_m3":float(np.max(np.abs(difference))),
        "baseline_morphology":ma.to_dict(),"optimized_morphology":mb.to_dict(),
    }


def main():
    baseline=BaselineTimeline(); optimized=ArchiveReader("output/optimization/alpha04_to_5011/time_series")
    x=np.arange(201)*1.25-125; y=250+np.arange(401)
    run=json.loads(Path("output/optimization/alpha04_to_5011/optimized_run_report.json").read_text(encoding="utf-8"))
    report={"operator_switch_time_s":run["operator_switch_time_s"],"comparisons":{}}
    for time_s in (3150,3772,4150,5011):
        report["comparisons"][str(time_s)]=_compare(
            baseline.frame(time_s),optimized.frame(time_s),x,y
        )
    path=Path("output/optimization/alpha04_baseline_vs_optimized_movie/comparison_metrics.json")
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
