"""Public numerical-validation suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._validation_impl import (
    _ssprk3,
    circular_advection_test,
    elliptic_manufactured_solution_test,
    slotted_cylinder_initial,
)


def temporal_accuracy_test() -> dict[str, object]:
    """Measure the order of the published half-step predictor/corrector."""

    errors = []
    time_steps = (0.025, 0.0125, 0.00625, 0.003125, 0.0015625)
    for dt in time_steps:
        history = [np.asarray([1.0])]
        nsteps = int(round(1.0 / dt))
        rhs = lambda value: -value
        for _ in range(nsteps):
            if len(history) < 3:
                updated = _ssprk3(history[0], dt, rhs)
            else:
                predictor = (
                    15.0 / 8.0 * history[0]
                    - 5.0 / 4.0 * history[1]
                    + 3.0 / 8.0 * history[2]
                )
                updated = history[0] + dt * rhs(predictor)
            history.insert(0, updated)
            del history[3:]
        errors.append(abs(float(history[0][0]) - float(np.exp(-1.0))))
    rates = [
        float(np.log(errors[index] / errors[index + 1]) / np.log(2.0))
        for index in range(len(errors) - 1)
    ]
    return {
        "time_steps": list(time_steps),
        "absolute_errors": errors,
        "observed_orders": rates,
        "expected_asymptotic_order_for_published_form": 2.0,
        "accuracy_pass": bool(min(rates[-2:]) >= 1.9),
        "note": (
            "The paper names the method AB3, but its printed half-step extrapolation "
            "followed by one flux update is asymptotically second order on this scalar test."
        ),
    }


def run_validation_suite(output_directory: str | Path) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "circular_advection": circular_advection_test(output),
        "elliptic_manufactured_solution": elliptic_manufactured_solution_test(),
        "temporal_accuracy": temporal_accuracy_test(),
    }
    report["all_pass"] = bool(
        all(report["circular_advection"]["criteria"].values())
        and report["elliptic_manufactured_solution"]["second_order_pass"]
        and report["temporal_accuracy"]["accuracy_pass"]
    )
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
