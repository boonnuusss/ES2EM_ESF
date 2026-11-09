"""Run the second Figure 3 continuation and then render the 0--5150 s movie."""

from __future__ import annotations

import json

from .figure3_complete_movie_5150 import create_movie_5150
from .figure3_extension_round2 import run_round2


def main() -> int:
    statuses = run_round2(["alpha_0p2", "alpha_0p4"])
    print("[round2-pipeline] continuation states:", flush=True)
    print(json.dumps(statuses, indent=2, ensure_ascii=False), flush=True)
    invalid = {
        name: status
        for name, status in statuses.items()
        if status["state"] not in {"completed", "paused_minimum_dt"}
    }
    if invalid:
        print(f"[round2-pipeline] movie skipped due to invalid states: {invalid}", flush=True)
        return 1
    report = create_movie_5150()
    print(f"[round2-pipeline] movie QA={report['all_checks_pass']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
