"""Compatibility entry point for archived Figure 3 configs with provenance keys."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

from .config import SimulationConfig


def _tolerant_from_json(cls, path: str | Path) -> SimulationConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in allowed})


SimulationConfig.from_json = classmethod(_tolerant_from_json)

from .figure3_extension import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
