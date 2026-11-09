"""Stable public interface for the complete Figure 3 workflow."""

from __future__ import annotations

from pathlib import Path

from . import _figure3_impl as _impl
from ._figure3_public_impl import (
    FIGURE3_CASES,
    Figure3Case,
    create_figure3,
    write_figure3_metrics,
    write_figure3_source_data,
)


_prepare_figure3_snapshots_impl = _impl.prepare_figure3_snapshots


def prepare_figure3_snapshots(base, *, simulate=True, resume=True):
    """Archive every case configuration, then prepare all four snapshots."""

    output = base.output_directory / "figure3"
    for case in FIGURE3_CASES:
        config = base.with_updates(
            scheme="pdm7",
            pdm_alpha=case.alpha,
            t_end_s=case.time_s,
            output_dir=str(output / "runs" / case.key),
        )
        _impl._archive_config(
            config, output / "configs" / f"{case.key}.json", case
        )
    return _prepare_figure3_snapshots_impl(
        base, simulate=simulate, resume=resume
    )


def reproduce_figure3(base, *, simulate=True, resume=True):
    _impl.prepare_figure3_snapshots = prepare_figure3_snapshots
    _impl.write_figure3_metrics = write_figure3_metrics
    return _impl.reproduce_figure3(
        base, simulate=simulate, resume=resume
    )


__all__ = [
    "FIGURE3_CASES",
    "Figure3Case",
    "create_figure3",
    "prepare_figure3_snapshots",
    "reproduce_figure3",
    "write_figure3_metrics",
    "write_figure3_source_data",
]
