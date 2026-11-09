"""Public entry point for the vertical-potential boundary experiment."""

from __future__ import annotations

import argparse
import json

from matplotlib.figure import Figure

from . import _experimental_boundary_split_impl as _impl
from ._experimental_boundary_split_impl import *  # noqa: F401,F403
from .config import SimulationConfig


def render_boundary_comparison(base, dirichlet_snapshots):
    """Render the comparison with a dedicated, non-overlapping colorbar."""

    original_colorbar = Figure.colorbar

    def fixed_colorbar(figure, mappable, *args, **kwargs):
        if "ax" in kwargs:
            kwargs.pop("ax", None)
            kwargs.pop("fraction", None)
            kwargs.pop("pad", None)
            kwargs.pop("aspect", None)
            kwargs["cax"] = figure.add_axes([0.738, 0.19, 0.011, 0.62])
        return original_colorbar(figure, mappable, *args, **kwargs)

    Figure.colorbar = fixed_colorbar
    try:
        return _impl.render_boundary_comparison(base, dirichlet_snapshots)
    finally:
        Figure.colorbar = original_colorbar


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m esf_sim.experimental_boundary_split"
    )
    parser.add_argument("--config", default="configs/figure2.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    base = SimulationConfig.from_json(args.config)
    snapshots = _impl.run_dirichlet_alpha06(base, resume=not args.no_resume)
    if args.render:
        print(json.dumps(render_boundary_comparison(base, snapshots), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
