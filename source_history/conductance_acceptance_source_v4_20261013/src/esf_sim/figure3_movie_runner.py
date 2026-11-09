"""Compatibility entry point for exact 1920x1080 Figure 3 movie encoding."""

from __future__ import annotations

from typing import Any

import imageio_ffmpeg
import matplotlib as mpl

from . import figure3_movie as workflow


_BASE_STYLE = workflow._style


def _movie_style() -> None:
    _BASE_STYLE()
    # DejaVu Sans supplies the superscript-minus glyph used by the colourbar.
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def _exact_size_writer(
    path,
    *,
    width_px: int,
    height_px: int,
    fps: int,
) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width_px, height_px),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=[
            "-crf",
            "18",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            "-metadata",
            "title=ESF Figure 3 alpha-parameter evolution",
        ],
    )
    writer.send(None)
    return writer


def configure() -> None:
    workflow._style = _movie_style
    workflow._open_writer = _exact_size_writer


def create_movie(**kwargs):
    configure()
    return workflow.create_movie(**kwargs)


def main(argv: list[str] | None = None) -> None:
    configure()
    workflow.main(argv)


if __name__ == "__main__":
    main()
