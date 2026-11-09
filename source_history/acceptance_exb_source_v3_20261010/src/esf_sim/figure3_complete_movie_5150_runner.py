"""Render the 5150-s movie with a corrected progress-axis terminal tick."""

from __future__ import annotations

from . import figure3_complete_movie_5150 as workflow


_BASE_CANVAS = workflow._create_canvas_5150


def _corrected_canvas(timelines, width, height, dpi):
    result = list(_BASE_CANVAS(timelines, width, height, dpi))
    figure = result[0]
    progress_line = result[5]
    progress_line.axes.set_xticks([0, 900, 1800, 2700, 3600, 4500, 5150])
    figure.canvas.draw()
    result[-1] = figure.canvas.copy_from_bbox(figure.bbox)
    return tuple(result)


workflow._create_canvas_5150 = _corrected_canvas


def main(argv: list[str] | None = None) -> None:
    workflow.main(argv)


if __name__ == "__main__":
    main()
