"""Render the Figure 3 movie through the second continuation target at t=5150 s."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# This compatibility import installs seekable archive access after seam checks.
from . import figure3_complete_movie_runner as _seek_compatibility  # noqa: F401
from . import figure3_complete_movie as workflow


DEFAULT_ROOT = Path("output/best_baseline/figure3_v2_timeseries")
DEFAULT_OUTPUT = DEFAULT_ROOT / "complete_movie_5150"
ROUND2_DIRECTORY = "extended_1000s_round2"
MOVIE_END_S = 5150


class Round2Timeline:
    """Join the canonical, first continuation, and second continuation segments."""

    def __init__(self, root: Path, case: workflow.CompleteMovieCase) -> None:
        self.root = root
        self.case = case
        self.base = workflow.ArchiveSegment(
            root / case.base_directory,
            expected_first_s=0,
            expected_last_s=3150,
            expected_alpha=case.alpha,
        )
        self.extension1 = workflow.ArchiveSegment(
            root / case.extension_directory,
            expected_first_s=3150,
            expected_last_s=4150,
            expected_alpha=case.alpha,
        )
        round2_archive = root / ROUND2_DIRECTORY / "time_series" / case.key
        round2_manifest = json.loads(
            (round2_archive / "manifest.json").read_text(encoding="utf-8")
        )
        extension2_last = int(round(float(round2_manifest["chunks"][-1]["last_time_s"])))
        self.extension2 = workflow.ArchiveSegment(
            round2_archive,
            expected_first_s=4150,
            expected_last_s=extension2_last,
            expected_alpha=case.alpha,
        )
        status_path = root / ROUND2_DIRECTORY / "runs" / case.key / "status.json"
        self.status = json.loads(status_path.read_text(encoding="utf-8"))
        if self.status["state"] not in {"completed", "paused_minimum_dt"}:
            raise ValueError(f"round-2 continuation is not terminal for {case.key}")
        if self.status["state"] != case.terminal_state or not math.isclose(
            float(self.status["time_s"]), case.terminal_time_s, abs_tol=1.0e-9
        ):
            raise ValueError(f"terminal status mismatch for {case.key}")
        snapshot = root / ROUND2_DIRECTORY / "runs" / case.key / "simulations" / "density_pdm7.npz"
        with np.load(snapshot, allow_pickle=False) as saved:
            self.terminal_density = np.asarray(saved["density"], dtype=np.float32)
            snapshot_time = float(saved["time_s"])
        if not math.isclose(snapshot_time, case.terminal_time_s, abs_tol=1.0e-9):
            raise ValueError(f"terminal snapshot mismatch for {case.key}")
        seam1_left = self.base.frame(3150)
        seam1_right = self.extension1.frame(3150)
        seam2_left = self.extension1.frame(4150)
        seam2_right = self.extension2.frame(4150)
        self.seam1_exact = bool(np.array_equal(seam1_left, seam1_right))
        self.seam2_exact = bool(np.array_equal(seam2_left, seam2_right))
        self.seam_exact = self.seam1_exact and self.seam2_exact
        if not self.seam_exact:
            raise ValueError(f"non-identical archive seam for {case.key}")

    @property
    def last_integer_time_s(self) -> int:
        return self.extension2.last_time_s

    def frame(self, global_time_s: int) -> np.ndarray:
        if global_time_s <= 3150:
            return self.base.frame(global_time_s)
        if global_time_s <= 4150:
            return self.extension1.frame(global_time_s)
        if global_time_s <= self.extension2.last_time_s:
            return self.extension2.frame(global_time_s)
        return self.terminal_density

    def is_held(self, global_time_s: int) -> bool:
        return global_time_s > self.extension2.last_time_s

    def hold_label(self) -> str:
        if self.case.terminal_state == "completed":
            return f"final state held at t = {self.case.terminal_time_s:.0f} s"
        return (
            f"simulation paused at t = {self.case.terminal_time_s:.4g} s\n"
            "minimum-step gate reached"
        )

    def report(self) -> dict[str, Any]:
        extension1 = self.extension1.report()
        extension2 = self.extension2.report()
        return {
            "alpha": self.case.alpha,
            "base": self.base.report(),
            "extension": {
                "all_chunks_loaded": bool(
                    extension1["all_chunks_loaded"] and extension2["all_chunks_loaded"]
                ),
                "first_continuation": extension1,
                "second_continuation": extension2,
            },
            "seam_time_s": [3150, 4150],
            "seam_float32_exact": self.seam_exact,
            "seam_3150_exact": self.seam1_exact,
            "seam_4150_exact": self.seam2_exact,
            "terminal_state": self.case.terminal_state,
            "terminal_time_s": self.case.terminal_time_s,
            "last_integer_archive_time_s": self.last_integer_time_s,
            "hold_begins_at_global_time_s": self.last_integer_time_s + 1,
            "hold_source": "terminal float64 snapshot converted to float32",
            "pause_reason": self.status.get("reason"),
        }


def _terminal_case(root: Path, key: str, alpha: float) -> workflow.CompleteMovieCase:
    status = json.loads(
        (root / ROUND2_DIRECTORY / "runs" / key / "status.json").read_text(encoding="utf-8")
    )
    base_directory = (
        "time_series/alpha_0p2_recovered" if key == "alpha_0p2" else "time_series/alpha_0p4"
    )
    return workflow.CompleteMovieCase(
        key,
        alpha,
        base_directory,
        3150,
        f"extended_1000s/time_series/{key}",
        status["state"],
        float(status["time_s"]),
    )


def build_cases(root: Path) -> tuple[workflow.CompleteMovieCase, ...]:
    return (
        _terminal_case(root, "alpha_0p2", 0.2),
        _terminal_case(root, "alpha_0p4", 0.4),
        workflow.CompleteMovieCase(
            "alpha_0p6",
            0.6,
            "time_series/alpha_0p6",
            3050,
            "extended_1000s/time_series/alpha_0p6",
            "paused_minimum_dt",
            3063.5625,
        ),
        workflow.CompleteMovieCase(
            "alpha_0p8",
            0.8,
            "time_series/alpha_0p8",
            2860,
            "extended_1000s/time_series/alpha_0p8",
            "paused_minimum_dt",
            2874.90625,
        ),
    )


def prepare_timelines_5150(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    cases = build_cases(root)
    timelines: dict[str, Any] = {
        cases[0].key: Round2Timeline(root, cases[0]),
        cases[1].key: Round2Timeline(root, cases[1]),
        cases[2].key: workflow.CompleteTimeline(root, cases[2]),
        cases[3].key: workflow.CompleteTimeline(root, cases[3]),
    }
    return timelines


_BASE_CREATE_CANVAS = workflow._create_canvas


def _create_canvas_5150(timelines, width, height, dpi):
    result = list(_BASE_CREATE_CANVAS(timelines, width, height, dpi))
    figure = result[0]
    figure.text(
        0.062,
        0.025,
        (
            "α = 0.2 uses the stabilized conservative potential operator after t = 2300 s.  "
            "A panel is held only after its recorded terminal time; numerical pauses are labelled."
        ),
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
    )
    figure.canvas.draw()
    result[-1] = figure.canvas.copy_from_bbox(figure.bbox)
    return tuple(result)


def _contact_sheet_5150(video: Path, output: Path, fps: int) -> Path:
    samples = (0, 2800, 3150, 4150, 4650, 5150)
    qa = output / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    for time_s in samples:
        path = qa / f"encoded_t{time_s:04d}.png"
        workflow._extract_frame(video, path, time_s / fps)
        with Image.open(path) as source:
            image = source.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 132, 27), fill=(0, 0, 0))
        draw.text((8, 7), f"t = {time_s} s", fill=(255, 255, 255))
        frames.append(image)
    sheet = Image.new("RGB", (1920, 720), "white")
    for index, image in enumerate(frames):
        sheet.paste(image, ((index % 3) * 640, (index // 3) * 360))
    destination = qa / "figure3_complete_evolution_5150_contact_sheet.png"
    sheet.save(destination, optimize=True)
    return destination


def create_movie_5150(
    *,
    root: str | Path = DEFAULT_ROOT,
    output_directory: str | Path = DEFAULT_OUTPUT,
    fps: int = 30,
    preview_stride: int = 10,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    root = Path(root)
    workflow.MOVIE_END_S = MOVIE_END_S
    workflow.CASES = build_cases(root)
    workflow.prepare_timelines = prepare_timelines_5150
    workflow._create_canvas = _create_canvas_5150
    workflow._contact_sheet = _contact_sheet_5150
    report = workflow.create_complete_movie(
        root=root,
        output_directory=output_directory,
        fps=fps,
        preview_stride=preview_stride,
        width=width,
        height=height,
    )
    output = Path(output_directory)
    old_poster = Path(report["poster"])
    new_poster = output / "figure3_epb_complete_evolution_poster_t5150.png"
    old_poster.replace(new_poster)
    report["format"] = "esf_figure3_complete_movie_5150_v1"
    report["poster"] = str(new_poster.resolve())
    report["timeline"]["second_continuation_target_s"] = 5150
    report_path = output / "figure3_epb_complete_evolution_movie_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview-stride", type=int, default=10)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args(argv)
    report = create_movie_5150(
        root=args.root,
        output_directory=args.output,
        fps=args.fps,
        preview_stride=args.preview_stride,
        width=args.width,
        height=args.height,
    )
    print(f"[complete-movie-5150] full: {report['full_movie']['path']}")
    print(f"[complete-movie-5150] preview: {report['preview_movie']['path']}")
    print(f"[complete-movie-5150] QA: {report['all_checks_pass']}")


if __name__ == "__main__":
    main()
