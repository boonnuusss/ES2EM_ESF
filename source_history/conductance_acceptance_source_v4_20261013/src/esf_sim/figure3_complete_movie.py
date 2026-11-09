"""Render the complete Figure 3 EPB evolution including validated continuations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_ROOT = Path("output/best_baseline/figure3_v2_timeseries")
DEFAULT_OUTPUT = DEFAULT_ROOT / "complete_movie"
MOVIE_END_S = 4150
OPERATOR_SWITCH_S = 2300
DISPLAY_SCALE_M3 = 1.0e11
DISPLAY_MIN = 0.0
DISPLAY_MAX = 11.0


@dataclass(frozen=True)
class CompleteMovieCase:
    key: str
    alpha: float
    base_directory: str
    base_end_s: int
    extension_directory: str
    terminal_state: str
    terminal_time_s: float


CASES = (
    CompleteMovieCase(
        "alpha_0p2",
        0.2,
        "time_series/alpha_0p2_recovered",
        3150,
        "extended_1000s/time_series/alpha_0p2",
        "completed",
        4150.0,
    ),
    CompleteMovieCase(
        "alpha_0p4",
        0.4,
        "time_series/alpha_0p4",
        3150,
        "extended_1000s/time_series/alpha_0p4",
        "completed",
        4150.0,
    ),
    CompleteMovieCase(
        "alpha_0p6",
        0.6,
        "time_series/alpha_0p6",
        3050,
        "extended_1000s/time_series/alpha_0p6",
        "paused_minimum_dt",
        3063.5625,
    ),
    CompleteMovieCase(
        "alpha_0p8",
        0.8,
        "time_series/alpha_0p8",
        2860,
        "extended_1000s/time_series/alpha_0p8",
        "paused_minimum_dt",
        2874.90625,
    ),
)


class ArchiveSegment:
    """Sequential, one-chunk-at-a-time reader for one exact-time archive segment."""

    def __init__(
        self,
        directory: str | Path,
        *,
        expected_first_s: int,
        expected_last_s: int,
        expected_alpha: float,
    ) -> None:
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.first_time_s = int(round(float(self.manifest["first_time_s"])))
        self.last_time_s = int(round(float(self.manifest["chunks"][-1]["last_time_s"])))
        self.chunks = tuple(self.manifest["chunks"])
        self._chunk_index = -1
        self._times: np.ndarray | None = None
        self._density: np.ndarray | None = None
        self.loaded_chunks: set[str] = set()
        self.minimum_density_m3 = math.inf
        self.maximum_density_m3 = -math.inf
        self._validate(expected_first_s, expected_last_s, expected_alpha)

    def _validate(self, first_s: int, last_s: int, alpha: float) -> None:
        expected = {
            "frame_interval_s": 1.0,
            "first_time_s": float(first_s),
            "density_shape": [401, 201],
            "storage_dtype": "float32",
            "pdm_alpha": alpha,
            "scheme": "pdm7",
            "potential_boundary_y": "equal_extrapolation",
        }
        mismatch = {
            key: {"expected": value, "actual": self.manifest.get(key)}
            for key, value in expected.items()
            if self.manifest.get(key) != value
        }
        if mismatch:
            raise ValueError(f"invalid manifest {self.manifest_path}: {mismatch}")
        if self.last_time_s != last_s:
            raise ValueError(
                f"archive {self.directory} ends at {self.last_time_s}, expected {last_s}"
            )
        next_time = first_s
        total = 0
        for chunk in self.chunks:
            first = int(round(float(chunk["first_time_s"])))
            last = int(round(float(chunk["last_time_s"])))
            count = int(chunk["frame_count"])
            if first != next_time or count != last - first + 1:
                raise ValueError(f"non-contiguous chunk entry: {chunk}")
            path = self.directory / chunk["file"]
            if not path.exists():
                raise FileNotFoundError(path)
            next_time = last + 1
            total += count
        if total != last_s - first_s + 1 or total != int(self.manifest["frame_count"]):
            raise ValueError(f"frame-count mismatch in {self.manifest_path}")

    def _load_chunk(self, index: int) -> None:
        entry = self.chunks[index]
        path = self.directory / entry["file"]
        with np.load(path, allow_pickle=False) as saved:
            times = np.asarray(saved["time_s"], dtype=np.float64)
            density = np.asarray(saved["density_m3"], dtype=np.float32)
        first = int(round(float(entry["first_time_s"])))
        last = int(round(float(entry["last_time_s"])))
        expected = np.arange(first, last + 1, dtype=np.float64)
        if not np.array_equal(times, expected):
            raise ValueError(f"invalid time vector in {path}")
        if density.shape != (times.size, 401, 201):
            raise ValueError(f"invalid density shape {density.shape} in {path}")
        if not np.all(np.isfinite(density)) or float(np.min(density)) <= 0.0:
            raise ValueError(f"invalid density values in {path}")
        self._chunk_index = index
        self._times = times
        self._density = density
        self.loaded_chunks.add(path.name)
        self.minimum_density_m3 = min(self.minimum_density_m3, float(np.min(density)))
        self.maximum_density_m3 = max(self.maximum_density_m3, float(np.max(density)))

    def frame(self, time_s: int) -> np.ndarray:
        if time_s < self.first_time_s or time_s > self.last_time_s:
            raise KeyError(f"t={time_s} is outside {self.first_time_s}:{self.last_time_s}")
        if self._times is None or not (self._times[0] <= time_s <= self._times[-1]):
            start = max(self._chunk_index, 0)
            for index in range(start, len(self.chunks)):
                entry = self.chunks[index]
                if int(entry["first_time_s"]) <= time_s <= int(entry["last_time_s"]):
                    self._load_chunk(index)
                    break
            else:
                raise KeyError(f"missing t={time_s} in {self.directory}")
        assert self._times is not None and self._density is not None
        return self._density[time_s - int(self._times[0])]

    def report(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory.resolve()),
            "first_time_s": self.first_time_s,
            "last_time_s": self.last_time_s,
            "frame_count": int(self.manifest["frame_count"]),
            "chunk_count": len(self.chunks),
            "loaded_chunk_count": len(self.loaded_chunks),
            "all_chunks_loaded": len(self.loaded_chunks) == len(self.chunks),
            "minimum_density_m3": self.minimum_density_m3,
            "maximum_density_m3": self.maximum_density_m3,
            "potential_operator": self.manifest["potential_operator"],
        }


class CompleteTimeline:
    """Join a canonical archive, a continuation, and an explicit terminal hold."""

    def __init__(self, root: Path, case: CompleteMovieCase) -> None:
        self.root = root
        self.case = case
        extension_manifest = json.loads(
            (root / case.extension_directory / "manifest.json").read_text(encoding="utf-8")
        )
        extension_last = int(
            round(float(extension_manifest["chunks"][-1]["last_time_s"]))
        )
        self.base = ArchiveSegment(
            root / case.base_directory,
            expected_first_s=0,
            expected_last_s=case.base_end_s,
            expected_alpha=case.alpha,
        )
        self.extension = ArchiveSegment(
            root / case.extension_directory,
            expected_first_s=case.base_end_s,
            expected_last_s=extension_last,
            expected_alpha=case.alpha,
        )
        status_path = root / "extended_1000s" / "runs" / case.key / "status.json"
        self.status = json.loads(status_path.read_text(encoding="utf-8"))
        if self.status["state"] != case.terminal_state or not math.isclose(
            float(self.status["time_s"]), case.terminal_time_s, abs_tol=1.0e-9
        ):
            raise ValueError(f"terminal status mismatch for {case.key}")
        snapshot_path = root / "extended_1000s" / "runs" / case.key / "simulations" / "density_pdm7.npz"
        with np.load(snapshot_path, allow_pickle=False) as saved:
            self.terminal_density = np.asarray(saved["density"], dtype=np.float32)
            snapshot_time = float(saved["time_s"])
        if not math.isclose(snapshot_time, case.terminal_time_s, abs_tol=1.0e-9):
            raise ValueError(f"terminal snapshot mismatch for {case.key}")
        base_seam = self.base.frame(case.base_end_s)
        extension_seam = self.extension.frame(case.base_end_s)
        self.seam_exact = bool(np.array_equal(base_seam, extension_seam))
        if not self.seam_exact:
            raise ValueError(f"non-identical archive seam for {case.key}")

    @property
    def last_integer_time_s(self) -> int:
        return self.extension.last_time_s

    def frame(self, global_time_s: int) -> np.ndarray:
        if global_time_s <= self.case.base_end_s:
            return self.base.frame(global_time_s)
        if global_time_s <= self.extension.last_time_s:
            return self.extension.frame(global_time_s)
        return self.terminal_density

    def is_held(self, global_time_s: int) -> bool:
        return global_time_s > self.extension.last_time_s

    def hold_label(self) -> str:
        if self.case.terminal_state == "completed":
            return f"final state held at t = {self.case.terminal_time_s:.0f} s"
        return f"simulation paused at t = {self.case.terminal_time_s:.4g} s\nminimum-step gate reached"

    def report(self) -> dict[str, Any]:
        return {
            "alpha": self.case.alpha,
            "base": self.base.report(),
            "extension": self.extension.report(),
            "seam_time_s": self.case.base_end_s,
            "seam_float32_exact": self.seam_exact,
            "terminal_state": self.case.terminal_state,
            "terminal_time_s": self.case.terminal_time_s,
            "last_integer_archive_time_s": self.last_integer_time_s,
            "hold_begins_at_global_time_s": self.last_integer_time_s + 1,
            "hold_source": "terminal float64 snapshot converted to float32",
            "pause_reason": self.status.get("reason"),
        }


def prepare_timelines(root: str | Path = DEFAULT_ROOT) -> dict[str, CompleteTimeline]:
    root = Path(root)
    return {case.key: CompleteTimeline(root, case) for case in CASES}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10.0,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _open_writer(path: Path, width: int, height: int, fps: int) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
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
            "title=Complete ESF Figure 3 evolution",
        ],
    )
    writer.send(None)
    return writer


def _create_canvas(
    timelines: dict[str, CompleteTimeline], width: int, height: int, dpi: int
) -> tuple[Any, list[Any], list[Any], Any, Any, Any, np.ndarray]:
    _style()
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    grid = figure.add_gridspec(
        1, 4, left=0.062, right=0.892, bottom=0.185, top=0.86, wspace=0.085
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    norm = Normalize(vmin=DISPLAY_MIN, vmax=DISPLAY_MAX)
    cmap = mpl.colormaps["plasma"]
    images: list[Any] = []
    status_texts: list[Any] = []
    operator_text = None
    for index, (axis, case) in enumerate(zip(axes, CASES)):
        image = axis.imshow(
            timelines[case.key].frame(0) / DISPLAY_SCALE_M3,
            origin="lower",
            extent=(-125.0, 125.0, 250.0, 650.0),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
            animated=True,
        )
        images.append(image)
        axis.set_xlim(-125.0, 125.0)
        axis.set_ylim(250.0, 650.0)
        axis.set_box_aspect(1.6)
        axis.set_xticks([-100.0, -50.0, 0.0, 50.0, 100.0])
        axis.set_yticks([250.0, 350.0, 450.0, 550.0, 650.0])
        axis.tick_params(top=True, right=True, length=4.0, labelleft=index == 0)
        axis.set_xlabel("East–West (km)")
        if index == 0:
            axis.set_ylabel("Altitude (km)")
        axis.set_title(f"{chr(97 + index)})  α = {case.alpha:.1f}", pad=7.0, fontweight="bold")
        status = axis.text(
            0.5,
            0.975,
            "",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="white",
            fontsize=8.0,
            linespacing=1.25,
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.68, "pad": 3.0},
            visible=False,
            animated=True,
        )
        status_texts.append(status)
        if case.key == "alpha_0p2":
            operator_text = axis.text(
                0.035,
                0.03,
                "point-centred v2",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                color="white",
                fontsize=7.5,
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 2.0},
                animated=True,
            )

    cax = figure.add_axes((0.912, 0.25, 0.016, 0.48))
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=np.arange(0.0, 12.0, 1.0)
    )
    colorbar.ax.tick_params(direction="out", labelsize=8.5, length=3.0)
    colorbar.ax.set_title(r"$10^5\,\mathrm{cm}^{-3}$", fontsize=9.0, pad=7.0)

    time_text = figure.text(
        0.48,
        0.94,
        "Complete ESF evolution — t = 0 s (0.0 min)",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        animated=True,
    )
    figure.text(
        0.48,
        0.897,
        "PDM7 · 201 × 401 cells · one archived field per simulated second",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.text(
        0.062,
        0.025,
        (
            "α = 0.2 uses the stabilized conservative potential operator after t = 2300 s.  "
            "α = 0.6 and 0.8 are held only after the minimum-step safety pause."
        ),
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#333333",
    )

    progress_axis = figure.add_axes((0.13, 0.096, 0.70, 0.035))
    progress_axis.set_xlim(0.0, float(MOVIE_END_S))
    progress_axis.set_ylim(0.0, 1.0)
    progress_axis.set_yticks([])
    progress_axis.set_xticks([0, 600, 1200, 1800, 2400, 3000, 3600, 4150])
    progress_axis.tick_params(axis="x", direction="out", length=3.0, labelsize=8.0)
    for side in ("left", "right", "top"):
        progress_axis.spines[side].set_visible(False)
    progress_axis.spines["bottom"].set_position(("axes", 0.25))
    progress_axis.plot([0.0, MOVIE_END_S], [0.25, 0.25], color="#cccccc", lw=5.0)
    progress_line, = progress_axis.plot(
        [0.0, 0.0], [0.25, 0.25], color="#4c78a8", lw=5.0, solid_capstyle="round", animated=True
    )
    progress_axis.set_xlabel("Simulation time (s)", labelpad=1.0, fontsize=8.5)
    figure.canvas.draw()
    background = figure.canvas.copy_from_bbox(figure.bbox)
    assert operator_text is not None
    return figure, images, status_texts, operator_text, time_text, progress_line, background


def _draw_frame(
    figure: Any,
    timelines: dict[str, CompleteTimeline],
    images: list[Any],
    status_texts: list[Any],
    operator_text: Any,
    time_text: Any,
    progress_line: Any,
    background: Any,
    time_s: int,
) -> np.ndarray:
    for image, status, case in zip(images, status_texts, CASES):
        timeline = timelines[case.key]
        image.set_data(timeline.frame(time_s) / DISPLAY_SCALE_M3)
        held = timeline.is_held(time_s)
        status.set_visible(held)
        if held:
            status.set_text(timeline.hold_label())
    operator_text.set_text(
        "point-centred v2" if time_s < OPERATOR_SWITCH_S else "stabilized potential operator"
    )
    time_text.set_text(
        f"Complete ESF evolution — t = {time_s:,d} s ({time_s / 60.0:.1f} min)"
    )
    progress_line.set_data([0.0, float(time_s)], [0.25, 0.25])
    canvas = figure.canvas
    canvas.restore_region(background)
    for image, status in zip(images, status_texts):
        image.axes.draw_artist(image)
        if status.get_visible():
            status.axes.draw_artist(status)
    operator_text.axes.draw_artist(operator_text)
    figure.draw_artist(time_text)
    progress_line.axes.draw_artist(progress_line)
    canvas.blit(figure.bbox)
    return np.ascontiguousarray(np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_metadata(path: Path) -> dict[str, Any]:
    frame_count, duration_s = imageio_ffmpeg.count_frames_and_secs(str(path))
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    reader.close()
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "frame_count": int(frame_count),
        "duration_s": float(duration_s),
        "fps": float(metadata["fps"]),
        "size_pixels": list(metadata["size"]),
        "codec": metadata.get("codec"),
        "pix_fmt": metadata.get("pix_fmt"),
    }


def _extract_frame(video: Path, destination: Path, frame_time_s: float) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{frame_time_s:.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
    )


def _contact_sheet(video: Path, output: Path, fps: int) -> Path:
    samples = (0, 2000, 2800, 3150, 3500, 4150)
    qa = output / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    for time_s in samples:
        path = qa / f"encoded_t{time_s:04d}.png"
        _extract_frame(video, path, time_s / fps)
        with Image.open(path) as source:
            image = source.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 132, 27), fill=(0, 0, 0))
        draw.text((8, 7), f"t = {time_s} s", fill=(255, 255, 255))
        frames.append(image)
    sheet = Image.new("RGB", (1920, 720), "white")
    for index, image in enumerate(frames):
        sheet.paste(image, ((index % 3) * 640, (index // 3) * 360))
    destination = qa / "figure3_complete_evolution_contact_sheet.png"
    sheet.save(destination, optimize=True)
    return destination


def create_complete_movie(
    *,
    root: str | Path = DEFAULT_ROOT,
    output_directory: str | Path = DEFAULT_OUTPUT,
    fps: int = 30,
    preview_stride: int = 10,
    width: int = 1920,
    height: int = 1080,
    dpi: int = 120,
) -> dict[str, Any]:
    if fps <= 0 or preview_stride <= 0 or width % 2 or height % 2:
        raise ValueError("fps/stride must be positive and video dimensions must be even")
    root = Path(root)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timelines = prepare_timelines(root)
    figure, images, statuses, operator, time_text, progress, background = _create_canvas(
        timelines, width, height, dpi
    )
    full_path = output / "figure3_epb_complete_evolution_1s_30fps.mp4"
    preview_path = output / "figure3_epb_complete_evolution_preview_10s_30fps.mp4"
    full_partial = full_path.with_name(full_path.stem + ".partial.mp4")
    preview_partial = preview_path.with_name(preview_path.stem + ".partial.mp4")
    full_partial.unlink(missing_ok=True)
    preview_partial.unlink(missing_ok=True)
    full_writer = _open_writer(full_partial, width, height, fps)
    preview_writer = _open_writer(preview_partial, width, height, fps)
    last_frame: np.ndarray | None = None
    try:
        for time_s in range(MOVIE_END_S + 1):
            frame = _draw_frame(
                figure,
                timelines,
                images,
                statuses,
                operator,
                time_text,
                progress,
                background,
                time_s,
            )
            full_writer.send(frame)
            if time_s % preview_stride == 0 or time_s == MOVIE_END_S:
                preview_writer.send(frame)
            last_frame = frame
            if time_s and time_s % 500 == 0:
                print(f"[complete-movie] encoded t={time_s} s", flush=True)
    finally:
        full_writer.close()
        preview_writer.close()
        plt.close(figure)
    if last_frame is None:
        raise RuntimeError("no frames rendered")
    full_partial.replace(full_path)
    preview_partial.replace(preview_path)
    poster = output / "figure3_epb_complete_evolution_poster_t4150.png"
    Image.fromarray(last_frame).save(poster, optimize=True)

    full_meta = _video_metadata(full_path)
    preview_meta = _video_metadata(preview_path)
    contact_sheet = _contact_sheet(full_path, output, fps)
    encoded_final = output / "qa" / "encoded_final.png"
    _extract_frame(full_path, encoded_final, MOVIE_END_S / fps)
    with Image.open(encoded_final) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.float64)
    rendered = last_frame.astype(np.float64)
    mae = float(np.mean(np.abs(decoded - rendered)) / 255.0)
    correlation = float(np.corrcoef(decoded.ravel(), rendered.ravel())[0, 1])
    inputs = {case.key: timelines[case.key].report() for case in CASES}
    expected_preview = MOVIE_END_S // preview_stride + 1
    checks = {
        "all_archive_seams_exact": all(x["seam_float32_exact"] for x in inputs.values()),
        "all_input_chunks_read": all(
            x[segment]["all_chunks_loaded"]
            for x in inputs.values()
            for segment in ("base", "extension")
        ),
        "full_frame_count": full_meta["frame_count"] == MOVIE_END_S + 1,
        "full_duration": abs(full_meta["duration_s"] - (MOVIE_END_S + 1) / fps) < 0.05,
        "full_dimensions": full_meta["size_pixels"] == [width, height],
        "full_fps": abs(full_meta["fps"] - fps) < 1.0e-6,
        "preview_frame_count": preview_meta["frame_count"] == expected_preview,
        "encoded_final_matches_render": mae < 0.04 and correlation > 0.98,
        "poster_dimensions": Image.open(poster).size == (width, height),
    }
    report: dict[str, Any] = {
        "format": "esf_figure3_complete_movie_v1",
        "timeline": {
            "first_time_s": 0,
            "last_time_s": MOVIE_END_S,
            "input_interval_s": 1,
            "output_fps": fps,
            "full_video_duration_s": (MOVIE_END_S + 1) / fps,
            "terminal_hold_policy": {
                case.key: timelines[case.key].hold_label()
                if timelines[case.key].last_integer_time_s < MOVIE_END_S
                else None
                for case in CASES
            },
        },
        "display": {
            "layout": "one row by four columns",
            "density_range_1e5_cm3": [DISPLAY_MIN, DISPLAY_MAX],
            "colormap": "plasma",
            "video_size_pixels": [width, height],
        },
        "inputs": inputs,
        "full_movie": full_meta,
        "preview_movie": preview_meta,
        "poster": str(poster.resolve()),
        "contact_sheet": str(contact_sheet.resolve()),
        "encoded_final_frame_qa": {
            "normalized_mean_absolute_error": mae,
            "pixel_correlation": correlation,
            "decoded_frame": str(encoded_final.resolve()),
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    report_path = output / "figure3_epb_complete_evolution_movie_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not report["all_checks_pass"]:
        raise RuntimeError(f"movie QA failed: {[key for key, ok in checks.items() if not ok]}")
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
    report = create_complete_movie(
        root=args.root,
        output_directory=args.output,
        fps=args.fps,
        preview_stride=args.preview_stride,
        width=args.width,
        height=args.height,
    )
    print(f"[complete-movie] full: {report['full_movie']['path']}")
    print(f"[complete-movie] preview: {report['preview_movie']['path']}")
    print(f"[complete-movie] QA: {report['all_checks_pass']}")


if __name__ == "__main__":
    main()
