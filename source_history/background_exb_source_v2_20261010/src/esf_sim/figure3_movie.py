"""Create a four-panel movie from the archived one-second Figure 3 fields."""

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
DEFAULT_OUTPUT = DEFAULT_ROOT / "movie"
DISPLAY_SCALE_M3 = 1.0e11
DISPLAY_MIN = 0.0
DISPLAY_MAX = 11.0
MOVIE_END_S = 3150
OPERATOR_SWITCH_S = 2300


@dataclass(frozen=True)
class MovieCase:
    key: str
    alpha: float
    target_time_s: int
    relative_directory: str
    recovered: bool = False


CASES = (
    MovieCase("alpha_0p2", 0.2, 3150, "time_series/alpha_0p2_recovered", True),
    MovieCase("alpha_0p4", 0.4, 3150, "time_series/alpha_0p4"),
    MovieCase("alpha_0p6", 0.6, 3050, "time_series/alpha_0p6"),
    MovieCase("alpha_0p8", 0.8, 2860, "time_series/alpha_0p8"),
)


class SequentialArchive:
    """Read a chunked archive sequentially while keeping one chunk in memory."""

    def __init__(self, directory: str | Path, case: MovieCase) -> None:
        self.directory = Path(directory)
        self.case = case
        self.manifest_path = self.directory / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"missing movie manifest: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest()
        self.chunks = tuple(self.manifest["chunks"])
        self._chunk_index = -1
        self._chunk_times: np.ndarray | None = None
        self._chunk_density: np.ndarray | None = None
        self.loaded_chunk_names: set[str] = set()
        self.minimum_density_m3 = math.inf
        self.maximum_density_m3 = -math.inf

    def _validate_manifest(self) -> None:
        expected = {
            "completed": True,
            "frame_interval_s": 1.0,
            "frame_count": self.case.target_time_s + 1,
            "expected_frame_count": self.case.target_time_s + 1,
            "first_time_s": 0.0,
            "target_time_s": float(self.case.target_time_s),
            "density_shape": [401, 201],
            "storage_dtype": "float32",
            "pdm_alpha": self.case.alpha,
            "scheme": "pdm7",
            "potential_boundary_y": "equal_extrapolation",
        }
        mismatches = {
            key: {"expected": value, "actual": self.manifest.get(key)}
            for key, value in expected.items()
            if self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"invalid archive manifest {self.manifest_path}: {mismatches}")
        next_time = 0
        frame_count = 0
        for chunk in self.manifest.get("chunks", []):
            first = int(round(float(chunk["first_time_s"])))
            last = int(round(float(chunk["last_time_s"])))
            count = int(chunk["frame_count"])
            if first != next_time or last - first + 1 != count:
                raise ValueError(f"non-contiguous chunk entry in {self.manifest_path}: {chunk}")
            if not (self.directory / chunk["file"]).exists():
                raise FileNotFoundError(self.directory / chunk["file"])
            next_time = last + 1
            frame_count += count
        if frame_count != self.case.target_time_s + 1:
            raise ValueError(f"chunk count mismatch in {self.manifest_path}")

    def _load_chunk(self, index: int) -> None:
        entry = self.chunks[index]
        path = self.directory / entry["file"]
        with np.load(path, allow_pickle=False) as saved:
            times = np.asarray(saved["time_s"], dtype=np.float64)
            density = np.asarray(saved["density_m3"], dtype=np.float32)
        first = int(round(float(entry["first_time_s"])))
        last = int(round(float(entry["last_time_s"])))
        expected_times = np.arange(first, last + 1, dtype=np.float64)
        if times.shape != expected_times.shape or not np.array_equal(times, expected_times):
            raise ValueError(f"invalid time vector in {path}")
        if density.shape != (times.size, 401, 201):
            raise ValueError(f"invalid density shape {density.shape} in {path}")
        if not np.all(np.isfinite(density)) or float(np.min(density)) <= 0.0:
            raise ValueError(f"non-finite or non-positive density in {path}")
        self._chunk_index = index
        self._chunk_times = times
        self._chunk_density = density
        self.loaded_chunk_names.add(path.name)
        self.minimum_density_m3 = min(self.minimum_density_m3, float(np.min(density)))
        self.maximum_density_m3 = max(self.maximum_density_m3, float(np.max(density)))

    def frame(self, time_s: int) -> np.ndarray:
        sample_time = min(int(time_s), self.case.target_time_s)
        if self._chunk_times is None or not (
            self._chunk_times[0] <= sample_time <= self._chunk_times[-1]
        ):
            start = max(self._chunk_index, 0)
            for index in range(start, len(self.chunks)):
                entry = self.chunks[index]
                if int(entry["first_time_s"]) <= sample_time <= int(
                    entry["last_time_s"]
                ):
                    self._load_chunk(index)
                    break
            else:
                raise KeyError(f"no frame t={sample_time} s in {self.directory}")
        assert self._chunk_times is not None and self._chunk_density is not None
        local_index = int(sample_time - int(self._chunk_times[0]))
        return self._chunk_density[local_index]

    def report(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory.resolve()),
            "manifest": str(self.manifest_path.resolve()),
            "target_time_s": self.case.target_time_s,
            "frame_count": self.manifest["frame_count"],
            "chunk_count": len(self.chunks),
            "loaded_chunk_count": len(self.loaded_chunk_names),
            "all_chunks_loaded": len(self.loaded_chunk_names) == len(self.chunks),
            "minimum_density_m3": self.minimum_density_m3,
            "maximum_density_m3": self.maximum_density_m3,
            "storage_dtype": self.manifest["storage_dtype"],
            "potential_operator_manifest": self.manifest["potential_operator"],
            "recovered_trajectory": self.case.recovered,
        }


def prepare_archives(root: str | Path = DEFAULT_ROOT) -> dict[str, SequentialArchive]:
    root = Path(root)
    return {
        case.key: SequentialArchive(root / case.relative_directory, case) for case in CASES
    }


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
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


def _open_writer(
    path: Path,
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


def _create_canvas(
    archives: dict[str, SequentialArchive],
    *,
    width_px: int,
    height_px: int,
    dpi: int,
) -> tuple[Any, list[Any], list[Any], Any, Any, np.ndarray]:
    _style()
    figure = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    grid = figure.add_gridspec(
        1,
        4,
        left=0.065,
        right=0.895,
        bottom=0.175,
        top=0.875,
        wspace=0.085,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    norm = Normalize(vmin=DISPLAY_MIN, vmax=DISPLAY_MAX)
    cmap = mpl.colormaps["plasma"]
    images: list[Any] = []
    status_texts: list[Any] = []
    operator_text = None
    for index, (axis, case) in enumerate(zip(axes, CASES)):
        density = archives[case.key].frame(0) / DISPLAY_SCALE_M3
        image = axis.imshow(
            density,
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
        axis.set_xlabel("East-West (km)")
        if index == 0:
            axis.set_ylabel("Altitude (km)")
        marker = "*" if case.recovered else ""
        axis.set_title(f"{chr(97 + index)})  α = {case.alpha:.1f}{marker}", pad=7.0)
        status = axis.text(
            0.5,
            0.975,
            "",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="white",
            fontsize=8.5,
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.62, "pad": 2.5},
            visible=False,
            animated=True,
        )
        status_texts.append(status)
        if case.recovered:
            operator_text = axis.text(
                0.035,
                0.03,
                "point-centred v2",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                color="white",
                fontsize=7.7,
                bbox={
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.55,
                    "pad": 2.0,
                },
                animated=True,
            )

    colorbar_axis = figure.add_axes((0.915, 0.255, 0.016, 0.48))
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axis,
        ticks=np.arange(0.0, 12.0, 1.0),
    )
    colorbar.ax.tick_params(direction="out", labelsize=8.5, length=3.0)
    colorbar.ax.set_title("10⁵ cm⁻³", fontsize=9.0, pad=7.0)

    time_text = figure.text(
        0.48,
        0.945,
        "ESF evolution — t = 0 s (0.0 min)",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        animated=True,
    )
    figure.text(
        0.48,
        0.905,
        "PDM7 · 201 × 401 cells · one archived field per simulated second",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.text(
        0.065,
        0.027,
        (
            "* α = 0.2 uses the formal v2 trajectory through 2300 s and the "
            "stabilized conservative potential operator thereafter."
        ),
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#333333",
    )

    progress_axis = figure.add_axes((0.13, 0.095, 0.70, 0.035))
    progress_axis.set_xlim(0.0, float(MOVIE_END_S))
    progress_axis.set_ylim(0.0, 1.0)
    progress_axis.set_yticks([])
    progress_axis.set_xticks([0, 600, 1200, 1800, 2400, 3000, 3150])
    progress_axis.tick_params(axis="x", direction="out", length=3.0, labelsize=8.0)
    for side in ("left", "right", "top"):
        progress_axis.spines[side].set_visible(False)
    progress_axis.spines["bottom"].set_position(("axes", 0.25))
    progress_axis.plot([0.0, MOVIE_END_S], [0.25, 0.25], color="#cccccc", lw=5.0)
    progress_line, = progress_axis.plot(
        [0.0, 0.0],
        [0.25, 0.25],
        color="#4c78a8",
        lw=5.0,
        solid_capstyle="round",
        animated=True,
    )
    progress_axis.set_xlabel("Simulation time (s)", labelpad=1.0, fontsize=8.5)

    figure.canvas.draw()
    background = figure.canvas.copy_from_bbox(figure.bbox)
    dynamic = np.array(images + status_texts, dtype=object)
    assert operator_text is not None
    return (
        figure,
        images,
        status_texts,
        operator_text,
        (time_text, progress_line),
        background,
    )


def _draw_frame(
    figure: Any,
    archives: dict[str, SequentialArchive],
    images: list[Any],
    status_texts: list[Any],
    operator_text: Any,
    time_text: Any,
    progress_line: Any,
    background: Any,
    time_s: int,
) -> np.ndarray:
    for image, status, case in zip(images, status_texts, CASES):
        image.set_data(archives[case.key].frame(time_s) / DISPLAY_SCALE_M3)
        held = time_s > case.target_time_s
        status.set_visible(held)
        if held:
            status.set_text(f"final state held at t = {case.target_time_s} s")
    operator_text.set_text(
        "point-centred v2"
        if time_s < OPERATOR_SWITCH_S
        else "stabilized potential operator"
    )
    time_text.set_text(
        f"ESF evolution — t = {time_s:,d} s ({time_s / 60.0:.1f} min)"
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
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return np.ascontiguousarray(rgba[:, :, :3])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_frame(video: Path, destination: Path, video_time_s: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{video_time_s:.6f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        str(destination),
    ]
    subprocess.run(command, check=True)


def _create_contact_sheet(
    video: Path,
    output_directory: Path,
    *,
    fps: int,
    width_px: int,
    height_px: int,
) -> Path:
    times = (0, 1200, 2000, 2400, 2800, 3150)
    images: list[Image.Image] = []
    qa_directory = output_directory / "qa"
    for time_s in times:
        path = qa_directory / f"encoded_t{time_s:04d}.png"
        _extract_frame(video, path, time_s / fps)
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(resized)
        draw.rectangle((0, 0, 118, 25), fill=(0, 0, 0))
        draw.text((7, 6), f"t = {time_s} s", fill=(255, 255, 255))
        images.append(resized)
    sheet = Image.new("RGB", (1920, 720), color="white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 3) * 640, (index // 3) * 360))
    destination = qa_directory / "figure3_epb_evolution_contact_sheet.png"
    sheet.save(destination, optimize=True)
    return destination


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


def create_movie(
    *,
    root: str | Path = DEFAULT_ROOT,
    output_directory: str | Path = DEFAULT_OUTPUT,
    fps: int = 30,
    preview_stride: int = 10,
    width_px: int = 1920,
    height_px: int = 1080,
    dpi: int = 120,
) -> dict[str, Any]:
    if fps <= 0 or preview_stride <= 0:
        raise ValueError("fps and preview_stride must be positive")
    if width_px % 2 or height_px % 2:
        raise ValueError("H.264 yuv420p output dimensions must be even")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    archives = prepare_archives(root)
    (
        figure,
        images,
        status_texts,
        operator_text,
        time_artists,
        background,
    ) = _create_canvas(
        archives, width_px=width_px, height_px=height_px, dpi=dpi
    )
    time_text, progress_line = time_artists

    full_path = output / "figure3_epb_evolution_1s_30fps.mp4"
    preview_path = output / "figure3_epb_evolution_preview_10s_30fps.mp4"
    full_partial = full_path.with_name(full_path.stem + ".partial.mp4")
    preview_partial = preview_path.with_name(preview_path.stem + ".partial.mp4")
    full_partial.unlink(missing_ok=True)
    preview_partial.unlink(missing_ok=True)
    full_writer = _open_writer(
        full_partial, width_px=width_px, height_px=height_px, fps=fps
    )
    preview_writer = _open_writer(
        preview_partial, width_px=width_px, height_px=height_px, fps=fps
    )
    last_frame: np.ndarray | None = None
    try:
        for time_s in range(MOVIE_END_S + 1):
            frame = _draw_frame(
                figure,
                archives,
                images,
                status_texts,
                operator_text,
                time_text,
                progress_line,
                background,
                time_s,
            )
            full_writer.send(frame)
            if time_s % preview_stride == 0 or time_s == MOVIE_END_S:
                preview_writer.send(frame)
            last_frame = frame
            if time_s and time_s % 500 == 0:
                print(f"[figure3-movie] encoded t={time_s} s", flush=True)
    finally:
        full_writer.close()
        preview_writer.close()
        plt.close(figure)
    if last_frame is None:
        raise RuntimeError("no movie frames were rendered")
    full_partial.replace(full_path)
    preview_partial.replace(preview_path)
    poster_path = output / "figure3_epb_evolution_poster_t3150.png"
    Image.fromarray(last_frame).save(poster_path, optimize=True)

    full_metadata = _video_metadata(full_path)
    preview_metadata = _video_metadata(preview_path)
    expected_preview_frames = MOVIE_END_S // preview_stride + 1
    if MOVIE_END_S % preview_stride:
        expected_preview_frames += 1
    contact_sheet = _create_contact_sheet(
        full_path,
        output,
        fps=fps,
        width_px=width_px,
        height_px=height_px,
    )
    encoded_final = output / "qa" / "encoded_final.png"
    _extract_frame(full_path, encoded_final, MOVIE_END_S / fps)
    with Image.open(encoded_final) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.float64)
    rendered = last_frame.astype(np.float64)
    final_mae = float(np.mean(np.abs(decoded - rendered)) / 255.0)
    final_correlation = float(np.corrcoef(decoded.ravel(), rendered.ravel())[0, 1])

    input_reports = {case.key: archives[case.key].report() for case in CASES}
    checks = {
        "all_input_archives_fully_read": bool(
            all(report["all_chunks_loaded"] for report in input_reports.values())
        ),
        "full_movie_has_one_frame_per_simulated_second": (
            full_metadata["frame_count"] == MOVIE_END_S + 1
        ),
        "full_movie_duration_matches": abs(
            full_metadata["duration_s"] - (MOVIE_END_S + 1) / fps
        )
        < 0.05,
        "full_movie_is_1920x1080": full_metadata["size_pixels"]
        == [width_px, height_px],
        "full_movie_is_30fps": abs(full_metadata["fps"] - fps) < 1.0e-6,
        "preview_frame_count_matches_stride": (
            preview_metadata["frame_count"] == expected_preview_frames
        ),
        "encoded_final_frame_matches_render": (
            final_mae < 0.04 and final_correlation > 0.98
        ),
        "poster_is_1920x1080": Image.open(poster_path).size
        == (width_px, height_px),
    }
    report: dict[str, Any] = {
        "format": "esf_figure3_movie_v1",
        "timeline": {
            "first_time_s": 0,
            "last_time_s": MOVIE_END_S,
            "input_interval_s": 1,
            "output_fps": fps,
            "seconds_of_simulation_per_video_frame": 1,
            "hold_policy": {
                case.key: (
                    None
                    if case.target_time_s == MOVIE_END_S
                    else f"hold t={case.target_time_s} s through t={MOVIE_END_S} s"
                )
                for case in CASES
            },
        },
        "display": {
            "layout": "one row by four columns",
            "x_range_km": [-125.0, 125.0],
            "y_range_km": [250.0, 650.0],
            "density_range_1e5_cm3": [DISPLAY_MIN, DISPLAY_MAX],
            "colormap": "plasma",
            "panel_height_width_ratio": 1.6,
        },
        "alpha_0p2_provenance": {
            "formal_v2_time_range_s": [0, OPERATOR_SWITCH_S],
            "stabilized_operator_time_range_s": [OPERATOR_SWITCH_S, MOVIE_END_S],
            "stabilized_operator": "exponential_conservative",
        },
        "inputs": input_reports,
        "full_movie": full_metadata,
        "preview_movie": preview_metadata,
        "preview_stride_s": preview_stride,
        "poster": str(poster_path.resolve()),
        "contact_sheet": str(contact_sheet.resolve()),
        "encoded_final_frame_qa": {
            "normalized_mean_absolute_error": final_mae,
            "pixel_correlation": final_correlation,
            "decoded_frame": str(encoded_final.resolve()),
        },
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }
    report_path = output / "figure3_epb_evolution_movie_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not report["all_checks_pass"]:
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"Figure 3 movie QA failed: {failed}")
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
    report = create_movie(
        root=args.root,
        output_directory=args.output,
        fps=args.fps,
        preview_stride=args.preview_stride,
        width_px=args.width,
        height_px=args.height,
    )
    print(f"[figure3-movie] full: {report['full_movie']['path']}", flush=True)
    print(f"[figure3-movie] preview: {report['preview_movie']['path']}", flush=True)
    print(f"[figure3-movie] duration: {report['full_movie']['duration_s']:.3f} s", flush=True)


if __name__ == "__main__":
    main()

