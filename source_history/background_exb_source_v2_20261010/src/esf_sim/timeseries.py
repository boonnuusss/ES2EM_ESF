"""Restart-safe compressed archives for regularly sampled ESF density fields."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .config import SimulationConfig


_CHUNK_PATTERN = re.compile(r"frames_(\d{6})_(\d{6})\.npz$")


class FrameArchiveError(RuntimeError):
    """Raised when a frame archive is inconsistent with its solver checkpoint."""


class ChunkedFrameArchive:
    """Store exact-time fields in restart-safe, atomically written NPZ chunks.

    Solver state remains float64 in checkpoints. The movie-oriented history uses
    float32, retaining about seven significant digits while halving storage.
    """

    def __init__(
        self,
        directory: str | Path,
        config: SimulationConfig,
        *,
        interval_s: float = 1.0,
        chunk_frames: int = 100,
        storage_dtype: str | np.dtype = np.float32,
    ) -> None:
        if interval_s <= 0.0:
            raise ValueError("frame interval must be positive")
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        count_float = config.t_end_s / interval_s
        if not math.isclose(count_float, round(count_float), abs_tol=1.0e-10):
            raise ValueError("t_end_s must be an integer multiple of frame interval")
        self.directory = Path(directory)
        self.config = config
        self.interval_s = float(interval_s)
        self.chunk_frames = int(chunk_frames)
        self.storage_dtype = np.dtype(storage_dtype)
        self.expected_times = np.arange(int(round(count_float)) + 1, dtype=np.float64)
        self.expected_times *= self.interval_s
        self._stored_count = 0
        self._buffer_times: list[float] = []
        self._buffer_density: list[np.ndarray] = []

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def next_time_s(self) -> float:
        index = self._stored_count + len(self._buffer_times)
        if index >= self.expected_times.size:
            return math.inf
        return float(self.expected_times[index])

    @property
    def frame_count(self) -> int:
        return self._stored_count + len(self._buffer_times)

    def _chunk_paths(self) -> list[Path]:
        return sorted(self.directory.glob("frames_??????_??????.npz"))

    @staticmethod
    def _chunk_times(path: Path) -> np.ndarray:
        match = _CHUNK_PATTERN.match(path.name)
        if match is None:
            raise FrameArchiveError(f"invalid frame chunk name: {path.name}")
        with np.load(path, allow_pickle=False) as saved:
            times = np.asarray(saved["time_s"], dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise FrameArchiveError(f"invalid time index in {path}")
        if int(round(times[0])) != int(match.group(1)) or int(round(times[-1])) != int(
            match.group(2)
        ):
            raise FrameArchiveError(f"file name and time index disagree in {path}")
        return times

    def _scan_prefix(self) -> np.ndarray:
        sequences = [self._chunk_times(path) for path in self._chunk_paths()]
        if not sequences:
            return np.empty(0, dtype=np.float64)
        times = np.concatenate(sequences)
        expected = self.expected_times[: times.size]
        if times.size > self.expected_times.size or not np.allclose(
            times, expected, rtol=0.0, atol=1.0e-9
        ):
            raise FrameArchiveError(
                f"frame chunks in {self.directory} are not a contiguous prefix"
            )
        return times

    def reset(self) -> None:
        """Remove only files owned by this archive."""

        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self._chunk_paths():
            path.unlink()
        self.manifest_path.unlink(missing_ok=True)
        self._stored_count = 0
        self._buffer_times.clear()
        self._buffer_density.clear()

    def prepare(
        self,
        *,
        resume_time_s: float,
        density: np.ndarray,
        resume: bool,
    ) -> None:
        """Align the archive to the solver checkpoint and capture that state if due."""

        self.directory.mkdir(parents=True, exist_ok=True)
        if not resume:
            self.reset()
        tolerance = 1.0e-9
        for path in self._chunk_paths():
            times = self._chunk_times(path)
            if float(times[-1]) > resume_time_s + tolerance:
                path.unlink()
        stored_times = self._scan_prefix()
        self._stored_count = int(stored_times.size)
        self._buffer_times.clear()
        self._buffer_density.clear()
        if self._stored_count and float(stored_times[-1]) > resume_time_s + tolerance:
            raise FrameArchiveError("frame archive extends beyond the solver checkpoint")

        next_time = self.next_time_s
        if next_time < resume_time_s - tolerance:
            raise FrameArchiveError(
                "solver checkpoint is ahead of a missing frame; restart this case from t=0"
            )
        if math.isclose(next_time, resume_time_s, rel_tol=0.0, abs_tol=tolerance):
            self.capture(resume_time_s, density)
        self._write_manifest(completed=False)

    def capture(self, time_s: float, density: np.ndarray) -> None:
        """Capture one field at the next prescribed output time."""

        target = self.next_time_s
        if not np.isfinite(target):
            raise FrameArchiveError("attempted to write beyond the target time")
        if not math.isclose(time_s, target, rel_tol=0.0, abs_tol=1.0e-8):
            raise FrameArchiveError(
                f"frame time {time_s:.12g} does not equal expected time {target:.12g}"
            )
        array = np.asarray(density)
        if array.shape != (self.config.ny, self.config.nx):
            raise FrameArchiveError(
                f"density shape {array.shape} != {(self.config.ny, self.config.nx)}"
            )
        if not np.all(np.isfinite(array)):
            raise FrameArchiveError(f"non-finite density at t={time_s:g} s")
        self._buffer_times.append(float(time_s))
        self._buffer_density.append(array.astype(self.storage_dtype, copy=True))
        if len(self._buffer_times) >= self.chunk_frames:
            self.flush()

    def flush(self) -> None:
        """Atomically persist the buffered prefix."""

        if not self._buffer_times:
            return
        times = np.asarray(self._buffer_times, dtype=np.float64)
        densities = np.stack(self._buffer_density)
        first = int(round(float(times[0])))
        last = int(round(float(times[-1])))
        destination = self.directory / f"frames_{first:06d}_{last:06d}.npz"
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            density_m3=densities,
            time_s=times,
            x_km=self.config.x_coordinates_m / 1.0e3 - 125.0,
            y_km=self.config.y_coordinates_m / 1.0e3,
            interval_s=self.interval_s,
            solver_dtype="float64",
            storage_dtype=self.storage_dtype.name,
            scheme=self.config.scheme,
            pdm_alpha=self.config.pdm_alpha,
            potential_operator=self.config.potential_operator,
            potential_boundary_y=self.config.potential_boundary_y,
        )
        temporary.replace(destination)
        self._stored_count += int(times.size)
        self._buffer_times.clear()
        self._buffer_density.clear()
        self._write_manifest(completed=False)

    def finalize(self, *, completed: bool) -> dict[str, Any]:
        self.flush()
        self._write_manifest(completed=completed)
        return validate_frame_archive(
            self.directory,
            expected_times=self.expected_times if completed else None,
            expected_shape=(self.config.ny, self.config.nx),
            full=completed,
        )

    def _write_manifest(self, *, completed: bool) -> None:
        chunks = []
        for path in self._chunk_paths():
            times = self._chunk_times(path)
            chunks.append(
                {
                    "file": path.name,
                    "first_time_s": float(times[0]),
                    "last_time_s": float(times[-1]),
                    "frame_count": int(times.size),
                    "bytes": path.stat().st_size,
                }
            )
        payload = {
            "format": "esf_chunked_npz_timeseries_v1",
            "completed": bool(completed and self.frame_count == self.expected_times.size),
            "frame_interval_s": self.interval_s,
            "frame_count": self.frame_count,
            "expected_frame_count": int(self.expected_times.size),
            "first_time_s": 0.0,
            "target_time_s": float(self.config.t_end_s),
            "density_shape": [self.config.ny, self.config.nx],
            "solver_dtype": "float64",
            "storage_dtype": self.storage_dtype.name,
            "density_units": "m^-3",
            "pdm_alpha": self.config.pdm_alpha,
            "scheme": self.config.scheme,
            "potential_operator": self.config.potential_operator,
            "potential_boundary_y": self.config.potential_boundary_y,
            "chunks": chunks,
        }
        temporary = self.manifest_path.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)


def validate_frame_archive(
    directory: str | Path,
    *,
    expected_times: np.ndarray | None = None,
    expected_shape: tuple[int, int] | None = None,
    full: bool = True,
) -> dict[str, Any]:
    """Validate chronological coverage and optionally every stored density array."""

    root = Path(directory)
    paths = sorted(root.glob("frames_??????_??????.npz"))
    all_times: list[np.ndarray] = []
    finite = True
    minimum = math.inf
    maximum = -math.inf
    storage_dtype = None
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            times = np.asarray(saved["time_s"], dtype=np.float64)
            all_times.append(times)
            if full:
                density = np.asarray(saved["density_m3"])
                if expected_shape is not None and density.shape[1:] != expected_shape:
                    raise FrameArchiveError(
                        f"{path} density shape {density.shape[1:]} != {expected_shape}"
                    )
                if density.shape[0] != times.size:
                    raise FrameArchiveError(f"frame/time count mismatch in {path}")
                finite = finite and bool(np.all(np.isfinite(density)))
                minimum = min(minimum, float(np.min(density)))
                maximum = max(maximum, float(np.max(density)))
                storage_dtype = str(density.dtype)
    times = np.concatenate(all_times) if all_times else np.empty(0, dtype=np.float64)
    contiguous = bool(
        times.size <= 1
        or np.allclose(np.diff(times), 1.0, rtol=0.0, atol=1.0e-9)
    )
    expected_match = expected_times is None or bool(
        times.shape == expected_times.shape
        and np.allclose(times, expected_times, rtol=0.0, atol=1.0e-9)
    )
    report = {
        "directory": str(root),
        "chunk_count": len(paths),
        "frame_count": int(times.size),
        "first_time_s": float(times[0]) if times.size else None,
        "last_time_s": float(times[-1]) if times.size else None,
        "one_second_spacing": contiguous,
        "expected_times_match": expected_match,
        "all_density_values_finite": bool(finite),
        "minimum_density_m3": None if not full or not times.size else minimum,
        "maximum_density_m3": None if not full or not times.size else maximum,
        "storage_dtype": storage_dtype,
        "total_bytes": int(sum(path.stat().st_size for path in paths)),
        "valid": bool(contiguous and expected_match and finite),
    }
    if not report["valid"]:
        raise FrameArchiveError(f"invalid frame archive: {report}")
    return report


def load_frame(directory: str | Path, time_s: int | float) -> np.ndarray:
    """Load one movie frame without expanding the full time series."""

    target = float(time_s)
    for path in sorted(Path(directory).glob("frames_??????_??????.npz")):
        match = _CHUNK_PATTERN.match(path.name)
        if match is None:
            continue
        if int(match.group(1)) <= target <= int(match.group(2)):
            with np.load(path, allow_pickle=False) as saved:
                times = np.asarray(saved["time_s"], dtype=np.float64)
                matches = np.flatnonzero(np.isclose(times, target, atol=1.0e-9))
                if matches.size == 1:
                    return np.asarray(saved["density_m3"][int(matches[0])])
    raise KeyError(f"no frame at t={target:g} s in {directory}")


__all__ = [
    "ChunkedFrameArchive",
    "FrameArchiveError",
    "load_frame",
    "validate_frame_archive",
]
