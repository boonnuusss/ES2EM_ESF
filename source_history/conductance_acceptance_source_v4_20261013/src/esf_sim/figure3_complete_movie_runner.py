"""Compatibility runner with bidirectional chunk lookup after seam validation."""

from __future__ import annotations

import numpy as np

from . import figure3_complete_movie as workflow


def _seekable_frame(self, time_s: int) -> np.ndarray:
    if time_s < self.first_time_s or time_s > self.last_time_s:
        raise KeyError(f"t={time_s} is outside {self.first_time_s}:{self.last_time_s}")
    if self._times is None or not (self._times[0] <= time_s <= self._times[-1]):
        for index, entry in enumerate(self.chunks):
            if int(entry["first_time_s"]) <= time_s <= int(entry["last_time_s"]):
                self._load_chunk(index)
                break
        else:
            raise KeyError(f"missing t={time_s} in {self.directory}")
    assert self._times is not None and self._density is not None
    return self._density[time_s - int(self._times[0])]


workflow.ArchiveSegment.frame = _seekable_frame


def main(argv: list[str] | None = None) -> None:
    workflow.main(argv)


if __name__ == "__main__":
    main()
