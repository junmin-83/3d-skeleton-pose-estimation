"""Multi-view frame reader for synchronized cameras.

Two modes:
  File: each source is a directory of sorted images, aligned by index
        (hardware-trigger assumption). Supply per-camera timestamps to switch
        to nearest-frame matching instead.
  Live: each source is an integer device index, grabbed via cv2.VideoCapture.

Timestamps are float seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class FrameSet:
    """One synchronized snapshot across all cameras.

    frames: name -> BGR image (H, W, 3) uint8. timestamps: name -> seconds.
    index: zero-based position in the sequence.
    """

    frames: dict[str, np.ndarray]
    timestamps: dict[str, float]
    index: int


def nearest_frame_match(
    reference_ts: float,
    candidate_ts: list[float],
    tolerance: float,
) -> int | None:
    """Index of the candidate closest to reference_ts, or None if none within tolerance (seconds)."""
    if not candidate_ts:
        return None

    best_idx = int(np.argmin([abs(ts - reference_ts) for ts in candidate_ts]))
    if abs(candidate_ts[best_idx] - reference_ts) <= tolerance:
        return best_idx
    return None


@dataclass
class CameraSpec:
    """One camera stream.

    name: key used in FrameSet.frames. source: directory for file mode, or
    int device index for live mode. timestamps: optional per-frame seconds;
    when given (file mode), frames match by nearest timestamp instead of index.
    """

    name: str
    source: str | int | Path
    timestamps: list[float] = field(default_factory=list)


class MultiViewFrameReader:
    """Reads synchronized frames from multiple cameras.

    File mode: each source is a directory of images. Files are listed once at
    construction and sorted lexicographically, so cam0/frame_0001.jpg lines up
    with cam1/frame_0001.jpg by position. If every camera has timestamps, sync
    uses nearest_frame_match against the first camera; otherwise by index.

    Live mode: each source is an int device index. read() pulls one frame from
    every camera in order.

    Iterate the reader, then close() (or use it as a context manager).
    """

    def __init__(self, specs: list[CameraSpec]) -> None:
        self._specs = specs
        self._index = 0
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._file_lists: dict[str, list[Path]] = {}

        # Mode comes from the first source's type.
        first_source = specs[0].source if specs else 0
        self._live = isinstance(first_source, int)

        if self._live:
            self._init_live()
        else:
            self._init_file()

    def _init_live(self) -> None:
        for spec in self._specs:
            cap = cv2.VideoCapture(int(spec.source))
            if not cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera device {spec.source!r} for '{spec.name}'."
                )
            self._caps[spec.name] = cap

    def _init_file(self) -> None:
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        for spec in self._specs:
            directory = Path(str(spec.source))
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Source directory not found for camera '{spec.name}': {directory}"
                )
            files = sorted(
                p for p in directory.iterdir() if p.suffix.lower() in image_extensions
            )
            if not files:
                raise ValueError(
                    f"No image files found in directory for camera '{spec.name}': {directory}"
                )
            self._file_lists[spec.name] = files

    def read(self) -> FrameSet | None:
        """Next FrameSet, or None when the sequence ends (file) or a camera drops a frame (live)."""
        if self._live:
            return self._read_live()
        return self._read_file()

    def close(self) -> None:
        """Release all resources."""
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def __iter__(self) -> Iterator[FrameSet]:
        while True:
            frameset = self.read()
            if frameset is None:
                break
            yield frameset

    def __enter__(self) -> "MultiViewFrameReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_live(self) -> FrameSet | None:
        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        now = time.time()
        for spec in self._specs:
            cap = self._caps[spec.name]
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            frames[spec.name] = frame
            timestamps[spec.name] = now
        result = FrameSet(frames=frames, timestamps=timestamps, index=self._index)
        self._index += 1
        return result

    def _read_file(self) -> FrameSet | None:
        # Stop once any camera runs out of frames.
        for spec in self._specs:
            file_list = self._file_lists[spec.name]
            if self._index >= len(file_list):
                return None

        # Match by timestamp only if every camera has them; else by index.
        use_ts_match = all(
            len(spec.timestamps) > 0 for spec in self._specs
        )

        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}

        if use_ts_match:
            # First camera is the timestamp reference.
            ref_spec = self._specs[0]
            ref_ts = ref_spec.timestamps[self._index]

            for spec in self._specs:
                if spec is ref_spec:
                    file_idx = self._index
                else:
                    matched = nearest_frame_match(
                        reference_ts=ref_ts,
                        candidate_ts=list(spec.timestamps),
                        tolerance=float("inf"),  # match best available
                    )
                    file_idx = matched if matched is not None else self._index

                path = self._file_lists[spec.name][file_idx]
                img = cv2.imread(str(path))
                if img is None:
                    raise IOError(f"Failed to read image: {path}")
                frames[spec.name] = img
                timestamps[spec.name] = (
                    spec.timestamps[file_idx]
                    if file_idx < len(spec.timestamps)
                    else float(self._index)
                )
        else:
            for spec in self._specs:
                path = self._file_lists[spec.name][self._index]
                img = cv2.imread(str(path))
                if img is None:
                    raise IOError(f"Failed to read image: {path}")
                frames[spec.name] = img
                timestamps[spec.name] = float(self._index)

        result = FrameSet(frames=frames, timestamps=timestamps, index=self._index)
        self._index += 1
        return result
