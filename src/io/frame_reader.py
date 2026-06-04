"""Multi-view frame reader for synchronized camera input.

Supports two modes:
  * File mode  – each camera source is a directory of sorted image files.
                 Hardware-trigger assumption: images are aligned by index.
                 Optional sidecar timestamp lists enable software nearest-frame
                 matching via ``nearest_frame_match``.
  * Live mode  – each camera source is an integer device index; frames are
                 grabbed from ``cv2.VideoCapture`` in sequence.

Units: timestamps are floating-point seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FrameSet:
    """One synchronised snapshot across all cameras.

    Attributes:
        frames:     camera name -> BGR image (H, W, 3) uint8.
        timestamps: camera name -> capture timestamp in seconds.
        index:      zero-based frame index within the sequence.
    """

    frames: dict[str, np.ndarray]
    timestamps: dict[str, float]
    index: int


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def nearest_frame_match(
    reference_ts: float,
    candidate_ts: list[float],
    tolerance: float,
) -> int | None:
    """Return the index of the candidate timestamp closest to *reference_ts*.

    Args:
        reference_ts:  The target timestamp (seconds).
        candidate_ts:  Ordered list of candidate timestamps to search.
        tolerance:     Maximum allowed absolute difference (seconds).
                       If the closest candidate is farther than this, return
                       ``None``.

    Returns:
        Index into *candidate_ts* of the best match, or ``None`` when no
        candidate is within *tolerance*.
    """
    if not candidate_ts:
        return None

    best_idx = int(np.argmin([abs(ts - reference_ts) for ts in candidate_ts]))
    if abs(candidate_ts[best_idx] - reference_ts) <= tolerance:
        return best_idx
    return None


# ---------------------------------------------------------------------------
# Camera spec
# ---------------------------------------------------------------------------

@dataclass
class CameraSpec:
    """Specification for one camera stream.

    Attributes:
        name:       Unique identifier used as the key in ``FrameSet.frames``.
        source:     Directory path (str or Path) for file mode, or integer
                    device index for live mode.
        timestamps: Optional list of per-frame timestamps (seconds) used in
                    file mode for software sync.  When provided, frames are
                    matched by nearest timestamp rather than by index.
    """

    name: str
    source: str | int | Path
    timestamps: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class MultiViewFrameReader:
    """Reads synchronised frames from multiple cameras.

    File mode
    ---------
    Each ``CameraSpec.source`` is a directory containing image files
    (JPEG/PNG/BMP/…).  Files are enumerated once at construction and sorted
    lexicographically so that ``cam0/frame_0001.jpg`` aligns with
    ``cam1/frame_0001.jpg`` by position.

    If *all* cameras supply ``CameraSpec.timestamps``, the reader uses
    ``nearest_frame_match`` against the first camera's timestamps as the
    reference.  Otherwise frames are aligned strictly by index.

    Live mode
    ---------
    Each ``CameraSpec.source`` is an integer device index passed to
    ``cv2.VideoCapture``.  ``read()`` calls ``cap.read()`` for every camera in
    order.

    Usage::

        specs = [
            CameraSpec("cam0", "/data/cam0"),
            CameraSpec("cam1", "/data/cam1"),
        ]
        reader = MultiViewFrameReader(specs)
        for frameset in reader:
            process(frameset)
        reader.close()
    """

    def __init__(self, specs: list[CameraSpec]) -> None:
        self._specs = specs
        self._index = 0
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._file_lists: dict[str, list[Path]] = {}

        # Detect mode from the first spec's source type.
        first_source = specs[0].source if specs else 0
        self._live = isinstance(first_source, int)

        if self._live:
            self._init_live()
        else:
            self._init_file()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self) -> FrameSet | None:
        """Read the next synchronised set of frames.

        Returns:
            A ``FrameSet`` on success, or ``None`` when the sequence is
            exhausted (file mode) or a camera fails to deliver a frame (live
            mode).
        """
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

    # ------------------------------------------------------------------
    # Internal read implementations
    # ------------------------------------------------------------------

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
        # Check that all cameras still have frames at the current index.
        for spec in self._specs:
            file_list = self._file_lists[spec.name]
            if self._index >= len(file_list):
                return None

        # Determine per-camera file index (index or nearest-timestamp match).
        use_ts_match = all(
            len(spec.timestamps) > 0 for spec in self._specs
        )

        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}

        if use_ts_match:
            # Use the first camera as the reference timestamp.
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
