"""Lazy MP4 writer shared by the example demos.

Defers opening the ``cv2.VideoWriter`` until the first frame so the frame size
is inferred from it, creates the parent directory, and uses the mp4v codec.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class LazyVideoWriter:
    """An MP4 writer that opens on the first :meth:`write` (size from the frame).

    Parameters
    ----------
    path:
        Destination ``.mp4`` path. Parent directories are created on open.
    fps:
        Playback frame rate.

    Notes
    -----
    If the codec/container cannot be opened, a warning is printed once and
    further writes are silently skipped (``opened`` stays ``False``).
    """

    def __init__(self, path: str | Path, fps: float) -> None:
        self._path = Path(path)
        self._fps = float(fps)
        self._writer: "cv2.VideoWriter | None" = None
        self.failed = False

    @property
    def opened(self) -> bool:
        return self._writer is not None

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None and not self.failed:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame_bgr.shape[:2]
            writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h)
            )
            if not writer.isOpened():
                writer.release()
                self.failed = True
                print(f"[render] WARNING: cannot open MP4 writer for {self._path}; video skipped.")
                return
            self._writer = writer
        if self._writer is not None:
            self._writer.write(frame_bgr)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
