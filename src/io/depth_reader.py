"""Depth frame sources for the RGB-D camera.

The depth camera SDK is not yet decided, so acquisition is hidden behind the
abstract :class:`DepthFrameSource`. Real SDK backends (Intel RealSense,
Azure Kinect, Orbbec, ...) subclass it and are plugged in later; until then the
pipeline runs on :class:`DummyDepthSource` (synthetic) or
:class:`FileDepthSource` (recorded frames).

Conventions (see also ``core/geometry.py``):
  - Depth is assumed **aligned to the colour stream**: the metric ``Z`` returned
    at colour pixel ``(u, v)`` is in **meters** (after applying ``depth_scale``).
  - ``read()`` returns ``(depth_map (H, W) float32 meters, timestamp seconds)``.
  - ``intrinsics()`` is the (3, 3) intrinsic of the stream the depth pixels live
    in (the aligned colour pixel grid).
"""

from __future__ import annotations

import abc
from pathlib import Path

import cv2
import numpy as np


class DepthFrameSource(abc.ABC):
    """Abstract source of aligned, metric depth frames.

    Real SDK backends (Intel RealSense / Azure Kinect / Orbbec) subclass this
    and implement :meth:`read`, :meth:`intrinsics`, and :attr:`depth_scale`.
    Depth is assumed already **aligned to the colour stream**, so the metric
    ``Z`` (meters) at colour pixel ``(u, v)`` can be back-projected with this
    source's intrinsic. ``open()``/``close()`` manage SDK/file handles and the
    source doubles as a context manager.
    """

    @abc.abstractmethod
    def read(self) -> tuple[np.ndarray, float]:
        """Return the next ``(depth_map (H, W) float32 meters, timestamp)``."""

    @abc.abstractmethod
    def intrinsics(self) -> np.ndarray:
        """Return the (3, 3) intrinsic matrix of the (aligned) depth stream."""

    @property
    @abc.abstractmethod
    def depth_scale(self) -> float:
        """Raw depth unit -> meters multiplier."""

    def open(self) -> None:
        """Acquire any underlying handle. No-op by default."""

    def close(self) -> None:
        """Release any underlying handle. No-op by default."""

    def __enter__(self) -> "DepthFrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class DummyDepthSource(DepthFrameSource):
    """Synthetic depth source for tests and offline development.

    Produces a constant-plane depth map (``default_z`` meters) into which
    specific metric depths can be stamped via :meth:`set_depth`. Each
    :meth:`read` returns the current map and an incrementing timestamp.
    """

    def __init__(
        self,
        width: int,
        height: int,
        K: np.ndarray,
        depth_scale: float = 1.0,
        default_z: float = 2.0,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._K = np.asarray(K, dtype=float).reshape(3, 3)
        self._depth_scale = float(depth_scale)
        self._default_z = float(default_z)
        self._map = np.full((self._height, self._width), self._default_z, dtype=np.float32)
        self._frame_idx = 0

    def set_depth(self, uv: tuple[float, float], z_meters: float) -> None:
        """Stamp a metric depth ``z_meters`` at rounded colour pixel ``(u, v)``."""
        u, v = uv
        col = int(round(float(u)))
        row = int(round(float(v)))
        if not (0 <= col < self._width and 0 <= row < self._height):
            raise IndexError(f"pixel ({u}, {v}) outside {self._width}x{self._height}")
        self._map[row, col] = np.float32(z_meters)

    def set_map(self, depth_map: np.ndarray) -> None:
        """Replace the whole depth map (meters); must match (H, W)."""
        arr = np.asarray(depth_map, dtype=np.float32)
        if arr.shape != (self._height, self._width):
            raise ValueError(f"expected ({self._height}, {self._width}), got {arr.shape}")
        self._map = arr

    def reset(self) -> None:
        """Restore the constant default plane."""
        self._map = np.full((self._height, self._width), self._default_z, dtype=np.float32)

    def read(self) -> tuple[np.ndarray, float]:
        timestamp = float(self._frame_idx)
        self._frame_idx += 1
        return self._map.copy(), timestamp

    def intrinsics(self) -> np.ndarray:
        return self._K.copy()

    @property
    def depth_scale(self) -> float:
        return self._depth_scale


class FileDepthSource(DepthFrameSource):
    """Depth source backed by a directory of recorded depth frames.

    Reads ``.npy`` (float) or 16-bit ``.png`` files in sorted filename order.
    Raw values are multiplied by ``depth_scale`` to obtain meters. Each
    :meth:`read` advances through the sequence; raises ``StopIteration`` when
    exhausted.
    """

    _SUFFIXES = (".npy", ".png")

    def __init__(self, directory: str | Path, K: np.ndarray, depth_scale: float = 1.0) -> None:
        self._directory = Path(directory)
        self._K = np.asarray(K, dtype=float).reshape(3, 3)
        self._depth_scale = float(depth_scale)
        self._paths: list[Path] = []
        self._index = 0

    def open(self) -> None:
        if not self._directory.is_dir():
            raise NotADirectoryError(f"depth frame directory not found: {self._directory}")
        self._paths = sorted(
            p for p in self._directory.iterdir() if p.suffix.lower() in self._SUFFIXES
        )
        self._index = 0

    def close(self) -> None:
        self._paths = []
        self._index = 0

    def _load_raw(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            return np.asarray(np.load(path))
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise OSError(f"failed to read depth image: {path}")
        return np.asarray(img)

    def read(self) -> tuple[np.ndarray, float]:
        if not self._paths:
            self.open()
        if self._index >= len(self._paths):
            raise StopIteration("no more depth frames")
        path = self._paths[self._index]
        raw = self._load_raw(path)
        depth_m = (raw.astype(np.float32)) * np.float32(self._depth_scale)
        timestamp = float(self._index)
        self._index += 1
        return depth_m, timestamp

    def intrinsics(self) -> np.ndarray:
        return self._K.copy()

    @property
    def depth_scale(self) -> float:
        return self._depth_scale
