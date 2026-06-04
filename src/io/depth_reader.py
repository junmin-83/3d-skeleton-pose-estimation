"""Depth frame sources for the RGB-D camera.

The depth camera SDK is not yet decided, so acquisition is hidden behind the
abstract :class:`DepthFrameSource`. Real SDK / dataset backends (Intel RealSense,
TUM RGB-D, Azure Kinect, ...) subclass it and are plugged in later; tests and
offline development run on :class:`DummyDepthSource` (synthetic).

Conventions (see also ``core/geometry.py``):
  - Depth is assumed **aligned to the colour stream**: the metric ``Z`` returned
    at colour pixel ``(u, v)`` is in **meters** (after applying ``depth_scale``).
  - ``read()`` returns ``(depth_map (H, W) float32 meters, timestamp seconds)``.
  - ``intrinsics()`` is the (3, 3) intrinsic of the stream the depth pixels live
    in (the aligned colour pixel grid).
"""

from __future__ import annotations

import abc

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
