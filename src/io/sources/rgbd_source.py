"""RGB-D frame source: aligned colour, metric depth, and intrinsics.

Concrete backends live in tum.py and realsense.py (demos used to inline this).
Each frame is the BGR colour image, a depth map (meters, aligned to colour),
and the colour intrinsic K.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class RGBDFrame:
    """One aligned RGB-D frame. Unpacks as (color, depth_m, K)."""

    color: np.ndarray    # (H, W, 3) BGR
    depth_m: np.ndarray  # (H, W) float32, meters (aligned to colour)
    K: np.ndarray        # (3, 3) colour intrinsic

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.color
        yield self.depth_m
        yield self.K


class RGBDSource(abc.ABC):
    """Abstract source of aligned RGB-D frames.

    Backends implement frames(). Also an iterator and a context manager.
    """

    @abc.abstractmethod
    def frames(self) -> Iterator[RGBDFrame]:
        """Yield RGBDFrames in acquisition order."""

    def open(self) -> None:
        """Acquire the handle. No-op by default."""

    def close(self) -> None:
        """Release the handle. No-op by default."""

    def __iter__(self) -> Iterator[RGBDFrame]:
        return self.frames()

    def __enter__(self) -> "RGBDSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
