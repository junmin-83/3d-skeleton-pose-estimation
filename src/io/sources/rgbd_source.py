"""RGB-D frame source abstraction (aligned colour + metric depth + intrinsics).

This realises the depth-acquisition abstraction with concrete backends (see
``tum.py``, ``realsense.py``) — previously the demos re-implemented acquisition
inline. Each yielded frame carries the BGR colour image, a metric depth map
(meters, aligned to colour), and the colour intrinsic ``K``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class RGBDFrame:
    """One aligned RGB-D frame.

    Unpacks as ``(color, depth_m, K)`` so callers can iterate either by
    attribute or by tuple unpacking.
    """

    color: np.ndarray    # (H, W, 3) BGR
    depth_m: np.ndarray  # (H, W) float32, meters (aligned to colour)
    K: np.ndarray        # (3, 3) colour intrinsic

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.color
        yield self.depth_m
        yield self.K


class RGBDSource(abc.ABC):
    """Abstract source of aligned RGB-D frames.

    Concrete backends implement :meth:`frames`. The source doubles as an
    iterator (``for frame in source``) and a context manager.
    """

    @abc.abstractmethod
    def frames(self) -> Iterator[RGBDFrame]:
        """Yield :class:`RGBDFrame` objects in acquisition order."""

    def open(self) -> None:
        """Acquire any underlying handle. No-op by default."""

    def close(self) -> None:
        """Release any underlying handle. No-op by default."""

    def __iter__(self) -> Iterator[RGBDFrame]:
        return self.frames()

    def __enter__(self) -> "RGBDSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
