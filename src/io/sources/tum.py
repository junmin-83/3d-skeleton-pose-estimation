"""TUM RGB-D dataset as an :class:`RGBDSource`.

Associates each colour frame with its nearest-timestamp depth frame and yields
aligned RGB-D frames. Intrinsics + depth scale are the freiburg3 defaults
(fx=535.4, fy=539.2, cx=320.1, cy=247.6; raw_depth / 5000 = meters).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.io.sources.rgbd_source import RGBDFrame, RGBDSource

# TUM freiburg3 RGB intrinsics + depth scale (raw / scale = meters).
_TUM_FR3 = dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6, depth_scale=5000.0)


def read_tum_assoc(tum_dir: Path, tolerance_s: float = 0.02) -> list[tuple[str, str]]:
    """Associate each RGB frame with the nearest-timestamp depth frame.

    Returns a list of ``(rgb_filename, depth_filename)`` pairs whose timestamps
    differ by at most ``tolerance_s`` seconds.
    """
    def load(name: str) -> list[tuple[float, str]]:
        out = []
        for line in (tum_dir / name).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, fname = line.split()
            out.append((float(ts), fname))
        return out

    rgb, depth = load("rgb.txt"), load("depth.txt")
    dts = np.array([t for t, _ in depth])
    pairs = []
    for ts, rgb_f in rgb:
        j = int(np.argmin(np.abs(dts - ts)))
        if abs(dts[j] - ts) <= tolerance_s:
            pairs.append((rgb_f, depth[j][1]))
    return pairs


class TUMSource(RGBDSource):
    """RGB-D frames from a TUM ``rgbd_dataset_freiburg3_*`` directory."""

    def __init__(self, tum_dir: str | Path, start: int = 0, num: int | None = None) -> None:
        self._dir = Path(tum_dir)
        self._start = start
        self._num = num

    @property
    def intrinsics(self) -> np.ndarray:
        return np.array([[_TUM_FR3["fx"], 0, _TUM_FR3["cx"]],
                         [0, _TUM_FR3["fy"], _TUM_FR3["cy"]], [0, 0, 1]])

    def frames(self) -> Iterator[RGBDFrame]:
        end = None if self._num is None else self._start + self._num
        pairs = read_tum_assoc(self._dir)[self._start:end]
        K = self.intrinsics
        for rgb_f, depth_f in pairs:
            color = cv2.imread(str(self._dir / rgb_f))
            raw = cv2.imread(str(self._dir / depth_f), cv2.IMREAD_UNCHANGED)
            if color is None or raw is None:
                continue
            depth_m = raw.astype(np.float32) / _TUM_FR3["depth_scale"]
            yield RGBDFrame(color, depth_m, K)
