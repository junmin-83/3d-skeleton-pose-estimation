"""Intel RealSense live camera as an :class:`RGBDSource` (needs pyrealsense2).

``pyrealsense2`` is imported lazily inside :meth:`RealSenseSource.frames` so the
rest of the package works without it installed.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from src.io.sources.rgbd_source import RGBDFrame, RGBDSource


class RealSenseSource(RGBDSource):
    """Live aligned RGB-D frames from an Intel RealSense camera.

    Depth is aligned to the colour stream; intrinsics and depth scale are read
    from the device. Yields up to ``num`` frames (startup/dropped frames are
    skipped, matching the original demo behaviour).
    """

    def __init__(self, num: int, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self._num = num
        self._size = (width, height)
        self._fps = fps

    def frames(self) -> Iterator[RGBDFrame]:
        import pyrealsense2 as rs  # lazy: only needed for RealSense

        w, h = self._size
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self._fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, self._fps)
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)  # depth -> color alignment
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        try:
            for _ in range(self._num):
                frames = align.process(pipe.wait_for_frames())
                cf, df = frames.get_color_frame(), frames.get_depth_frame()
                if not cf or not df:  # startup / dropped frame
                    continue
                color = np.asanyarray(cf.get_data())
                depth = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale
                yield RGBDFrame(color, depth, K)
        finally:
            pipe.stop()
