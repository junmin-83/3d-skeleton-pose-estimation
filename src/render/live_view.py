"""Live OpenCV window for the multi-view pipeline.

Each frame, draw the per-camera 2D overlays next to a matplotlib 3D skeleton in
one image and show it in a cv2 window. run.py's --live mode uses this so a webcam
(or any configured source) can be watched in real time instead of only written
to a file. Press q or Esc to stop.

The 3D panel is rendered with matplotlib (Agg) per frame, so frame rate is bound
by detection + that render (roughly 15-20 FPS for the 3D draw alone).
"""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.core.types import CameraParams, Pose3D  # noqa: E402
from src.render.skeleton_2d import draw_skeleton_2d, label_panel  # noqa: E402
from src.render.skeleton_3d import render_pose3d_frame  # noqa: E402

_WINDOW = "3D skeleton (live) - press q or Esc to quit"


class LiveView:
    """A reusable cv2 window drawing 2D overlays + a 3D skeleton each frame."""

    def __init__(
        self,
        cameras: list[CameraParams],
        score_thr: float = 0.3,
        panel_size: tuple[int, int] = (360, 240),
        pad: float = 0.7,
    ) -> None:
        self.cameras = cameras
        self.score_thr = float(score_thr)
        self.pw, self.ph = panel_size
        self.pad = float(pad)
        self._fig = plt.figure(figsize=(self.pw / 100, self.ph / 100), dpi=100)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._lims = None  # fixed once from the first reconstructed pose

    def _compose(self, frameset, keypoints: np.ndarray, scores: np.ndarray,
                 pose: Pose3D) -> np.ndarray:
        """Build the composite frame: one 2D panel per view + a 3D panel."""
        panels = []
        for i, cam in enumerate(self.cameras):
            frame = frameset.frames[cam.name]
            h0, w0 = frame.shape[:2]
            panel = cv2.resize(frame, (self.pw, self.ph))
            draw_skeleton_2d(panel, keypoints[i], scores[i], self.score_thr,
                             scale=(self.pw / w0, self.ph / h0))
            panels.append(label_panel(panel, f"{cam.name} 2D", self.pw))

        if self._lims is None and pose.valid.any():
            c = pose.points[pose.valid].mean(axis=0)
            self._lims = [(c[a] - self.pad, c[a] + self.pad) for a in range(3)]
        p3d = render_pose3d_frame(self._fig, self._ax, pose,
                                  self._lims or [(-1, 1)] * 3, (self.pw, self.ph),
                                  view_init=(-75, -90))
        panels.append(label_panel(p3d, "3D", self.pw))
        return np.hstack(panels)

    def show(self, frameset, keypoints: np.ndarray, scores: np.ndarray,
             pose: Pose3D) -> bool:
        """Render one frame to the window; return False when the user quits."""
        cv2.imshow(_WINDOW, self._compose(frameset, keypoints, scores, pose))
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)

    def close(self) -> None:
        """Tear down the matplotlib figure and the cv2 window."""
        plt.close(self._fig)
        cv2.destroyAllWindows()
