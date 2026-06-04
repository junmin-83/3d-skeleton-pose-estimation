"""3D skeleton rendering (COCO-17, meters, world frame).

Two paths:
  - render_pose3d_frame draws a pose onto a reused matplotlib 3D axes and returns
    a BGR image (video demos, panel by panel).
  - plot_skeleton_3d / save_skeleton_png produce a one-off equal-aspect plot for
    PNG export (run.py).

Agg backend is set at import time for headless use.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401 - registers 3D projection

from src.core.types import COCO_SKELETON, Pose3D  # noqa: E402


def _draw_bones(ax, pts: np.ndarray, valid: np.ndarray, color, lw: float) -> None:
    """Plot COCO bones with both endpoints valid onto ax."""
    for i, j in COCO_SKELETON:
        if i < len(valid) and j < len(valid) and valid[i] and valid[j]:
            ax.plot(*[[pts[i, a], pts[j, a]] for a in range(3)], c=color, lw=lw)


def render_pose3d_frame(
    fig,
    ax,
    pose: Pose3D,
    lims,
    out_size: tuple[int, int],
    *,
    point_size: int = 18,
    view_init: tuple[float, float] | None = None,
    color: str = "royalblue",
) -> np.ndarray:
    """Draw pose on a reused 3D ax, return a BGR image of out_size.

    lims is ((xmin,xmax), (ymin,ymax), (zmin,zmax)); view_init is an optional
    (elev, azim). The axes is cleared each call so one figure serves all frames.
    """
    ax.cla()
    pts, valid = pose.points, pose.valid
    _draw_bones(ax, pts, valid, color, 2)
    if valid.any():
        ax.scatter(*pts[valid].T, c=color, s=point_size)
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])
    ax.set_xlabel("X(m)")
    ax.set_ylabel("Y(m)")
    ax.set_zlabel("Z(m)")
    if view_init is not None:
        ax.view_init(elev=view_init[0], azim=view_init[1])
    fig.canvas.draw()
    bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    return cv2.resize(bgr, out_size)


def plot_skeleton_3d(
    pose3d: Pose3D,
    ax: "Axes3D | None" = None,
    title: str | None = None,
    elev: float = 15.0,
    azim: float = -70.0,
) -> tuple[plt.Figure, "Axes3D"]:
    """Plot a 3D skeleton on a matplotlib 3D axes (equal aspect).

    Only valid keypoints are drawn; bones touching an invalid joint are skipped.
    Creates a new figure+axes when ax is None. Returns (fig, ax).
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    pts = pose3d.points   # (K, 3)
    valid = pose3d.valid  # (K,) bool

    valid_pts = pts[valid]
    if valid_pts.size > 0:
        ax.scatter(
            valid_pts[:, 0],
            valid_pts[:, 1],
            valid_pts[:, 2],
            c="royalblue",
            s=30,
            zorder=5,
        )

    for i, j in COCO_SKELETON:
        if i < len(valid) and j < len(valid) and valid[i] and valid[j]:
            xs = [pts[i, 0], pts[j, 0]]
            ys = [pts[i, 1], pts[j, 1]]
            zs = [pts[i, 2], pts[j, 2]]
            ax.plot(xs, ys, zs, color="steelblue", linewidth=1.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title)

    # Equal-ish aspect: same range on all three axes.
    if valid_pts.size > 0:
        ranges = valid_pts.max(axis=0) - valid_pts.min(axis=0)
        max_range = float(ranges.max()) if ranges.max() > 0 else 1.0
        mid = (valid_pts.max(axis=0) + valid_pts.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    return fig, ax


def save_skeleton_png(pose3d: Pose3D, path: str) -> None:
    """Render a 3D skeleton and save it as a PNG (parent dir created)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_skeleton_3d(pose3d)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
