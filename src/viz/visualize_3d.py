"""3D skeleton visualisation and keypoint export/import utilities.

The matplotlib Agg backend is activated at import time so this module works
in headless/test environments without a display server.

Units: all 3D coordinates are in **meters** (world frame, COCO-17 layout).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection

from src.core.types import COCO_SKELETON, Pose3D


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_skeleton_3d(
    pose3d: Pose3D,
    ax: "Axes3D | None" = None,
    title: str | None = None,
    elev: float = 15.0,
    azim: float = -70.0,
) -> tuple[plt.Figure, "Axes3D"]:
    """Plot a 3D skeleton on a matplotlib 3D axes.

    Only valid keypoints (``pose3d.valid == True``) are drawn.  Bones that
    touch at least one invalid joint are skipped.

    Args:
        pose3d: The 3D pose to visualise.
        ax:     Existing ``Axes3D`` to draw on.  A new figure+axes is created
                when ``None``.
        title:  Optional axes title.
        elev:   Elevation angle for the 3D view (degrees).
        azim:   Azimuth angle for the 3D view (degrees).

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    pts = pose3d.points   # (K, 3)
    valid = pose3d.valid  # (K,) bool

    # Scatter valid keypoints.
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

    # Draw bones.
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

    # Equal-ish aspect ratio: set equal range on all axes.
    if valid_pts.size > 0:
        ranges = valid_pts.max(axis=0) - valid_pts.min(axis=0)
        max_range = float(ranges.max()) if ranges.max() > 0 else 1.0
        mid = (valid_pts.max(axis=0) + valid_pts.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    return fig, ax


def save_skeleton_png(pose3d: Pose3D, path: str) -> None:
    """Render a 3D skeleton and save it as a PNG file.

    The parent directory is created automatically if it does not exist.

    Args:
        pose3d: The 3D pose to render.
        path:   Destination file path (should end in ``.png``).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_skeleton_3d(pose3d)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Export / import
# ---------------------------------------------------------------------------

def export_keypoints(
    poses: "Pose3D | Sequence[Pose3D]",
    path: str,
    fmt: str = "json",
) -> None:
    """Export one or more poses to a file.

    Args:
        poses: A single ``Pose3D`` or a list/sequence of them (one per frame).
        path:  Destination file path.  For ``fmt="npy"`` the path should end
               in ``.npy``; a sidecar ``<path>.npz`` is written alongside it
               containing ``scores``, ``valid``, and ``source``.
        fmt:   ``"json"`` (default) or ``"npy"``.

    JSON format
    -----------
    A JSON array where each element is a dict::

        {
            "points": [[x, y, z], ...],   # K × 3 float
            "scores": [s0, s1, ...],       # K float
            "valid":  [true, false, ...],  # K bool
            "source": ["tri", ...]         # K str
        }

    NPY format
    ----------
    The ``.npy`` file contains a ``float64`` array of shape ``(N, K, 3)``.
    The sidecar ``.npz`` contains ``scores (N, K)``, ``valid (N, K)``, and
    ``source (N, K)`` (object/str dtype).
    """
    # Normalise to list.
    if isinstance(poses, Pose3D):
        pose_list: list[Pose3D] = [poses]
    else:
        pose_list = list(poses)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        records = []
        for p in pose_list:
            records.append(
                {
                    "points": p.points.tolist(),
                    "scores": p.scores.tolist(),
                    "valid": p.valid.tolist(),
                    "source": list(p.source),
                }
            )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)

    elif fmt == "npy":
        points_arr = np.stack([p.points for p in pose_list], axis=0)  # (N, K, 3)
        scores_arr = np.stack([p.scores for p in pose_list], axis=0)  # (N, K)
        valid_arr = np.stack([p.valid for p in pose_list], axis=0)    # (N, K)
        source_arr = np.array(
            [p.source for p in pose_list], dtype=object
        )  # (N, K) or (N, len(source))

        np.save(str(out_path), points_arr)
        sidecar = out_path.with_suffix(".npy.npz")
        np.savez(
            str(sidecar),
            scores=scores_arr,
            valid=valid_arr,
            source=source_arr,
        )
    else:
        raise ValueError(f"Unsupported export format: {fmt!r}. Use 'json' or 'npy'.")


def load_keypoints(path: str, fmt: str = "json") -> list[Pose3D]:
    """Load poses previously saved by :func:`export_keypoints`.

    Args:
        path: File path passed to ``export_keypoints``.
        fmt:  ``"json"`` or ``"npy"``.

    Returns:
        List of :class:`~src.core.types.Pose3D` objects (one per frame).
    """
    in_path = Path(path)

    if fmt == "json":
        with open(in_path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        poses = []
        for rec in records:
            poses.append(
                Pose3D(
                    points=np.array(rec["points"], dtype=float),
                    scores=np.array(rec["scores"], dtype=float),
                    valid=np.array(rec["valid"], dtype=bool),
                    source=list(rec.get("source", [])),
                )
            )
        return poses

    elif fmt == "npy":
        points_arr = np.load(str(in_path))          # (N, K, 3)
        sidecar = in_path.with_suffix(".npy.npz")
        data = np.load(str(sidecar), allow_pickle=True)
        scores_arr = data["scores"]   # (N, K)
        valid_arr = data["valid"]     # (N, K)
        source_arr = data["source"]   # (N, ...)

        poses = []
        for i in range(len(points_arr)):
            src = list(source_arr[i]) if i < len(source_arr) else []
            poses.append(
                Pose3D(
                    points=points_arr[i],
                    scores=scores_arr[i],
                    valid=valid_arr[i].astype(bool),
                    source=src,
                )
            )
        return poses

    else:
        raise ValueError(f"Unsupported load format: {fmt!r}. Use 'json' or 'npy'.")
