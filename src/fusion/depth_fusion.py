"""Depth fusion: back-project aligned depth and fuse it with triangulation.

The RGB-D view gives a metric Z per colour pixel. Per COCO-17 keypoint we sample
that depth, back-project the pixel to a world point, and merge it with the
multi-view triangulation result:

  - both valid: confidence-weighted average of the two world points
  - only triangulation valid: keep triangulation
  - only depth valid (and fill_missing): fill from depth
  - neither valid: mark missing

Conventions (see core/geometry.py): meters; (u, v) pixel order; world is the
reference camera frame; extrinsics (R, t) map world to camera, here the depth
camera's (depth assumed aligned to its colour stream).
"""

from __future__ import annotations

import numpy as np

from src.core.geometry import back_project_pixels
from src.core.types import Pose3D


def sample_depth(
    depth_map: np.ndarray,
    uv: np.ndarray,
    patch_radius: int = 2,
    depth_min: float = 0.2,
    depth_max: float = 6.0,
) -> tuple[float, bool]:
    """Robust metric depth around pixel (u, v): median of the valid samples.

    Uses the (2*patch_radius + 1) square patch centred on the rounded (u, v).
    A depth is invalid if it's 0, NaN, inf, or outside [depth_min, depth_max];
    pixels outside the map are ignored.

    Returns:
        (z, valid): z is the median depth in meters (nan if no usable sample),
        valid says whether any usable sample existed.
    """
    depth_map = np.asarray(depth_map)
    height, width = depth_map.shape[:2]

    u, v = float(uv[0]), float(uv[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return float("nan"), False
    col = int(round(u))
    row = int(round(v))
    r = int(patch_radius)

    row_lo = max(0, row - r)
    row_hi = min(height, row + r + 1)
    col_lo = max(0, col - r)
    col_hi = min(width, col + r + 1)
    if row_lo >= row_hi or col_lo >= col_hi:
        return float("nan"), False

    patch = np.asarray(depth_map[row_lo:row_hi, col_lo:col_hi], dtype=float).ravel()
    finite = patch[np.isfinite(patch)]
    usable = finite[(finite > 0.0) & (finite >= depth_min) & (finite <= depth_max)]
    if usable.size == 0:
        return float("nan"), False
    return float(np.median(usable)), True


def back_project_depth_keypoints(
    uv: np.ndarray,
    depth_map: np.ndarray,
    depth_K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    patch_radius: int = 2,
    depth_min: float = 0.2,
    depth_max: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depth-view keypoints to world coordinates.

    Args:
        uv: (K, 2) colour-pixel coords (u, v) per keypoint.
        depth_map: (H, W) metric depth (meters), aligned to the colour stream.
        depth_K: (3, 3) intrinsic of the (aligned) depth stream.
        R, t: depth camera extrinsics (world to camera).
        patch_radius, depth_min, depth_max: forwarded to sample_depth.

    Returns:
        (points_world (K, 3), valid (K,)). Unusable rows are nan and valid=False.
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    num_kpts = uv.shape[0]
    points_world = np.full((num_kpts, 3), np.nan, dtype=float)
    valid = np.zeros(num_kpts, dtype=bool)

    for k in range(num_kpts):
        z, ok = sample_depth(
            depth_map, uv[k], patch_radius=patch_radius,
            depth_min=depth_min, depth_max=depth_max,
        )
        if not ok:
            continue
        points_world[k] = back_project_pixels(depth_K, uv[k], z, R, t)
        valid[k] = True

    return points_world, valid


def fuse(
    triangulated: Pose3D,
    depth_points: np.ndarray,
    depth_valid: np.ndarray,
    depth_scores: np.ndarray,
    fill_missing: bool = True,
    depth_weight: float = 1.0,
) -> Pose3D:
    """Fuse triangulation and depth back-projection into one Pose3D (new object).

    Per keypoint:
      * both valid: confidence-weighted average of the two world points (weights
        triangulated.scores[k] and depth_scores[k] * depth_weight),
        source='fused', combined score.
      * triangulation valid, depth invalid: keep triangulation, source='triangulation'.
      * triangulation invalid, depth valid, fill_missing: use the depth point,
        source='depth', valid=True.
      * else: valid=False, source='missing'.
    """
    tri_points = np.asarray(triangulated.points, dtype=float).reshape(-1, 3)
    tri_scores = np.asarray(triangulated.scores, dtype=float).reshape(-1)
    tri_valid = np.asarray(triangulated.valid, dtype=bool).reshape(-1)

    depth_points = np.asarray(depth_points, dtype=float).reshape(-1, 3)
    depth_valid = np.asarray(depth_valid, dtype=bool).reshape(-1)
    depth_scores = np.asarray(depth_scores, dtype=float).reshape(-1)

    num_kpts = tri_points.shape[0]
    points = np.zeros((num_kpts, 3), dtype=float)
    scores = np.zeros(num_kpts, dtype=float)
    valid = np.zeros(num_kpts, dtype=bool)
    source: list[str] = ["missing"] * num_kpts

    for k in range(num_kpts):
        tri_ok = bool(tri_valid[k])
        dep_ok = bool(depth_valid[k])

        if tri_ok and dep_ok:
            w_tri = float(tri_scores[k])
            w_dep = float(depth_scores[k]) * float(depth_weight)
            w_sum = w_tri + w_dep
            if w_sum > 0.0:
                points[k] = (w_tri * tri_points[k] + w_dep * depth_points[k]) / w_sum
            else:
                points[k] = 0.5 * (tri_points[k] + depth_points[k])
            scores[k] = 0.5 * (float(tri_scores[k]) + float(depth_scores[k]))
            valid[k] = True
            source[k] = "fused"
        elif tri_ok:
            points[k] = tri_points[k]
            scores[k] = float(tri_scores[k])
            valid[k] = True
            source[k] = "triangulation"
        elif dep_ok and fill_missing:
            points[k] = depth_points[k]
            scores[k] = float(depth_scores[k])
            valid[k] = True
            source[k] = "depth"
        else:
            points[k] = tri_points[k]
            scores[k] = 0.0
            valid[k] = False
            source[k] = "missing"

    return Pose3D(points=points, scores=scores, valid=valid, source=source)
