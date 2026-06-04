"""Confidence-weighted Direct Linear Transform (DLT) triangulation.

Coordinate conventions (see ``src/core/geometry.py`` and ``src/core/types.py``):
  - Pixels are ``(u, v)`` order and are assumed **already undistorted**.
  - Projection matrices ``P = K [R | t]`` map a homogeneous **world** point to
    pixels (world -> pixel). The world frame is the single fixed project frame
    (default: reference camera cam0). All lengths are **meters**.
  - Triangulated points are returned in that same **world** frame, meters.

This module performs pure linear triangulation only. View selection, score
thresholding and outlier rejection live in ``robust.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def triangulate_point_dlt(
    points_2d: np.ndarray,
    proj_matrices: Sequence[np.ndarray],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Triangulate one 3D world point from multi-view pixel observations.

    Builds the homogeneous DLT linear system and solves it via SVD. For each
    view with observed pixel ``(u, v)``, per-view weight ``w`` and projection
    rows ``P0, P1, P2`` of ``P = K [R | t]``::

        row_a = w * (u * P2 - P0)
        row_b = w * (v * P2 - P1)

    Stacking the ``2V`` rows gives ``A`` (shape ``(2V, 4)``). The solution is
    the right-singular vector of ``A`` for the smallest singular value, then
    dehomogenized by dividing by its last component.

    Args:
        points_2d: (V, 2) pixel observations ``(u, v)`` across ``V`` views,
            undistorted, in the same view order as ``proj_matrices``.
        proj_matrices: sequence of ``V`` (3, 4) world->pixel matrices ``P``.
        weights: (V,) non-negative per-view confidence. Defaults to all ones
            (unweighted). Each view's two equations are scaled by its weight.

    Returns:
        (3,) triangulated point in the **world** frame, meters.
    """
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    n_views = pts.shape[0]
    proj = [np.asarray(P, dtype=float).reshape(3, 4) for P in proj_matrices]
    if len(proj) != n_views:
        raise ValueError(
            f"proj_matrices has {len(proj)} entries but points_2d has "
            f"{n_views} views"
        )
    if n_views < 2:
        raise ValueError(f"need >= 2 views to triangulate, got {n_views}")

    if weights is None:
        w = np.ones(n_views, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n_views:
            raise ValueError(
                f"weights has {w.shape[0]} entries but points_2d has "
                f"{n_views} views"
            )

    if not np.any(np.abs(w) > 1e-12):
        # All contributing views have zero weight -> the system carries no
        # constraint; don't fabricate a point. robust.py drops the joint on NaN.
        return np.full(3, np.nan)

    rows = np.empty((2 * n_views, 4), dtype=float)
    for i, (P, (u, v)) in enumerate(zip(proj, pts)):
        rows[2 * i] = w[i] * (u * P[2] - P[0])
        rows[2 * i + 1] = w[i] * (v * P[2] - P[1])

    # Smallest-singular-value right vector -> homogeneous solution X = (x,y,z,W).
    _, _, vh = np.linalg.svd(rows)
    x_hom = vh[-1]
    # Degenerate/collinear geometry (parallel rays, a point near infinity) drives
    # the homogeneous coordinate W -> 0; dehomogenizing would yield inf/NaN that
    # bypasses outlier rejection and poisons the temporal filter. Flag as NaN so
    # robust.py drops the joint.
    if abs(x_hom[3]) < 1e-9:
        return np.full(3, np.nan)
    return x_hom[:3] / x_hom[3]


def triangulate_keypoints(
    keypoints_per_view: np.ndarray,
    scores_per_view: np.ndarray,
    proj_matrices: Sequence[np.ndarray],
    weight_by_score: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-DLT triangulate every keypoint over *all* provided views.

    No thresholding or view selection happens here; every view contributes to
    every keypoint (occlusion handling lives in ``robust.py``). When
    ``weight_by_score`` is True the per-view ``score`` is used as the DLT weight
    so low-confidence (e.g. occluded) observations are down-weighted.

    Args:
        keypoints_per_view: (V, K, 2) undistorted pixel coords ``(u, v)``.
        scores_per_view: (V, K) per-view per-keypoint confidence in [0, 1].
        proj_matrices: sequence of ``V`` (3, 4) world->pixel matrices, same
            view order as the first axis of ``keypoints_per_view``.
        weight_by_score: if True weight each view by its score, else weight all
            views equally (unit weights).

    Returns:
        points_3d: (K, 3) world coords, meters.
        conf: (K,) mean of the per-view scores used for each keypoint.
    """
    kpts = np.asarray(keypoints_per_view, dtype=float)
    scores = np.asarray(scores_per_view, dtype=float)
    if kpts.ndim != 3 or kpts.shape[2] != 2:
        raise ValueError("keypoints_per_view must have shape (V, K, 2)")
    n_views, n_kpts, _ = kpts.shape
    if scores.shape != (n_views, n_kpts):
        raise ValueError("scores_per_view must have shape (V, K)")
    proj = [np.asarray(P, dtype=float).reshape(3, 4) for P in proj_matrices]
    if len(proj) != n_views:
        raise ValueError("proj_matrices length must match number of views")

    points_3d = np.empty((n_kpts, 3), dtype=float)
    conf = np.empty(n_kpts, dtype=float)
    for k in range(n_kpts):
        view_scores = scores[:, k]
        weights = view_scores if weight_by_score else None
        points_3d[k] = triangulate_point_dlt(kpts[:, k, :], proj, weights)
        conf[k] = float(np.mean(view_scores))

    return points_3d, conf
