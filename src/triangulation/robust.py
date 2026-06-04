"""Robust multi-view triangulation: view selection + outlier rejection.

Wraps the pure confidence-weighted DLT (:mod:`src.triangulation.dlt`) with the
per-keypoint policy from SPEC section 4-4:

  - drop views whose score is below ``score_threshold``;
  - if fewer than ``min_views`` remain, mark the joint invalid and defer it to
    depth fusion (``source='missing'``);
  - otherwise weighted-DLT triangulate from the kept views;
  - optionally (``ransac=True``) reject views whose reprojection error exceeds
    ``reproj_threshold_px`` and re-triangulate from the inliers.

Coordinate conventions match the rest of the project: pixels are ``(u, v)`` and
undistorted; ``P = K [R | t]`` maps world -> pixel; results are in the **world**
frame in meters. Returns a :class:`~src.core.types.Pose3D`.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from src.core.geometry import project_points
from src.core.types import Pose3D
from src.triangulation.dlt import triangulate_point_dlt


def _reprojection_errors(
    point_3d: np.ndarray,
    points_2d: np.ndarray,
    proj_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    """Per-view reprojection error in pixels for one triangulated 3D point."""
    errs = np.empty(len(proj_matrices), dtype=float)
    for i, P in enumerate(proj_matrices):
        uv_hat = project_points(P, point_3d)
        errs[i] = float(np.linalg.norm(uv_hat - points_2d[i]))
    return errs


def _ransac_inliers(
    points_2d: np.ndarray,
    proj_matrices: Sequence[np.ndarray],
    weights: np.ndarray,
    min_views: int,
    reproj_threshold_px: float,
) -> np.ndarray | None:
    """Find the largest consistent set of views via minimal-subset RANSAC.

    A least-squares DLT over *all* views is biased by an outlier, inflating the
    residuals of the good views too, so a single fit-then-threshold pass cannot
    isolate the outlier. Instead, enumerate every minimal (2-view) subset,
    triangulate it, and score how many views fall within ``reproj_threshold_px``
    of that hypothesis. The hypothesis with the most inliers (ties broken by
    lowest total inlier error) wins.

    Args:
        points_2d: (V, 2) undistorted pixel observations.
        proj_matrices: ``V`` (3, 4) world->pixel matrices.
        weights: (V,) per-view confidence used when re-fitting a subset.
        min_views: minimum inlier count to accept a consensus set.
        reproj_threshold_px: inlier reprojection-error threshold (pixels).

    Returns:
        Indices (into the V views) of the best inlier set, or ``None`` if no
        consensus of at least ``min_views`` views exists.
    """
    n_views = len(proj_matrices)
    best_inliers: np.ndarray | None = None
    best_count = 0
    best_err = np.inf

    for combo in combinations(range(n_views), 2):
        idx = list(combo)
        hypothesis = triangulate_point_dlt(
            points_2d[idx], [proj_matrices[i] for i in idx], weights[idx]
        )
        errs = _reprojection_errors(hypothesis, points_2d, proj_matrices)
        inlier_mask = errs <= reproj_threshold_px
        count = int(inlier_mask.sum())
        if count < min_views:
            continue
        total_err = float(errs[inlier_mask].sum())
        if count > best_count or (count == best_count and total_err < best_err):
            best_count = count
            best_err = total_err
            best_inliers = np.flatnonzero(inlier_mask)

    return best_inliers


def triangulate_robust(
    keypoints_per_view: np.ndarray,
    scores_per_view: np.ndarray,
    proj_matrices: Sequence[np.ndarray],
    score_threshold: float = 0.3,
    min_views: int = 2,
    ransac: bool = False,
    reproj_threshold_px: float = 8.0,
) -> Pose3D:
    """Triangulate a single person's COCO keypoints with outlier rejection.

    Args:
        keypoints_per_view: (V, K, 2) undistorted pixel coords ``(u, v)``.
        scores_per_view: (V, K) per-view per-keypoint confidence in [0, 1].
        proj_matrices: sequence of ``V`` (3, 4) world->pixel matrices ``P``,
            same view order as the first axis of ``keypoints_per_view``.
        score_threshold: views with ``score < score_threshold`` are excluded
            for that keypoint.
        min_views: minimum kept views required to reconstruct a keypoint.
        ransac: if True, also reject views whose reprojection error exceeds
            ``reproj_threshold_px`` and re-triangulate from the inliers.
        reproj_threshold_px: inlier reprojection-error threshold (pixels).

    Returns:
        Pose3D with ``points`` (K, 3) world coords (NaN for unreconstructed
        joints), ``scores`` (K,) mean confidence of contributing views (0 for
        missing), ``valid`` (K,) bool, and ``source`` per-joint provenance tags
        (``'triangulation'`` or ``'missing'``).
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

    points = np.full((n_kpts, 3), np.nan, dtype=float)
    out_scores = np.zeros(n_kpts, dtype=float)
    valid = np.zeros(n_kpts, dtype=bool)
    source: list[str] = []

    for k in range(n_kpts):
        view_scores = scores[:, k]
        sel = np.flatnonzero(view_scores >= score_threshold)
        if sel.size < min_views:
            source.append("missing")
            continue

        sel_pts = kpts[sel, k, :]
        sel_proj = [proj[i] for i in sel]
        sel_scores = view_scores[sel]

        if ransac:
            # Consensus-based outlier rejection: pick the largest set of views
            # mutually consistent under reprojection (handles a high-score view
            # that is geometrically wrong, which score thresholding misses).
            inliers = _ransac_inliers(
                sel_pts, sel_proj, sel_scores, min_views, reproj_threshold_px
            )
            if inliers is None:
                source.append("missing")
                continue
            sel = sel[inliers]
            sel_pts = kpts[sel, k, :]
            sel_proj = [proj[i] for i in sel]
            sel_scores = view_scores[sel]

        point = triangulate_point_dlt(sel_pts, sel_proj, weights=sel_scores)
        if not np.isfinite(point).all():
            # Degenerate/ill-conditioned geometry (e.g. parallel rays) produced a
            # non-finite solution; drop the joint rather than emit inf/NaN.
            source.append("missing")
            continue

        points[k] = point
        out_scores[k] = float(np.mean(sel_scores))
        valid[k] = True
        source.append("triangulation")

    return Pose3D(points=points, scores=out_scores, valid=valid, source=source)
