"""Reprojection-error reporting — the calibration accuracy gate (SPEC 4-3).

Lengths are meters; pixels are ``(u, v)``. Extrinsics ``(R, t)`` map
WORLD -> CAMERA; ``reprojection_report`` projects world points through each
camera's ``P = K [R | t]`` and reports the per-camera pixel RMS.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.types import CameraParams


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec_or_R: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """Mean pixel RMS reprojection error via ``cv2.projectPoints``.

    Args:
        object_points: (N, 3) board points, meters.
        image_points: (N, 2) observed pixels ``(u, v)``.
        K: (3, 3) intrinsic.
        dist: (k,) distortion coefficients.
        rvec_or_R: (3,)/(3, 1) Rodrigues vector OR (3, 3) rotation matrix,
            board -> camera.
        tvec: (3,) translation board -> camera.

    Returns:
        RMS reprojection error in pixels (sqrt of mean squared L2 residual).
    """
    obj = np.asarray(object_points, np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_points, np.float64).reshape(-1, 2)
    rot = np.asarray(rvec_or_R, np.float64)
    rvec = cv2.Rodrigues(rot.reshape(3, 3))[0] if rot.size == 9 else rot.reshape(3, 1)
    K = np.asarray(K, np.float64).reshape(3, 3)
    dist = np.asarray(dist, np.float64).reshape(-1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)

    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    residuals = projected - img
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def reprojection_report(
    cameras: list[CameraParams],
    observations: dict[str, tuple[np.ndarray, np.ndarray]],
    verbose: bool = True,
) -> dict[str, float]:
    """Per-camera reprojection RMS using each camera's WORLD -> pixel ``P``.

    For each camera with an entry in ``observations`` (world points, observed
    pixels), the world points are projected through ``P = K [R | t]`` and the
    pixel RMS is computed. Distortion is applied via ``cv2.projectPoints`` when
    the camera has non-zero coefficients (real data); for synthetic, undistorted
    data the result matches the linear ``P`` projection.

    Args:
        cameras: list of calibrated cameras.
        observations: ``name -> (object_points_world (N, 3), image_points (N, 2))``.
        verbose: print per-camera and mean RMS.

    Returns:
        ``name -> rms`` plus a ``'mean'`` aggregate over reported cameras.
    """
    report: dict[str, float] = {}
    for cam in cameras:
        if cam.name not in observations:
            continue
        world_pts, image_pts = observations[cam.name]
        world_pts = np.asarray(world_pts, np.float64).reshape(-1, 3)
        image_pts = np.asarray(image_pts, np.float64).reshape(-1, 2)
        # World -> camera pose IS the extrinsic (R, t); reuse reprojection_error.
        rms = reprojection_error(world_pts, image_pts, cam.K, cam.dist, cam.R, cam.t)
        report[cam.name] = rms
        if verbose:
            print(f"[reprojection] {cam.name}: {rms:.6f} px")

    if report:
        mean_rms = float(np.mean(list(report.values())))
        report["mean"] = mean_rms
        if verbose:
            print(f"[reprojection] mean: {mean_rms:.6f} px")
    return report
