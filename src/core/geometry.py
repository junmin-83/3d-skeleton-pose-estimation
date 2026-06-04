"""Geometric primitives for the multi-view 3D pose pipeline.

Conventions (whole project): lengths in meters, pixels as (u, v).
  - World: one fixed right-handed frame, by default the reference camera (cam0).
  - Extrinsics (R, t) map world -> camera: X_cam = R @ X_world + t.
  - Pinhole projection (undistort first): x_pix ~ K @ X_cam, P = K @ [R | t].
  - Depth back-projection needs metric depth: Z (meters) along the optical axis
    at pixel (u, v), i.e. depth already aligned to colour as RGB-D SDKs provide.
"""

from __future__ import annotations

import numpy as np


def build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return the (3, 4) projection matrix P = K [R | t], world -> pixel."""
    K = np.asarray(K, float).reshape(3, 3)
    R = np.asarray(R, float).reshape(3, 3)
    t = np.asarray(t, float).reshape(3, 1)
    return K @ np.hstack([R, t])


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Append a 1 to each row: (N, D) -> (N, D+1)."""
    points = np.atleast_2d(np.asarray(points, float))
    return np.hstack([points, np.ones((points.shape[0], 1))])


def project_points(P: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """Project world point(s) to pixels through P.

    Args:
        P: (3, 4) projection matrix.
        points_world: (3,) or (N, 3) world coords, meters.

    Returns:
        (2,) or (N, 2) pixel coords (u, v).
    """
    P = np.asarray(P, float).reshape(3, 4)
    pts = np.asarray(points_world, float)
    single = pts.ndim == 1
    hom = to_homogeneous(pts)                # (N, 4)
    proj = hom @ P.T                         # (N, 3)
    uv = proj[:, :2] / proj[:, 2:3]
    return uv[0] if single else uv


def world_to_camera(R: np.ndarray, t: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """Map world point(s) into the camera frame: X_cam = R X_world + t."""
    R = np.asarray(R, float).reshape(3, 3)
    t = np.asarray(t, float).reshape(1, 3)
    pts = np.asarray(points_world, float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    out = pts @ R.T + t
    return out[0] if single else out


def camera_to_world(R: np.ndarray, t: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    """Inverse of world_to_camera: X_world = R^T (X_cam - t)."""
    R = np.asarray(R, float).reshape(3, 3)
    t = np.asarray(t, float).reshape(1, 3)
    pts = np.asarray(points_cam, float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    out = (pts - t) @ R
    return out[0] if single else out


def back_project_pixels(
    K: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    R: np.ndarray | None = None,
    t: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project pixel(s) + metric depth to 3D.

    depth is Z in meters along the optical axis (aligned depth). Result is in the
    camera frame when R/t are omitted, else the world frame.

    Args:
        K: (3, 3) intrinsic of the stream the pixels came from.
        uv: (2,) or (N, 2) pixel coords (u, v).
        depth: scalar or (N,) metric depth, meters.
        R, t: optional extrinsics (world -> camera); pass both to get world coords.

    Returns:
        (3,) or (N, 3) coords, meters.
    """
    if (R is None) != (t is None):
        raise ValueError(
            "back_project_pixels: provide both R and t (-> world frame) or "
            "neither (-> camera frame); got exactly one."
        )
    K = np.asarray(K, float).reshape(3, 3)
    uv_arr = np.asarray(uv, float)
    single = uv_arr.ndim == 1
    uv_arr = np.atleast_2d(uv_arr)                  # (N, 2)
    depth_arr = np.atleast_1d(np.asarray(depth, float))  # (N,)

    pix_h = to_homogeneous(uv_arr)                  # (N, 3), trailing 1
    rays = pix_h @ np.linalg.inv(K).T               # (N, 3) with z == 1
    x_cam = rays * depth_arr[:, None]               # scale so component z == depth

    if R is None or t is None:
        return x_cam[0] if single else x_cam

    x_world = camera_to_world(R, t, x_cam)
    return x_world[0] if single else x_world


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera centre in world coords: -R^T t (meters)."""
    R = np.asarray(R, float).reshape(3, 3)
    t = np.asarray(t, float).reshape(3)
    return -R.T @ t
