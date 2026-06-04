"""Common-world extrinsic calibration + CameraParams assembly.

Extrinsics (R, t) map WORLD -> CAMERA: X_cam = R @ X_world + t, with
P = K[R|t] taking a homogeneous world point to pixels. solvePnP /
estimate_board_pose return BOARD -> CAMERA poses; calibrate_extrinsics
re-expresses them as WORLD -> CAMERA, where the world is the reference camera
cam0 (R = I, t = 0) by default, or the board origin. Lengths in meters; pixels
are (u, v). See core/geometry.py and core/types.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.types import CameraParams, DepthCameraParams


def estimate_board_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the BOARD -> CAMERA pose of a single board view.

    Args:
        object_points: (N, 3) board-frame points, meters.
        image_points: (N, 2) pixel corners (u, v).
        K: (3, 3) intrinsic.
        dist: (k,) distortion coefficients.

    Returns:
        (R, t): (3, 3) rotation and (3,) translation, board -> camera
        (X_cam = R @ X_board + t).
    """
    obj = np.asarray(object_points, np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_points, np.float64).reshape(-1, 1, 2)
    K = np.asarray(K, np.float64).reshape(3, 3)
    dist = np.asarray(dist, np.float64).reshape(-1)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed to estimate the board pose")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def calibrate_extrinsics(
    board_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    reference: str = "cam0",
    world_frame: str = "reference_camera",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert per-camera BOARD -> CAMERA poses into WORLD -> CAMERA extrinsics.

    Each board_poses[name] = (R_i, t_i) maps board -> camera_i:
    X_cami = R_i X_board + t_i.

    world_frame == 'board_origin': world IS the board, so the extrinsics are
    the board poses unchanged.

    world_frame == 'reference_camera': world IS the reference camera frame. A
    world point relates to the board via the reference pose (R_ref, t_ref):
    X_board = R_ref^T (X_world - t_ref). Substituting into camera_i's board
    pose gives X_cami = R_i R_ref^T X_world + (t_i - R_i R_ref^T t_ref), i.e.
    R_world_to_cami = R_i @ R_ref.T and t = t_i - R_i @ R_ref.T @ t_ref. The
    reference camera itself becomes R = I, t = 0.

    Args:
        board_poses: name -> (R, t) board -> camera poses.
        reference: reference camera name (the world for 'reference_camera').
        world_frame: 'reference_camera' or 'board_origin'.

    Returns:
        name -> (R, t) world -> camera extrinsics.
    """
    if world_frame == "board_origin":
        return {
            name: (np.asarray(R, float).reshape(3, 3), np.asarray(t, float).reshape(3))
            for name, (R, t) in board_poses.items()
        }
    if world_frame != "reference_camera":
        raise ValueError(f"unknown world_frame: {world_frame!r}")
    if reference not in board_poses:
        raise KeyError(f"reference camera {reference!r} missing from board_poses")

    R_ref, t_ref = board_poses[reference]
    R_ref = np.asarray(R_ref, float).reshape(3, 3)
    t_ref = np.asarray(t_ref, float).reshape(3)

    extrinsics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (R_i, t_i) in board_poses.items():
        if name == reference:
            # World == reference camera frame, so its extrinsic is exactly
            # identity. Set it directly to avoid ~1e-16 residue from
            # R_ref @ R_ref.T.
            extrinsics[name] = (np.eye(3), np.zeros(3))
            continue
        R_i = np.asarray(R_i, float).reshape(3, 3)
        t_i = np.asarray(t_i, float).reshape(3)
        R_world = R_i @ R_ref.T
        t_world = t_i - R_world @ t_ref
        extrinsics[name] = (R_world, t_world)
    return extrinsics


def build_camera_params(
    name: str,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int],
    type: str = "rgb",
    **depth_kwargs: object,
) -> CameraParams:
    """Build a CameraParams, or DepthCameraParams when type == 'rgbd'.

    Args:
        name, K, dist, R, t, image_size: standard camera fields (see types).
        type: 'rgb' or 'rgbd'.
        **depth_kwargs: forwarded to DepthCameraParams for rgbd cameras
            (depth_K, depth_scale, depth_to_color_R, depth_to_color_t).
    """
    common = dict(
        name=name,
        K=K,
        dist=dist,
        R=R,
        t=t,
        image_size=(int(image_size[0]), int(image_size[1])),
    )
    if type == "rgbd":
        return DepthCameraParams(**common, **depth_kwargs)
    return CameraParams(**common)
