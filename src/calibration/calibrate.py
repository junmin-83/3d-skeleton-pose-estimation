"""Camera calibration: intrinsics, common-world extrinsics, reprojection error.

This module is the accuracy bottleneck of the whole pipeline (see SPEC 4-3), so
it always reports reprojection error.

Coordinate conventions (consistent with ``core/geometry.py`` and ``core/types``):
  - Lengths are **meters**; pixel coordinates are ``(u, v)`` order.
  - Extrinsics ``(R, t)`` map WORLD -> CAMERA:  ``X_cam = R @ X_world + t``.
  - ``P = K [R | t]`` maps a homogeneous world point to pixels.
  - The default world frame is the reference camera ``cam0`` (R = I, t = 0).
    Set ``world_frame='board_origin'`` to anchor the world to the board instead.

``estimate_board_pose`` and OpenCV return BOARD -> CAMERA poses; the helper
``calibrate_extrinsics`` re-expresses them as WORLD -> CAMERA extrinsics.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from src.core.types import CameraParams, DepthCameraParams

__all__ = [
    "build_object_points",
    "find_checkerboard_corners",
    "calibrate_intrinsics",
    "estimate_board_pose",
    "calibrate_extrinsics",
    "build_camera_params",
    "reprojection_error",
    "reprojection_report",
    "save_cameras_yaml",
    "load_cameras_yaml",
]

# cornerSubPix refinement criteria (shared by intrinsic calibration).
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
_SUBPIX_WIN = (11, 11)


def build_object_points(pattern_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    """Planar board points (Z = 0) for one view.

    Args:
        pattern_size: ``(cols, rows)`` of INNER corners.
        square_size_m: edge length of one square, meters.

    Returns:
        (cols*rows, 3) float64 board-frame coordinates, row-major in ``(x, y)``.
    """
    cols, rows = pattern_size
    obj = np.zeros((rows * cols, 3), dtype=np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= float(square_size_m)
    return obj


def find_checkerboard_corners(
    images: list[np.ndarray],
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect + sub-pixel-refine checkerboard corners across images.

    Args:
        images: list of images (grayscale or BGR ``np.ndarray``).
        pattern_size: ``(cols, rows)`` of inner corners.
        square_size_m: square edge length, meters.

    Returns:
        ``(object_points_list, image_points_list)`` containing one entry per
        image where the board was found. ``object_points`` are (N, 3) meters;
        ``image_points`` are (N, 2) pixel ``(u, v)``.
    """
    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    obj_template = build_object_points(pattern_size, square_size_m)

    for image in images:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, _SUBPIX_WIN, (-1, -1), _SUBPIX_CRITERIA
        )
        object_points_list.append(obj_template.copy())
        image_points_list.append(corners.reshape(-1, 2).astype(np.float64))

    return object_points_list, image_points_list


def calibrate_intrinsics(
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate intrinsics via ``cv2.calibrateCamera``.

    Args:
        object_points_list: per-view (N, 3) board points, meters.
        image_points_list: per-view (N, 2) pixel corners.
        image_size: ``(width, height)`` in pixels.

    Returns:
        ``(K, dist, rms)`` where ``K`` is (3, 3), ``dist`` is (k,), and ``rms``
        is the OpenCV overall reprojection RMS (pixels).
    """
    object_points = [p.astype(np.float32).reshape(-1, 1, 3) for p in object_points_list]
    image_points = [p.astype(np.float32).reshape(-1, 1, 2) for p in image_points_list]
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, tuple(image_size), None, None
    )
    return np.asarray(K, float), np.asarray(dist, float).reshape(-1), float(rms)


def estimate_board_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the BOARD -> CAMERA pose of a single board view.

    Args:
        object_points: (N, 3) board-frame points, meters.
        image_points: (N, 2) pixel corners ``(u, v)``.
        K: (3, 3) intrinsic.
        dist: (k,) distortion coefficients.

    Returns:
        ``(R, t)``: (3, 3) rotation and (3,) translation mapping board -> camera
        (``X_cam = R @ X_board + t``).
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

    Each ``board_poses[name] = (R_i, t_i)`` maps board -> camera_i:
    ``X_cami = R_i X_board + t_i``.

    If ``world_frame == 'board_origin'`` the world IS the board, so the
    extrinsics are the board poses unchanged.

    If ``world_frame == 'reference_camera'`` the world IS the reference camera
    frame. A world point expressed in the reference camera frame relates to the
    board via the reference pose ``(R_ref, t_ref)``:
    ``X_board = R_ref^T (X_world - t_ref)``. Substituting into camera_i's
    board pose gives
    ``X_cami = R_i R_ref^T X_world + (t_i - R_i R_ref^T t_ref)``, hence
    ``R_world_to_cami = R_i @ R_ref.T`` and ``t = t_i - R_i @ R_ref.T @ t_ref``.
    The reference camera itself becomes ``R = I, t = 0``.

    Args:
        board_poses: ``name -> (R, t)`` board -> camera poses.
        reference: name of the reference camera (used as world for
            ``reference_camera``).
        world_frame: ``'reference_camera'`` or ``'board_origin'``.

    Returns:
        ``name -> (R, t)`` world -> camera extrinsics.
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
            # world == reference camera frame, so its extrinsic is exactly
            # identity (avoids ~1e-16 residue from R_ref @ R_ref.T).
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
    """Construct a ``CameraParams`` (or ``DepthCameraParams`` for ``rgbd``).

    Args:
        name, K, dist, R, t, image_size: standard camera fields (see types).
        type: ``'rgb'`` or ``'rgbd'``.
        **depth_kwargs: forwarded to ``DepthCameraParams`` for ``rgbd`` cameras
            (``depth_K``, ``depth_scale``, ``depth_to_color_R``,
            ``depth_to_color_t``).
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


def _camera_to_dict(cam: CameraParams) -> dict:
    """Serialize one camera to the ``config/cameras.yaml`` schema."""
    is_rgbd = isinstance(cam, DepthCameraParams)
    entry: dict = {
        "name": cam.name,
        "type": "rgbd" if is_rgbd else "rgb",
        "K": cam.K.tolist(),
        "dist": cam.dist.tolist(),
        "R": cam.R.tolist(),
        "t": cam.t.tolist(),
        "image_size": [int(cam.image_size[0]), int(cam.image_size[1])],
        "source": getattr(cam, "source", None),
    }
    if is_rgbd:
        entry["depth_K"] = np.asarray(cam.depth_K, float).reshape(3, 3).tolist()
        entry["depth_scale"] = float(cam.depth_scale)
        d2c_R = cam.depth_to_color_R if cam.depth_to_color_R is not None else np.eye(3)
        d2c_t = cam.depth_to_color_t if cam.depth_to_color_t is not None else np.zeros(3)
        entry["depth_to_color_R"] = np.asarray(d2c_R, float).reshape(3, 3).tolist()
        entry["depth_to_color_t"] = np.asarray(d2c_t, float).reshape(3).tolist()
    return entry


def save_cameras_yaml(
    cameras: list[CameraParams],
    path: str | Path,
    units: str = "meter",
    world_frame: str = "reference_camera",
    reference: str = "cam0",
) -> None:
    """Write cameras to ``path`` matching the ``config/cameras.yaml`` schema.

    Top-level keys are ``units``, ``world`` and ``cameras``. Each camera carries
    ``name, type, K, dist, R, t, image_size, source`` and, for ``rgbd`` cameras,
    the depth fields (``depth_K, depth_scale, depth_to_color_R,
    depth_to_color_t``).
    """
    doc = {
        "units": {"length": units},
        "world": {"frame": world_frame, "reference_camera": reference},
        "cameras": [_camera_to_dict(cam) for cam in cameras],
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=None)


def load_cameras_yaml(path: str | Path) -> list[CameraParams]:
    """Load cameras from a ``config/cameras.yaml``-schema file.

    Returns:
        list of ``CameraParams`` (``rgbd`` cameras become ``DepthCameraParams``
        with their depth fields restored).
    """
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    cameras: list[CameraParams] = []
    for entry in doc.get("cameras", []):
        cam_type = entry.get("type", "rgb")
        image_size = tuple(int(v) for v in entry["image_size"])
        if cam_type == "rgbd":
            cam: CameraParams = DepthCameraParams(
                name=entry["name"],
                K=np.asarray(entry["K"], float),
                dist=np.asarray(entry["dist"], float),
                R=np.asarray(entry["R"], float),
                t=np.asarray(entry["t"], float),
                image_size=image_size,
                depth_K=np.asarray(entry["depth_K"], float),
                depth_scale=float(entry["depth_scale"]),
                depth_to_color_R=np.asarray(entry["depth_to_color_R"], float),
                depth_to_color_t=np.asarray(entry["depth_to_color_t"], float),
            )
        else:
            cam = CameraParams(
                name=entry["name"],
                K=np.asarray(entry["K"], float),
                dist=np.asarray(entry["dist"], float),
                R=np.asarray(entry["R"], float),
                t=np.asarray(entry["t"], float),
                image_size=image_size,
            )
        # ``source`` is not a dataclass field; attach it for round-trip fidelity.
        cam.source = entry.get("source")
        cameras.append(cam)
    return cameras
