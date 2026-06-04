"""Camera intrinsic calibration: checkerboard detection + cv2.calibrateCamera.

Lengths in meters; pixels are (u, v) (matches core/geometry.py). Intrinsic
accuracy is the pipeline bottleneck (SPEC 4-3), so check the returned
reprojection RMS.
"""

from __future__ import annotations

import cv2
import numpy as np

# cornerSubPix refinement criteria, shared across intrinsic calibration.
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
_SUBPIX_WIN = (11, 11)


def build_object_points(pattern_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    """Planar board points (Z = 0) for one view.

    Args:
        pattern_size: (cols, rows) of inner corners.
        square_size_m: square edge length, meters.

    Returns:
        (cols*rows, 3) float64 board-frame coords, row-major in (x, y).
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
    """Detect and sub-pixel-refine checkerboard corners across images.

    Args:
        images: grayscale or BGR np.ndarrays.
        pattern_size: (cols, rows) of inner corners.
        square_size_m: square edge length, meters.

    Returns:
        (object_points_list, image_points_list), one entry per image where the
        board was found. object_points are (N, 3) meters; image_points are
        (N, 2) pixel (u, v).
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
    """Estimate intrinsics via cv2.calibrateCamera.

    Args:
        object_points_list: per-view (N, 3) board points, meters.
        image_points_list: per-view (N, 2) pixel corners.
        image_size: (width, height) in pixels.

    Returns:
        (K, dist, rms): K is (3, 3), dist is (k,), rms is OpenCV's overall
        reprojection RMS (pixels).
    """
    object_points = [p.astype(np.float32).reshape(-1, 1, 3) for p in object_points_list]
    image_points = [p.astype(np.float32).reshape(-1, 1, 2) for p in image_points_list]
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, tuple(image_size), None, None
    )
    return np.asarray(K, float), np.asarray(dist, float).reshape(-1), float(rms)
