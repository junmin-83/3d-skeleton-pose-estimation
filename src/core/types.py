"""Shared data types and the fixed COCO-17 keypoint layout.

Units & frames (project-wide, see also ``geometry.py``):
  - All lengths are **meters**; all pixel coordinates are ``(u, v)`` order.
  - 3D points live in a single **world** frame (default: the reference camera
    cam0 frame). Camera extrinsics ``(R, t)`` map world -> camera.

The COCO-17 index order below is *fixed* and identical for every view, which
is what guarantees cross-view keypoint correspondence for triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- COCO-17 keypoint layout (fixed index order, shared by every view) ------
COCO_17_KEYPOINTS: tuple[str, ...] = (
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
)

NUM_KEYPOINTS: int = len(COCO_17_KEYPOINTS)

# Bone connectivity (0-based keypoint indices) for visualization only.
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (5, 7), (7, 9),            # left arm
    (6, 8), (8, 10),           # right arm
    (5, 6),                    # shoulders
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),        # left leg
    (12, 14), (14, 16),        # right leg
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (0, 5), (0, 6),            # neck-ish links
)


@dataclass
class CameraParams:
    """Intrinsics + extrinsics for one camera.

    Extrinsics map WORLD -> CAMERA:  ``X_cam = R @ X_world + t``.
    The projection matrix ``P = K @ [R | t]`` maps a homogeneous world point to
    pixels. Distortion must be removed from pixels *before* using ``P`` (the
    pinhole model in ``P`` does not account for lens distortion).
    """

    name: str
    K: np.ndarray            # (3, 3) intrinsic matrix, pixels
    dist: np.ndarray         # (k,) OpenCV distortion coeffs (k in {4,5,8,12,14})
    R: np.ndarray            # (3, 3) rotation, world -> camera
    t: np.ndarray            # (3,) translation, world -> camera, meters
    image_size: tuple[int, int]  # (width, height) in pixels

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=float).reshape(-1)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)

    @property
    def P(self) -> np.ndarray:
        """(3, 4) projection matrix ``K [R | t]`` (world -> pixel)."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def center(self) -> np.ndarray:
        """Camera centre in **world** coordinates: ``-R^T t`` (meters)."""
        return -self.R.T @ self.t


@dataclass
class DepthCameraParams(CameraParams):
    """An RGB-D camera whose depth stream is aligned to its colour stream.

    ``K``/``dist``/``R``/``t`` describe the colour stream (used for 2D pose and
    triangulation like any RGB view). The depth-specific fields below are used
    by depth back-projection. When depth is aligned to colour, ``depth_K`` ==
    ``K``; the fields are kept separate so an un-aligned setup can be supported.
    """

    depth_K: np.ndarray | None = None        # (3, 3) depth intrinsic, pixels
    depth_scale: float = 1.0                  # raw depth unit -> meters
    depth_to_color_R: np.ndarray | None = None  # (3, 3) depth -> color rotation
    depth_to_color_t: np.ndarray | None = None  # (3,) depth -> color translation, m

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.depth_K is None:
            self.depth_K = self.K.copy()
        else:
            self.depth_K = np.asarray(self.depth_K, dtype=float).reshape(3, 3)


@dataclass
class Pose2D:
    """One person's 2D pose in one view."""

    keypoints: np.ndarray  # (K, 2) pixel coords (u, v)
    scores: np.ndarray     # (K,) confidence in [0, 1]

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=float).reshape(-1, 2)
        self.scores = np.asarray(self.scores, dtype=float).reshape(-1)


@dataclass
class Pose3D:
    """One person's reconstructed 3D pose in the world frame (meters)."""

    points: np.ndarray   # (K, 3) world coords, meters
    scores: np.ndarray   # (K,) fused confidence in [0, 1]
    valid: np.ndarray    # (K,) bool; False == joint could not be reconstructed
    source: list[str] = field(default_factory=list)  # per-joint provenance tag

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        self.scores = np.asarray(self.scores, dtype=float).reshape(-1)
        self.valid = np.asarray(self.valid, dtype=bool).reshape(-1)
