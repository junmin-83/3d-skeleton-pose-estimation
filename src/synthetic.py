"""Synthetic data generation for offline end-to-end pipeline validation.

Lets the whole pipeline (triangulation -> depth fusion -> smoothing -> export)
run and be verified without rtmlib, a GPU, or physical cameras: a known 3D
skeleton is projected into the configured cameras to produce 2D observations and
a depth map, so the recovered 3D can be compared against ground truth.

World/camera convention (see ``core/geometry.py``): +X right, +Y down,
+Z forward; world == reference camera (cam0). All lengths in meters.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.core.geometry import project_points, world_to_camera
from src.core.types import NUM_KEYPOINTS, CameraParams, DepthCameraParams

# COCO-17 offsets (meters) from the body centre for a frontal standing pose.
# +Y is DOWN, so the head has the most negative Y and ankles the most positive.
_SKELETON_OFFSETS: dict[int, tuple[float, float, float]] = {
    0: (0.00, -0.65, 0.00),    # nose
    1: (0.03, -0.70, 0.00), 2: (-0.03, -0.70, 0.00),   # eyes
    3: (0.07, -0.68, 0.00), 4: (-0.07, -0.68, 0.00),   # ears
    5: (0.18, -0.45, 0.00), 6: (-0.18, -0.45, 0.00),   # shoulders
    7: (0.26, -0.15, 0.05), 8: (-0.26, -0.15, 0.05),   # elbows (slightly forward)
    9: (0.30, 0.15, 0.10), 10: (-0.30, 0.15, 0.10),    # wrists (more forward)
    11: (0.12, 0.05, 0.00), 12: (-0.12, 0.05, 0.00),   # hips
    13: (0.13, 0.45, 0.00), 14: (-0.13, 0.45, 0.00),   # knees
    15: (0.14, 0.85, 0.00), 16: (-0.14, 0.85, 0.00),   # ankles
}


def make_synthetic_skeleton(center: tuple[float, float, float] = (0.0, 0.0, 2.5)) -> np.ndarray:
    """A plausible standing COCO-17 skeleton in the world frame (meters)."""
    cx, cy, cz = center
    out = np.zeros((NUM_KEYPOINTS, 3), dtype=float)
    for idx, (dx, dy, dz) in _SKELETON_OFFSETS.items():
        out[idx] = (cx + dx, cy + dy, cz + dz)
    return out


def synthesize_observations(
    skeleton_world: np.ndarray,
    cameras: list[CameraParams],
    base_score: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a 3D skeleton into every camera to get (V,K,2) px + (V,K) scores."""
    n_views = len(cameras)
    n_kpts = skeleton_world.shape[0]
    keypoints = np.zeros((n_views, n_kpts, 2), dtype=float)
    scores = np.full((n_views, n_kpts), float(base_score), dtype=float)
    for i, cam in enumerate(cameras):
        keypoints[i] = project_points(cam.P, skeleton_world)
    return keypoints, scores


def synthesize_depth_map(
    skeleton_world: np.ndarray,
    depth_cam: DepthCameraParams,
    stamp_radius: int = 3,
) -> np.ndarray:
    """Build a metric (meters) depth map for the depth camera.

    Background is 0 (invalid); a small ``stamp_radius`` block around each
    keypoint's colour pixel is set to that keypoint's camera-frame ``Z`` so a
    patch-median depth sample recovers the true depth.
    """
    width, height = depth_cam.image_size
    depth_map = np.zeros((height, width), dtype=np.float32)
    uv = project_points(depth_cam.P, skeleton_world)
    z_cam = world_to_camera(depth_cam.R, depth_cam.t, skeleton_world)[:, 2]
    for k in range(skeleton_world.shape[0]):
        cu, cv = int(round(uv[k, 0])), int(round(uv[k, 1]))
        for du in range(-stamp_radius, stamp_radius + 1):
            for dv in range(-stamp_radius, stamp_radius + 1):
                u, v = cu + du, cv + dv
                if 0 <= u < width and 0 <= v < height:
                    depth_map[v, u] = np.float32(z_cam[k])
    return depth_map


def synthesize_sequence(
    num_frames: int,
    amplitude: float = 0.1,
    jitter: float = 0.0,
    seed: int | None = 0,
) -> Iterator[np.ndarray]:
    """Yield a moving (and optionally jittered) skeleton, one per frame."""
    base = make_synthetic_skeleton()
    rng = np.random.default_rng(seed)
    for f in range(num_frames):
        offset = np.array([amplitude * np.sin(f * 0.2), 0.0, 0.0])
        skeleton = base + offset
        if jitter > 0.0:
            skeleton = skeleton + rng.normal(0.0, jitter, size=skeleton.shape)
        yield skeleton
