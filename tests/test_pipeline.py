"""Integration tests for src/pipeline.py (Pipeline.process strategy selection).

Synthetic, offline, no rtmlib/GPU: project known 3D points into camera views and
(for the depth paths) stamp their metric Z into an aligned depth map, then check
``Pipeline.process`` recovers the 3D pose. Covers the three reconstruction
strategies — multi-view triangulation, depth-only back-projection, and the
triangulation+depth fusion hybrid — plus low-confidence gating.
"""

from __future__ import annotations

import numpy as np

from src.core.geometry import project_points
from src.core.types import NUM_KEYPOINTS, CameraParams, DepthCameraParams
from src.pipeline import Pipeline

_K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
_IMG = (640, 480)
_ZERO_DIST = np.zeros(5)


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _world_from_pixels(pixels_uv: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Inverse-project (u, v, Z) -> camera-frame XYZ (== world when R=I, t=0)."""
    fx, fy, cx, cy = _K[0, 0], _K[1, 1], _K[0, 2], _K[1, 2]
    pts = np.zeros((len(zs), 3))
    for i, ((u, v), z) in enumerate(zip(pixels_uv, zs)):
        pts[i] = [(u - cx) / fx * z, (v - cy) / fy * z, z]
    return pts


def _synthetic_pose(n: int = NUM_KEYPOINTS):
    """n distinct in-bounds integer pixels + varied depths, with world points."""
    us = np.linspace(120, 500, n).round()
    vs = np.linspace(120, 380, n).round()
    pixels = np.stack([us, vs], axis=1)
    zs = np.linspace(1.5, 3.0, n)
    world = _world_from_pixels(pixels, zs)
    return world, pixels, zs


def _rgb_cam(name: str, R: np.ndarray, t: np.ndarray) -> CameraParams:
    return CameraParams(name=name, K=_K, dist=_ZERO_DIST, R=R, t=t, image_size=_IMG)


def _depth_cam(name: str, R: np.ndarray, t: np.ndarray) -> DepthCameraParams:
    return DepthCameraParams(name=name, K=_K, dist=_ZERO_DIST, R=R, t=t,
                             image_size=_IMG, depth_K=_K, depth_scale=1.0)


def _depth_map_at(pixels_uv: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Stamp metric Z at each (u, v) pixel into an otherwise-zero depth map."""
    w, h = _IMG
    dm = np.zeros((h, w), dtype=np.float32)
    for (u, v), z in zip(pixels_uv, zs):
        dm[int(round(v)), int(round(u))] = np.float32(z)
    return dm


# ---------------------------------------------------------------------------
# Multi-view triangulation
# ---------------------------------------------------------------------------

class TestTriangulationStrategy:
    def test_two_view_recovers_world_points(self):
        world, _, _ = _synthetic_pose()
        cams = [_rgb_cam("cam0", np.eye(3), np.zeros(3)),
                _rgb_cam("cam1", np.eye(3), np.array([0.5, 0.0, 0.0]))]
        kpts = np.stack([project_points(c.P, world) for c in cams])  # (2, K, 2)
        scores = np.full((2, NUM_KEYPOINTS), 0.9)
        config = {"triangulation": {"score_threshold": 0.3, "min_views": 2},
                  "depth_fusion": {"enabled": False},
                  "smoothing": {"enabled": False}}
        result = Pipeline(config, cams).process(kpts, scores, depth_map=None)
        assert result.valid.all()
        np.testing.assert_allclose(result.points, world, atol=1e-6)


# ---------------------------------------------------------------------------
# Depth-only back-projection (single RGB-D view — the rgbd demo's path)
# ---------------------------------------------------------------------------

class TestDepthOnlyStrategy:
    def test_single_rgbd_view_back_projects(self):
        world, pixels, zs = _synthetic_pose()
        cam = _depth_cam("cam0", np.eye(3), np.zeros(3))  # world == camera
        kpts = pixels.reshape(1, NUM_KEYPOINTS, 2)
        scores = np.full((1, NUM_KEYPOINTS), 0.9)
        depth_map = _depth_map_at(pixels, zs)
        config = {"triangulation": {"score_threshold": 0.3, "min_views": 2},
                  "depth_fusion": {"enabled": True, "fill_missing": True,
                                   "patch_radius_px": 0, "depth_min": 0.2, "depth_max": 6.0},
                  "smoothing": {"enabled": False}}
        result = Pipeline(config, [cam]).process(kpts, scores, depth_map=depth_map)
        assert result.valid.all()
        np.testing.assert_allclose(result.points, world, atol=1e-5)


# ---------------------------------------------------------------------------
# Hybrid: triangulation + depth fusion (both estimate the same point)
# ---------------------------------------------------------------------------

class TestHybridFusionStrategy:
    def test_triangulation_plus_depth_agree(self):
        world, pixels, zs = _synthetic_pose()
        cam0 = _rgb_cam("cam0", np.eye(3), np.zeros(3))
        cam1 = _depth_cam("cam1", np.eye(3), np.array([0.15, 0.0, 0.0]))
        cams = [cam0, cam1]
        kpts = np.stack([project_points(c.P, world) for c in cams])  # (2, K, 2)
        scores = np.full((2, NUM_KEYPOINTS), 0.9)
        # cam1 is a pure-x translation of cam0, so its camera-frame Z == world Z;
        # stamp that at cam1's projected pixels.
        pix1 = project_points(cam1.P, world)
        depth_map = _depth_map_at(pix1, zs)
        config = {"triangulation": {"score_threshold": 0.3, "min_views": 2},
                  "depth_fusion": {"enabled": True, "fill_missing": True,
                                   "patch_radius_px": 0, "depth_min": 0.2, "depth_max": 6.0},
                  "smoothing": {"enabled": False}}
        result = Pipeline(config, cams).process(kpts, scores, depth_map=depth_map)
        assert result.valid.all()
        np.testing.assert_allclose(result.points, world, atol=1e-3)


# ---------------------------------------------------------------------------
# Confidence gating
# ---------------------------------------------------------------------------

class TestConfidenceGating:
    def test_subthreshold_joint_is_invalid(self):
        world, _, _ = _synthetic_pose()
        cams = [_rgb_cam("cam0", np.eye(3), np.zeros(3)),
                _rgb_cam("cam1", np.eye(3), np.array([0.5, 0.0, 0.0]))]
        kpts = np.stack([project_points(c.P, world) for c in cams])
        scores = np.full((2, NUM_KEYPOINTS), 0.9)
        scores[:, 5] = 0.1  # joint 5 below threshold in both views
        config = {"triangulation": {"score_threshold": 0.3, "min_views": 2},
                  "depth_fusion": {"enabled": False},
                  "smoothing": {"enabled": False}}
        result = Pipeline(config, cams).process(kpts, scores, depth_map=None)
        assert not result.valid[5]
        assert result.valid[0]
