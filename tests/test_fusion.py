"""Depth-fusion unit tests: back-projection recovery, sampling validity, fusion."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.core.geometry import (
    build_projection_matrix,
    project_points,
    world_to_camera,
)
from src.core.types import Pose3D
from src.fusion.depth_fusion import (
    back_project_depth_keypoints,
    fuse,
    sample_depth,
)
from src.io.depth_reader import DummyDepthSource


def _make_depth_camera(seed: int = 0):
    """A plausible depth camera with non-trivial pose (world -> camera)."""
    width, height = 1280, 720
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    R = Rotation.from_euler("xyz", [8.0, -20.0, 4.0], degrees=True).as_matrix()
    t = np.array([0.25, -0.05, 2.6])  # meters
    return width, height, K, R, t


def test_depth_back_projection_recovers_world_point():
    """A known world point in front of the depth camera is recovered to 1e-6 m."""
    width, height, K, R, t = _make_depth_camera()
    P = build_projection_matrix(K, R, t)

    point_world = np.array([0.12, -0.08, 0.05])
    uv = project_points(P, point_world)              # world -> colour pixel
    z_cam = world_to_camera(R, t, point_world)[2]    # camera-frame Z (meters)

    # Stamp the metric Z at that pixel in a dummy aligned depth map.
    src = DummyDepthSource(width, height, K, depth_scale=1.0, default_z=2.0)
    src.set_depth((uv[0], uv[1]), z_cam)
    depth_map, _ = src.read()

    # Sample with radius 0 so only the stamped pixel contributes.
    points_world, valid = back_project_depth_keypoints(
        uv.reshape(1, 2), depth_map, src.intrinsics(), R, t,
        patch_radius=0, depth_min=0.2, depth_max=6.0,
    )

    assert valid[0]
    np.testing.assert_allclose(points_world[0], point_world, atol=1e-6)


def test_sample_depth_rejects_zero_nan_and_out_of_range():
    height, width = 20, 20
    uv = np.array([10.0, 10.0])

    # All-zero patch -> invalid.
    zero_map = np.zeros((height, width), dtype=np.float32)
    z, valid = sample_depth(zero_map, uv, patch_radius=2)
    assert not valid
    assert np.isnan(z)

    # All-NaN patch -> invalid.
    nan_map = np.full((height, width), np.nan, dtype=np.float32)
    z, valid = sample_depth(nan_map, uv, patch_radius=2)
    assert not valid

    # Out-of-range (too far) -> invalid.
    far_map = np.full((height, width), 50.0, dtype=np.float32)
    z, valid = sample_depth(far_map, uv, patch_radius=2, depth_min=0.2, depth_max=6.0)
    assert not valid

    # Out-of-range (too near) -> invalid.
    near_map = np.full((height, width), 0.05, dtype=np.float32)
    z, valid = sample_depth(near_map, uv, patch_radius=2, depth_min=0.2, depth_max=6.0)
    assert not valid

    # A valid depth -> recovered.
    ok_map = np.full((height, width), 2.5, dtype=np.float32)
    z, valid = sample_depth(ok_map, uv, patch_radius=2, depth_min=0.2, depth_max=6.0)
    assert valid
    assert z == pytest.approx(2.5)


def test_fuse_equal_weights_gives_midpoint():
    point_a = np.array([1.0, 0.0, 3.0])  # triangulation
    point_b = np.array([1.2, 0.4, 3.2])  # depth

    tri = Pose3D(
        points=point_a.reshape(1, 3),
        scores=np.array([0.5]),
        valid=np.array([True]),
    )
    depth_points = point_b.reshape(1, 3)
    depth_valid = np.array([True])
    depth_scores = np.array([0.5])

    fused = fuse(tri, depth_points, depth_valid, depth_scores, depth_weight=1.0)

    np.testing.assert_allclose(fused.points[0], 0.5 * (point_a + point_b), atol=1e-12)
    assert fused.valid[0]
    assert fused.source[0] == "fused"


def test_fuse_unequal_weights_is_weighted():
    point_a = np.array([0.0, 0.0, 2.0])  # triangulation, weight 0.8
    point_b = np.array([1.0, 0.0, 2.0])  # depth, weight 0.2

    tri = Pose3D(
        points=point_a.reshape(1, 3),
        scores=np.array([0.8]),
        valid=np.array([True]),
    )
    fused = fuse(
        tri,
        point_b.reshape(1, 3),
        np.array([True]),
        np.array([0.2]),
        depth_weight=1.0,
    )

    expected = (0.8 * point_a + 0.2 * point_b) / (0.8 + 0.2)
    np.testing.assert_allclose(fused.points[0], expected, atol=1e-12)
    assert fused.source[0] == "fused"


def test_fuse_depth_weight_scales_contribution():
    point_a = np.array([0.0, 0.0, 2.0])  # triangulation
    point_b = np.array([1.0, 0.0, 2.0])  # depth

    tri = Pose3D(
        points=point_a.reshape(1, 3),
        scores=np.array([0.5]),
        valid=np.array([True]),
    )
    # depth_weight=3 -> effective depth weight 0.5*3 = 1.5 vs tri 0.5.
    fused = fuse(
        tri,
        point_b.reshape(1, 3),
        np.array([True]),
        np.array([0.5]),
        depth_weight=3.0,
    )

    w_tri, w_dep = 0.5, 0.5 * 3.0
    expected = (w_tri * point_a + w_dep * point_b) / (w_tri + w_dep)
    np.testing.assert_allclose(fused.points[0], expected, atol=1e-12)


def test_fuse_fills_missing_triangulation_from_depth():
    depth_point = np.array([0.3, 0.4, 2.5])

    tri = Pose3D(
        points=np.zeros((1, 3)),
        scores=np.array([0.0]),
        valid=np.array([False]),
    )
    fused = fuse(
        tri,
        depth_point.reshape(1, 3),
        np.array([True]),
        np.array([0.9]),
        fill_missing=True,
    )

    assert fused.valid[0]
    assert fused.source[0] == "depth"
    np.testing.assert_allclose(fused.points[0], depth_point, atol=1e-12)
    assert fused.scores[0] == pytest.approx(0.9)


def test_fuse_both_invalid_is_missing():
    tri = Pose3D(
        points=np.zeros((1, 3)),
        scores=np.array([0.0]),
        valid=np.array([False]),
    )
    fused = fuse(
        tri,
        np.full((1, 3), np.nan),
        np.array([False]),
        np.array([0.0]),
        fill_missing=True,
    )

    assert not fused.valid[0]
    assert fused.source[0] == "missing"


def test_fuse_does_not_mutate_input():
    point_a = np.array([1.0, 0.0, 3.0])
    tri = Pose3D(
        points=point_a.reshape(1, 3),
        scores=np.array([0.5]),
        valid=np.array([True]),
    )
    original_points = tri.points.copy()
    original_source = list(tri.source)

    fuse(tri, np.array([[1.2, 0.4, 3.2]]), np.array([True]), np.array([0.5]))

    np.testing.assert_array_equal(tri.points, original_points)
    assert tri.source == original_source
