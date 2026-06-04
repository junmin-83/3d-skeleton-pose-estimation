"""Geometry round-trips: projection, back-projection, frame transforms."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.core.geometry import (
    back_project_pixels,
    build_projection_matrix,
    camera_center,
    camera_to_world,
    project_points,
    world_to_camera,
)


def _make_camera(seed: int = 0):
    """A pinhole camera with a non-trivial pose (world -> camera)."""
    rng = np.random.default_rng(seed)
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    R = Rotation.from_euler("xyz", [10.0, -25.0, 5.0], degrees=True).as_matrix()
    t = np.array([0.3, -0.1, 2.5])  # meters
    return K, R, t, rng


def test_projection_matrix_shape():
    K, R, t, _ = _make_camera()
    P = build_projection_matrix(K, R, t)
    assert P.shape == (3, 4)
    # P = K [R|t] => left 3x3 block is K @ R
    np.testing.assert_allclose(P[:, :3], K @ R, rtol=0, atol=1e-12)


def test_world_camera_roundtrip():
    K, R, t, rng = _make_camera(1)
    pts = rng.uniform(-1.0, 1.0, size=(20, 3))
    back = camera_to_world(R, t, world_to_camera(R, t, pts))
    np.testing.assert_allclose(back, pts, atol=1e-12)


def test_project_then_backproject_recovers_point():
    """Project world points, then back-project with their true depth, recover them."""
    K, R, t, rng = _make_camera(2)
    P = build_projection_matrix(K, R, t)
    pts_world = rng.uniform(-0.8, 0.8, size=(30, 3)) + np.array([0.0, 0.0, 0.0])

    uv = project_points(P, pts_world)            # world -> pixel
    depth = world_to_camera(R, t, pts_world)[:, 2]  # Z in camera frame
    recovered = back_project_pixels(K, uv, depth, R, t)

    np.testing.assert_allclose(recovered, pts_world, atol=1e-9)


def test_single_point_api():
    K, R, t, _ = _make_camera(3)
    P = build_projection_matrix(K, R, t)
    X = np.array([0.1, -0.2, 0.05])
    uv = project_points(P, X)
    assert uv.shape == (2,)
    Z = world_to_camera(R, t, X)[2]
    rec = back_project_pixels(K, uv, Z, R, t)
    assert rec.shape == (3,)
    np.testing.assert_allclose(rec, X, atol=1e-9)


def test_camera_center():
    K, R, t, _ = _make_camera(4)
    c = camera_center(R, t)
    # The camera centre sits at the camera-frame origin.
    np.testing.assert_allclose(world_to_camera(R, t, c), np.zeros(3), atol=1e-12)


def test_back_project_partial_extrinsics_raises():
    """Passing exactly one of R/t is a frame-consistency footgun, so it errors."""
    K, R, t, _ = _make_camera(5)
    uv = np.array([640.0, 360.0])
    with pytest.raises(ValueError):
        back_project_pixels(K, uv, 2.0, R=R, t=None)
    with pytest.raises(ValueError):
        back_project_pixels(K, uv, 2.0, R=None, t=t)
    # Both omitted: camera-frame output, no error.
    assert back_project_pixels(K, uv, 2.0).shape == (3,)
