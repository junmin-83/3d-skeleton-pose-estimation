"""Calibration tests on synthetic data (no camera / network required).

A planar checkerboard is projected with a known intrinsic/extrinsic via
``cv2.projectPoints`` (zero distortion). The tests check:
  - zero-error reprojection on exact synthetic data,
  - ``solvePnP`` recovers the known board pose,
  - ``calibrate_extrinsics`` yields R=I,t=0 for the reference camera and maps a
    second camera correctly (verified through ``build_projection_matrix`` +
    ``project_points``),
  - YAML save/load round-trips intrinsics, extrinsics, P, and depth fields.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.calibration.camera_io import load_cameras_yaml, save_cameras_yaml
from src.calibration.extrinsics import (
    build_camera_params,
    calibrate_extrinsics,
    estimate_board_pose,
)
from src.calibration.intrinsics import build_object_points
from src.calibration.reprojection import reprojection_error
from src.core.geometry import build_projection_matrix, project_points
from src.core.types import DepthCameraParams

PATTERN_SIZE = (7, 6)        # (cols, rows) inner corners
SQUARE_SIZE_M = 0.03         # 3 cm squares
K_TRUE = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
ZERO_DIST = np.zeros(5)


def _project(object_points, R, t, K=K_TRUE, dist=ZERO_DIST):
    """Project board points (board->camera R,t) to pixels via cv2.projectPoints."""
    import cv2

    rvec = cv2.Rodrigues(np.asarray(R, float))[0]
    tvec = np.asarray(t, float).reshape(3, 1)
    pts, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, K, dist
    )
    return pts.reshape(-1, 2)


def _board_pose(seed):
    """A known board->camera pose placing the board well in front of the camera."""
    R = Rotation.from_euler("xyz", [5.0, -10.0, 3.0], degrees=True).as_matrix()
    rng = np.random.default_rng(seed)
    t = np.array([0.05, -0.02, 0.8]) + rng.uniform(-0.01, 0.01, size=3)
    return R, t


def test_reprojection_error_zero_on_exact_data():
    obj = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)
    R, t = _board_pose(0)
    img = _project(obj, R, t)
    rms = reprojection_error(obj, img, K_TRUE, ZERO_DIST, R, t)
    assert rms < 1e-6, f"expected ~0 px, got {rms}"


def test_solvepnp_recovers_board_pose():
    obj = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)
    R_true, t_true = _board_pose(1)
    img = _project(obj, R_true, t_true)

    R_est, t_est = estimate_board_pose(obj, img, K_TRUE, ZERO_DIST)

    # Rotation difference as geodesic angle (radians).
    angle = np.linalg.norm(Rotation.from_matrix(R_est @ R_true.T).as_rotvec())
    assert angle < 1e-3, f"rotation error {angle} rad"
    np.testing.assert_allclose(t_est, t_true, atol=1e-4)


def test_extrinsics_reference_is_identity_and_second_camera_maps_correctly():
    obj = build_object_points(PATTERN_SIZE, SQUARE_SIZE_M)
    R0, t0 = _board_pose(2)        # board -> cam0 (reference)
    R1 = Rotation.from_euler("xyz", [2.0, 30.0, -4.0], degrees=True).as_matrix()
    t1 = np.array([0.6, 0.01, 0.85])  # board -> cam1

    board_poses = {"cam0": (R0, t0), "cam1": (R1, t1)}
    ext = calibrate_extrinsics(board_poses, reference="cam0", world_frame="reference_camera")

    # Reference camera extrinsic must be R=I, t=0.
    np.testing.assert_allclose(ext["cam0"][0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(ext["cam0"][1], np.zeros(3), atol=1e-12)

    # A world point = a board point expressed in cam0 frame (X_world = R0 X_board + t0).
    # Projecting it with cam1's world->camera P must equal the direct board->cam1 pixel.
    R1_w, t1_w = ext["cam1"]
    P1 = K_TRUE @ build_projection_matrix(np.eye(3), R1_w, t1_w)  # K @ [R|t] = full P

    board_pt = obj[10]                       # some board corner
    x_world = R0 @ board_pt + t0             # board point in cam0 (== world) frame
    uv_world = project_points(P1, x_world)   # world -> cam1 pixel
    uv_direct = _project(board_pt.reshape(1, 3), R1, t1)[0]  # board -> cam1 pixel
    np.testing.assert_allclose(uv_world, uv_direct, atol=1e-9)


def test_extrinsics_board_origin_passthrough():
    R0, t0 = _board_pose(3)
    R1 = Rotation.from_euler("xyz", [1.0, 5.0, 2.0], degrees=True).as_matrix()
    t1 = np.array([0.3, 0.0, 0.9])
    board_poses = {"cam0": (R0, t0), "cam1": (R1, t1)}
    ext = calibrate_extrinsics(board_poses, world_frame="board_origin")
    np.testing.assert_allclose(ext["cam0"][0], R0, atol=1e-12)
    np.testing.assert_allclose(ext["cam0"][1], t0, atol=1e-12)
    np.testing.assert_allclose(ext["cam1"][0], R1, atol=1e-12)
    np.testing.assert_allclose(ext["cam1"][1], t1, atol=1e-12)


def test_yaml_roundtrip_two_rgb_one_rgbd(tmp_path):
    R1 = Rotation.from_euler("xyz", [3.0, 15.0, -2.0], degrees=True).as_matrix()
    R2 = Rotation.from_euler("xyz", [-4.0, -20.0, 1.0], degrees=True).as_matrix()
    cams = [
        build_camera_params(
            "cam0", K_TRUE, ZERO_DIST, np.eye(3), np.zeros(3), (1280, 720), type="rgb"
        ),
        build_camera_params(
            "cam1", K_TRUE, ZERO_DIST, R1, np.array([0.6, 0.0, 0.0]), (1280, 720), type="rgb"
        ),
        build_camera_params(
            "cam2",
            K_TRUE,
            np.array([0.01, -0.02, 0.0, 0.0, 0.001]),
            R2,
            np.array([-0.6, 0.0, 0.1]),
            (1280, 720),
            type="rgbd",
            depth_K=K_TRUE,
            depth_scale=0.001,
            depth_to_color_R=np.eye(3),
            depth_to_color_t=np.array([0.015, 0.0, 0.0]),
        ),
    ]
    # Preserve source like the real config schema.
    for cam, src in zip(cams, [0, 1, 2]):
        cam.source = src

    path = tmp_path / "cameras.yaml"
    save_cameras_yaml(cams, path, units="meter", world_frame="reference_camera", reference="cam0")
    loaded = load_cameras_yaml(path)

    assert [c.name for c in loaded] == ["cam0", "cam1", "cam2"]
    for orig, got in zip(cams, loaded):
        np.testing.assert_allclose(got.K, orig.K)
        np.testing.assert_allclose(got.dist, orig.dist)
        np.testing.assert_allclose(got.R, orig.R)
        np.testing.assert_allclose(got.t, orig.t)
        assert tuple(got.image_size) == tuple(orig.image_size)
        np.testing.assert_allclose(got.P, orig.P)
        assert got.source == orig.source

    # The rgbd camera must round-trip as DepthCameraParams with its depth fields.
    rgbd = loaded[2]
    assert isinstance(rgbd, DepthCameraParams)
    assert not isinstance(loaded[0], DepthCameraParams)
    np.testing.assert_allclose(rgbd.depth_K, K_TRUE)
    assert rgbd.depth_scale == pytest.approx(0.001)
    np.testing.assert_allclose(rgbd.depth_to_color_R, np.eye(3))
    np.testing.assert_allclose(rgbd.depth_to_color_t, np.array([0.015, 0.0, 0.0]))
