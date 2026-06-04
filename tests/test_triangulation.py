"""Synthetic-data triangulation tests (SPEC 7-2).

Known 3D world points are projected (noise-free) into three virtual cameras
with build_projection_matrix, then triangulated back. Recovery must be at
numerical precision. Also covers confidence weighting, robust outlier
rejection, and the min_views deferral case.

All units are meters; pixels are ``(u, v)``; ``P = K [R | t]`` maps world ->
pixel. The world frame coincides with the reference camera (cam0) frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from src.core.geometry import build_projection_matrix, project_points
from src.core.types import NUM_KEYPOINTS
from src.triangulation.dlt import triangulate_keypoints, triangulate_point_dlt
from src.triangulation.robust import triangulate_robust


def _make_cameras():
    """Three virtual cameras with distinct K, R (scipy), t (meters).

    cam0 is the reference (world == cam0 frame: R=I, t=0). cam1/cam2 are
    rotated and translated so all three see a person standing ~2-3 m away.
    Returns a list of (3, 4) projection matrices, world -> pixel.
    """
    K0 = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    K1 = np.array([[1000.0, 0.0, 620.0], [0.0, 1000.0, 380.0], [0.0, 0.0, 1.0]])
    K2 = np.array([[850.0, 0.0, 660.0], [0.0, 870.0, 350.0], [0.0, 0.0, 1.0]])

    # cam0 is the world reference frame.
    R0 = np.eye(3)
    t0 = np.zeros(3)
    # cam1: yaw +35 deg, shifted +1.5 m in x.
    R1 = Rotation.from_euler("xyz", [3.0, 35.0, -2.0], degrees=True).as_matrix()
    t1 = np.array([1.5, -0.1, 0.2])
    # cam2: yaw -40 deg, shifted -1.6 m in x.
    R2 = Rotation.from_euler("xyz", [-4.0, -40.0, 3.0], degrees=True).as_matrix()
    t2 = np.array([-1.6, 0.05, 0.3])

    return [
        build_projection_matrix(K0, R0, t0),
        build_projection_matrix(K1, R1, t1),
        build_projection_matrix(K2, R2, t2),
    ]


def _standing_skeleton():
    """17 plausible standing-person 3D points (meters) in the world frame.

    Roughly a 1 m-wide person at ~2.5 m depth (+z), head up (+y down image is
    handled by projection). Values are deterministic, not random.
    """
    z = 2.5  # depth from cam0, meters
    # (x, y) laid out as a coarse upright skeleton; z constant-ish.
    pts = np.array(
        [
            [0.00, 0.75, z],        # nose
            [-0.03, 0.78, z],       # left_eye
            [0.03, 0.78, z],        # right_eye
            [-0.07, 0.76, z],       # left_ear
            [0.07, 0.76, z],        # right_ear
            [-0.18, 0.55, z],       # left_shoulder
            [0.18, 0.55, z],        # right_shoulder
            [-0.22, 0.25, z],       # left_elbow
            [0.22, 0.25, z],        # right_elbow
            [-0.25, -0.05, z],      # left_wrist
            [0.25, -0.05, z],       # right_wrist
            [-0.12, -0.05, z],      # left_hip
            [0.12, -0.05, z],       # right_hip
            [-0.13, -0.45, z],      # left_knee
            [0.13, -0.45, z],       # right_knee
            [-0.14, -0.85, z],      # left_ankle
            [0.14, -0.85, z],       # right_ankle
        ],
        dtype=float,
    )
    # Add mild per-joint depth variation so the geometry is non-degenerate.
    pts[:, 2] += np.linspace(-0.1, 0.1, pts.shape[0])
    assert pts.shape == (NUM_KEYPOINTS, 3)
    return pts


def _project_all(proj_matrices, pts_world):
    """Project (K, 3) world points into each view -> (V, K, 2) pixels."""
    return np.stack([project_points(P, pts_world) for P in proj_matrices], axis=0)


def test_single_point_exact_recovery():
    """Noise-free triangulation of a single point recovers it exactly."""
    proj = _make_cameras()
    x_true = np.array([0.12, 0.30, 2.45])
    obs = np.stack([project_points(P, x_true) for P in proj], axis=0)
    x_hat = triangulate_point_dlt(obs, proj)
    np.testing.assert_allclose(x_hat, x_true, atol=1e-6)


def test_keypoints_exact_recovery():
    """Full 17-keypoint skeleton recovered to numerical precision (<1e-6 m)."""
    proj = _make_cameras()
    pts_world = _standing_skeleton()
    kpts = _project_all(proj, pts_world)                  # (V, K, 2)
    scores = np.ones((len(proj), NUM_KEYPOINTS))          # (V, K)

    points_3d, conf = triangulate_keypoints(kpts, scores, proj)

    np.testing.assert_allclose(points_3d, pts_world, atol=1e-6)
    np.testing.assert_allclose(conf, np.ones(NUM_KEYPOINTS), atol=1e-12)


def test_robust_exact_recovery_all_views():
    """robust path with all views above threshold matches the truth exactly."""
    proj = _make_cameras()
    pts_world = _standing_skeleton()
    kpts = _project_all(proj, pts_world)
    scores = np.full((len(proj), NUM_KEYPOINTS), 0.9)

    pose = triangulate_robust(kpts, scores, proj, score_threshold=0.3, min_views=2)

    assert pose.valid.all()
    assert pose.source == ["triangulation"] * NUM_KEYPOINTS
    np.testing.assert_allclose(pose.points, pts_world, atol=1e-6)


def test_weighting_improves_accuracy_with_noisy_view():
    """Down-weighting one noisy view yields a result closer to the truth."""
    proj = _make_cameras()
    x_true = np.array([0.10, 0.20, 2.55])
    obs = np.stack([project_points(P, x_true) for P in proj], axis=0)

    # Corrupt view index 2 with a sizeable pixel offset (noise on one view).
    obs_noisy = obs.copy()
    obs_noisy[2] += np.array([25.0, -18.0])

    # Equal weights: the noisy view drags the estimate.
    equal = triangulate_point_dlt(obs_noisy, proj, weights=np.ones(3))
    # Confidence weights: noisy view gets a low weight.
    weighted = triangulate_point_dlt(
        obs_noisy, proj, weights=np.array([1.0, 1.0, 0.02])
    )

    err_equal = np.linalg.norm(equal - x_true)
    err_weighted = np.linalg.norm(weighted - x_true)
    assert err_weighted < err_equal


def test_robust_outlier_rejection_recovers_within_mm():
    """A badly corrupted, low-score view is excluded; recovery within mm."""
    proj = _make_cameras()
    pts_world = _standing_skeleton()
    kpts = _project_all(proj, pts_world)

    # All views confident...
    scores = np.full((len(proj), NUM_KEYPOINTS), 0.9)
    # ...except view 2 on a single keypoint, which we corrupt badly and
    # mark below threshold so it is excluded.
    bad_kp = 9  # left_wrist
    kpts[2, bad_kp] += np.array([120.0, -90.0])
    scores[2, bad_kp] = 0.05  # below score_threshold

    pose = triangulate_robust(
        kpts, scores, proj, score_threshold=0.3, min_views=2
    )

    assert pose.valid[bad_kp]
    # Excluded the corrupted view -> remaining 2 views recover it within mm.
    err = np.linalg.norm(pose.points[bad_kp] - pts_world[bad_kp])
    assert err < 5e-3  # < 5 mm
    # Untouched keypoints still recover exactly.
    other = [k for k in range(NUM_KEYPOINTS) if k != bad_kp]
    np.testing.assert_allclose(pose.points[other], pts_world[other], atol=1e-6)


def test_ransac_rejects_outlier_view_when_score_high():
    """ransac=True rejects a view that is geometrically inconsistent.

    Here the corrupted view keeps a *high* score, so score thresholding alone
    would not drop it; reprojection-based RANSAC must.
    """
    proj = _make_cameras()
    pts_world = _standing_skeleton()
    kpts = _project_all(proj, pts_world)
    scores = np.full((len(proj), NUM_KEYPOINTS), 0.9)

    bad_kp = 10  # right_wrist
    kpts[2, bad_kp] += np.array([150.0, 110.0])  # large reprojection outlier

    pose = triangulate_robust(
        kpts, scores, proj,
        score_threshold=0.3, min_views=2,
        ransac=True, reproj_threshold_px=8.0,
    )

    assert pose.valid[bad_kp]
    err = np.linalg.norm(pose.points[bad_kp] - pts_world[bad_kp])
    assert err < 5e-3  # outlier view rejected -> within mm


def test_min_views_marks_keypoint_invalid():
    """A keypoint above threshold in only one view -> valid=False, missing."""
    proj = _make_cameras()
    pts_world = _standing_skeleton()
    kpts = _project_all(proj, pts_world)
    scores = np.full((len(proj), NUM_KEYPOINTS), 0.9)

    lonely = 16  # right_ankle visible (confident) in only cam0
    scores[1, lonely] = 0.1
    scores[2, lonely] = 0.1

    pose = triangulate_robust(
        kpts, scores, proj, score_threshold=0.3, min_views=2
    )

    assert not pose.valid[lonely]
    assert pose.source[lonely] == "missing"
    assert np.isnan(pose.points[lonely]).all()
    assert pose.scores[lonely] == 0.0
    # Every other keypoint is still valid.
    assert pose.valid.sum() == NUM_KEYPOINTS - 1


def test_dlt_zero_weights_returns_nan_and_robust_drops_joint():
    """All-zero weights carry no constraint -> NaN; robust marks the joint missing."""
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    p_a = build_projection_matrix(K, np.eye(3), np.zeros(3))
    p_b = build_projection_matrix(K, np.eye(3), np.array([0.5, 0.0, 0.0]))
    obs = np.array([[640.0, 360.0], [600.0, 360.0]])

    pt = triangulate_point_dlt(obs, [p_a, p_b], weights=np.array([0.0, 0.0]))
    assert not np.isfinite(pt).all()

    pose = triangulate_robust(obs[:, None, :], np.array([[0.0], [0.0]]),
                              [p_a, p_b], score_threshold=0.0, min_views=2)
    assert not pose.valid[0]
    assert pose.source[0] == "missing"
