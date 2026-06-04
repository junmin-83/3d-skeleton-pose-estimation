"""End-to-end pipeline integration tests on synthetic data.

Project a known 3D skeleton into the configured rig, run the full
triangulation -> depth-fusion -> smoothing chain, and confirm the original 3D is
recovered. Exercises the real production code path (no mocks) without rtmlib.
"""

import numpy as np

from src.pipeline import Pipeline
from src.synthetic import (
    make_synthetic_skeleton,
    synthesize_depth_map,
    synthesize_observations,
    synthesize_sequence,
)
from src.viz.visualize_3d import export_keypoints, load_keypoints

CONFIG = "config/cameras.yaml"


def test_pipeline_recovers_synthetic_skeleton():
    pipeline = Pipeline.from_config(CONFIG)
    skeleton = make_synthetic_skeleton()
    keypoints, scores = synthesize_observations(skeleton, pipeline.cameras)
    depth_cam = pipeline.cameras[pipeline.depth_idx]
    depth_map = synthesize_depth_map(skeleton, depth_cam)

    # First frame: One-Euro passes the first sample through unchanged.
    pose = pipeline.process(keypoints, scores, depth_map, timestamp=0.0)

    assert pose.points.shape == (17, 3)
    assert pose.valid.all(), "every keypoint should be reconstructed"
    np.testing.assert_allclose(pose.points, skeleton, atol=1e-3)


def test_pipeline_triangulation_only_recovers_skeleton():
    pipeline = Pipeline.from_config(CONFIG)
    pipeline.fusion_enabled = False
    pipeline.smoother = None
    skeleton = make_synthetic_skeleton()
    keypoints, scores = synthesize_observations(skeleton, pipeline.cameras)

    pose = pipeline.process(keypoints, scores, depth_map=None)

    assert pose.valid.all()
    np.testing.assert_allclose(pose.points, skeleton, atol=1e-6)


def test_pipeline_fusion_fills_missing_joint_from_depth():
    pipeline = Pipeline.from_config(CONFIG)
    pipeline.smoother = None
    skeleton = make_synthetic_skeleton()
    keypoints, scores = synthesize_observations(skeleton, pipeline.cameras)
    depth_cam = pipeline.cameras[pipeline.depth_idx]
    depth_map = synthesize_depth_map(skeleton, depth_cam)

    # Knock out one keypoint in both RGB views so triangulation fails for it,
    # leaving only the depth view -> depth fusion must fill it.
    rgb_views = [i for i in range(len(pipeline.cameras)) if i != pipeline.depth_idx]
    scores[rgb_views[0], 9] = 0.0
    scores[rgb_views[1], 9] = 0.0

    pose = pipeline.process(keypoints, scores, depth_map)

    assert pose.valid[9], "wrist should be filled from depth"
    assert pose.source[9] == "depth"
    np.testing.assert_allclose(pose.points[9], skeleton[9], atol=1e-2)


def test_pipeline_runs_multi_frame_sequence():
    pipeline = Pipeline.from_config(CONFIG)
    depth_cam = pipeline.cameras[pipeline.depth_idx]
    poses = []
    for frame_idx, skeleton in enumerate(synthesize_sequence(num_frames=15)):
        keypoints, scores = synthesize_observations(skeleton, pipeline.cameras)
        depth_map = synthesize_depth_map(skeleton, depth_cam)
        poses.append(pipeline.process(keypoints, scores, depth_map, timestamp=frame_idx / 30.0))

    assert len(poses) == 15
    assert all(p.valid.all() for p in poses)


def test_pipeline_export_roundtrip(tmp_path):
    pipeline = Pipeline.from_config(CONFIG)
    skeleton = make_synthetic_skeleton()
    keypoints, scores = synthesize_observations(skeleton, pipeline.cameras)
    depth_cam = pipeline.cameras[pipeline.depth_idx]
    depth_map = synthesize_depth_map(skeleton, depth_cam)
    pose = pipeline.process(keypoints, scores, depth_map)

    for fmt in ("json", "npy"):
        path = tmp_path / f"poses.{fmt}"
        export_keypoints([pose], str(path), fmt=fmt)
        loaded = load_keypoints(str(path), fmt=fmt)
        assert len(loaded) == 1
        np.testing.assert_allclose(loaded[0].points, pose.points, atol=1e-9)


def test_depth_weight_biases_fusion():
    """depth_weight controls how much the depth point pulls the fused result."""
    from src.core.types import Pose3D
    from src.fusion.depth_fusion import fuse

    tri = Pose3D(points=np.zeros((1, 3)), scores=np.array([1.0]),
                 valid=np.array([True]), source=["triangulation"])
    depth_pts = np.array([[1.0, 0.0, 0.0]])
    dv, ds = np.array([True]), np.array([1.0])

    ignore_depth = fuse(tri, depth_pts, dv, ds, depth_weight=0.0)
    trust_depth = fuse(tri, depth_pts, dv, ds, depth_weight=1e6)
    np.testing.assert_allclose(ignore_depth.points[0], [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(trust_depth.points[0], [1, 0, 0], atol=1e-3)


def test_depth_view_low_score_does_not_fill():
    """An occluded joint in the depth view must NOT be filled from depth."""
    pipeline = Pipeline.from_config(CONFIG)
    pipeline.smoother = None
    skeleton = make_synthetic_skeleton()
    keypoints, scores = synthesize_observations(skeleton, pipeline.cameras)
    depth_cam = pipeline.cameras[pipeline.depth_idx]
    depth_map = synthesize_depth_map(skeleton, depth_cam)

    rgb = [i for i in range(len(pipeline.cameras)) if i != pipeline.depth_idx]
    scores[rgb[0], 9] = 0.0
    scores[rgb[1], 9] = 0.0
    scores[pipeline.depth_idx, 9] = 0.0  # occluded in the depth view too

    pose = pipeline.process(keypoints, scores, depth_map)
    assert not pose.valid[9], "occluded joint must not be filled from low-score depth"
    assert pose.source[9] == "missing"


def test_dlt_degenerate_returns_nan_and_robust_marks_invalid():
    """Parallel rays -> point at infinity -> NaN; robust must drop the joint."""
    from src.core.geometry import build_projection_matrix
    from src.triangulation.dlt import triangulate_point_dlt
    from src.triangulation.robust import triangulate_robust

    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    p_a = build_projection_matrix(K, np.eye(3), np.zeros(3))
    p_b = build_projection_matrix(K, np.eye(3), np.array([1.0, 0.0, 0.0]))
    obs = np.array([[640.0, 360.0], [640.0, 360.0]])  # principal point in both

    pt = triangulate_point_dlt(obs, [p_a, p_b])
    assert not np.isfinite(pt).all(), "degenerate triangulation should be NaN"

    kpts = obs[:, None, :]                  # (V=2, K=1, 2)
    scs = np.array([[0.9], [0.9]])
    pose = triangulate_robust(kpts, scs, [p_a, p_b], min_views=2)
    assert not pose.valid[0]
    assert pose.source[0] == "missing"


def test_dlt_zero_weights_returns_nan_and_robust_drops_joint():
    """All-zero weights carry no constraint -> NaN, not a fabricated point."""
    from src.core.geometry import build_projection_matrix
    from src.triangulation.dlt import triangulate_point_dlt
    from src.triangulation.robust import triangulate_robust

    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    p_a = build_projection_matrix(K, np.eye(3), np.zeros(3))
    p_b = build_projection_matrix(K, np.eye(3), np.array([0.5, 0.0, 0.0]))
    obs = np.array([[640.0, 360.0], [600.0, 360.0]])

    pt = triangulate_point_dlt(obs, [p_a, p_b], weights=np.array([0.0, 0.0]))
    assert not np.isfinite(pt).all()

    # score_threshold=0 selects the zero-score views; the joint must still drop.
    kpts = obs[:, None, :]
    scs = np.array([[0.0], [0.0]])
    pose = triangulate_robust(kpts, scs, [p_a, p_b], score_threshold=0.0, min_views=2)
    assert not pose.valid[0]
    assert pose.source[0] == "missing"


def test_one_euro_rejects_nonpositive_cutoff():
    """Guard against divide-by-zero from a non-positive cutoff/frequency."""
    import pytest

    from src.smoothing.one_euro import OneEuroFilter

    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0)
    with pytest.raises(ValueError):
        OneEuroFilter(freq=0.0)
    with pytest.raises(ValueError):
        OneEuroFilter(d_cutoff=-1.0)
