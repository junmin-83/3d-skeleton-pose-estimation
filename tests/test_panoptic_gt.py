"""Panoptic GT loader tests: COCO19->COCO17 remap, cm->m, body selection.

Uses a hand-built joints19 array (no data files) plus a tmp JSON to exercise
load_gt_frame's file path and multi-body selection.
"""

import json

import numpy as np

from src.core.types import COCO_17_KEYPOINTS
from src.eval.panoptic_gt import (
    COCO17_TO_COCO19,
    coco19_to_coco17,
    gt_pose_from_body,
    load_gt_frame,
    pelvis_coco19,
)


def _synthetic_joints19():
    """joints19 where each joint's (x,y,z) encodes its own index for traceability.

    Joint i -> position (i, i, i) cm, confidence 0.9. Returns (19, 4).
    """
    j = np.zeros((19, 4), dtype=float)
    for i in range(19):
        j[i, :3] = [i, i, i]
        j[i, 3] = 0.9
    return j


def test_mapping_table_is_a_permutation_of_distinct_indices():
    assert len(COCO17_TO_COCO19) == 17
    assert len(set(COCO17_TO_COCO19)) == 17
    assert all(0 <= idx < 19 for idx in COCO17_TO_COCO19)


def test_remap_picks_right_joints_and_converts_cm_to_m():
    j = _synthetic_joints19()
    points, valid = coco19_to_coco17(j)
    assert points.shape == (17, 3)
    assert valid.all()
    # COCO-17 nose (idx 0) maps to COCO19 Nose (idx 1) -> (1,1,1) cm = (0.01,)*3 m
    assert np.allclose(points[0], [0.01, 0.01, 0.01])
    # right_ankle (idx 16) maps to COCO19 idx 14
    assert np.allclose(points[16], [0.14, 0.14, 0.14])


def test_known_landmark_mapping():
    # left_shoulder (COCO17 idx 5) must come from COCO19 lShoulder (idx 3).
    assert COCO17_TO_COCO19[COCO_17_KEYPOINTS.index("left_shoulder")] == 3
    assert COCO17_TO_COCO19[COCO_17_KEYPOINTS.index("right_hip")] == 12
    assert COCO17_TO_COCO19[COCO_17_KEYPOINTS.index("nose")] == 1


def test_nonpositive_confidence_marks_invalid():
    j = _synthetic_joints19()
    j[3, 3] = 0.0    # COCO19 lShoulder -> COCO17 left_shoulder invalid
    j[5, 3] = -1.0   # COCO19 lWrist    -> COCO17 left_wrist invalid
    _, valid = coco19_to_coco17(j)
    assert not valid[COCO_17_KEYPOINTS.index("left_shoulder")]
    assert not valid[COCO_17_KEYPOINTS.index("left_wrist")]
    assert valid[COCO_17_KEYPOINTS.index("nose")]


def test_pelvis_is_bodycenter():
    j = _synthetic_joints19()
    # BodyCenter is COCO19 idx 2 -> (2,2,2) cm = (0.02,)*3 m
    assert np.allclose(pelvis_coco19(j), [0.02, 0.02, 0.02])


def test_gt_pose_from_body_tags_source_gt():
    pose = gt_pose_from_body(_synthetic_joints19())
    assert pose.points.shape == (17, 3)
    assert pose.source == ["gt"] * 17
    assert pose.valid.all()


def test_load_gt_frame_selects_highest_confidence_body(tmp_path):
    gt_dir = tmp_path / "hdPose3d_stage1_coco19"
    gt_dir.mkdir()
    low = _synthetic_joints19()
    low[:, 3] = 0.2
    high = _synthetic_joints19()
    high[:, 3] = 0.8
    doc = {
        "version": 0.7,
        "univTime": 0.0,
        "bodies": [
            {"id": 0, "joints19": low.reshape(-1).tolist()},
            {"id": 1, "joints19": high.reshape(-1).tolist()},
        ],
    }
    (gt_dir / "body3DScene_00000005.json").write_text(json.dumps(doc))

    pose = load_gt_frame(str(gt_dir), 5)
    assert pose is not None
    # High-confidence body chosen -> scores ~0.8
    assert np.allclose(pose.scores, 0.8)
    # Body-id selection also works
    pose0 = load_gt_frame(str(gt_dir), 5, body_id=0)
    assert np.allclose(pose0.scores, 0.2)


def test_load_gt_frame_missing_returns_none(tmp_path):
    gt_dir = tmp_path / "hdPose3d_stage1_coco19"
    gt_dir.mkdir()
    assert load_gt_frame(str(gt_dir), 999) is None
