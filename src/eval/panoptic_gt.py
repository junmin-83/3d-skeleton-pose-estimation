"""CMU Panoptic 3D ground-truth loader (hdPose3d_stage1_coco19 -> COCO-17).

Panoptic GT lives in ``<seq>/hdPose3d_stage1_coco19/body3DScene_<frame>.json``.
Each body has a flat ``joints19`` list of 19*(x, y, z, confidence), in the
Panoptic world frame and **centimeters**. We convert to meters and remap the 19
OpenPose-ordered joints to the pipeline's COCO-17 order so MPJPE-family metrics
(src/eval/metrics.py) compare like-for-like.

Panoptic world vs the pipeline's world: both demos build their world from the
Panoptic calibration (load_panoptic_hd_cameras), so reconstructed poses already
live in the Panoptic world frame -> GT needs no extra rigid transform. The
calibration t is cm*0.01 (src/io/sources/panoptic.py); GT here uses the same
cm->m so the two share one metric world frame.

COCO19 joint order (OpenPose-based, per CMU panoptic-toolbox):
  0 Neck 1 Nose 2 BodyCenter(pelvis) 3 lShoulder 4 lElbow 5 lWrist
  6 lHip 7 lKnee 8 lAnkle 9 rShoulder 10 rElbow 11 rWrist
  12 rHip 13 rKnee 14 rAnkle 15 lEye 16 lEar 17 rEye 18 rEar
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from src.core.types import NUM_KEYPOINTS, Pose3D

_CM_TO_M = 0.01

# COCO-17 index -> COCO19(Panoptic) index. Order matches COCO_17_KEYPOINTS in
# src/core/types.py (0 nose, 1 left_eye, ... 16 right_ankle).
COCO17_TO_COCO19: tuple[int, ...] = (
    1,   # 0  nose          <- Nose
    15,  # 1  left_eye      <- lEye
    17,  # 2  right_eye     <- rEye
    16,  # 3  left_ear      <- lEar
    18,  # 4  right_ear     <- rEar
    3,   # 5  left_shoulder <- lShoulder
    9,   # 6  right_shoulder<- rShoulder
    4,   # 7  left_elbow    <- lElbow
    10,  # 8  right_elbow   <- rElbow
    5,   # 9  left_wrist    <- lWrist
    11,  # 10 right_wrist   <- rWrist
    6,   # 11 left_hip      <- lHip
    12,  # 12 right_hip     <- rHip
    7,   # 13 left_knee     <- lKnee
    13,  # 14 right_knee    <- rKnee
    8,   # 15 left_ankle    <- lAnkle
    14,  # 16 right_ankle   <- rAnkle
)

_BODYCENTER_COCO19 = 2  # Panoptic pelvis (mid-hip), for root-relative alignment


def coco19_to_coco17(joints19: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remap a (19, 4) [x,y,z,conf] cm array to COCO-17 world meters + validity.

    Args:
        joints19: (19, 4) Panoptic joints, centimeters; col 3 is confidence.

    Returns:
        points (17, 3) meters, valid (17,) bool (confidence > 0; Panoptic marks
        unreliable/occluded joints with non-positive confidence).
    """
    j = np.asarray(joints19, dtype=float).reshape(19, 4)
    idx = np.asarray(COCO17_TO_COCO19, dtype=int)
    points = j[idx, :3] * _CM_TO_M
    valid = j[idx, 3] > 0.0
    return points, valid


def pelvis_coco19(joints19: np.ndarray) -> np.ndarray:
    """Panoptic BodyCenter (pelvis) in meters, for GT root-relative alignment."""
    j = np.asarray(joints19, dtype=float).reshape(19, 4)
    return j[_BODYCENTER_COCO19, :3] * _CM_TO_M


def gt_pose_from_body(joints19: np.ndarray) -> Pose3D:
    """Build a COCO-17 Pose3D (world meters) from one Panoptic body's joints19.

    scores carry the Panoptic per-joint confidence (clipped to >=0); source is
    tagged 'gt' so downstream code can tell GT from reconstructions.
    """
    j = np.asarray(joints19, dtype=float).reshape(19, 4)
    points, valid = coco19_to_coco17(j)
    idx = np.asarray(COCO17_TO_COCO19, dtype=int)
    scores = np.clip(j[idx, 3], 0.0, None)
    return Pose3D(points=points, scores=scores, valid=valid,
                  source=["gt"] * NUM_KEYPOINTS)


def load_gt_frame(
    gt_dir: str,
    frame: int,
    body_id: int | None = None,
) -> Pose3D | None:
    """Load one frame's GT as a COCO-17 Pose3D, or None if absent/empty.

    Args:
        gt_dir: ``.../hdPose3d_stage1_coco19`` directory.
        frame: HD frame index; file is ``body3DScene_<frame:08d>.json``.
        body_id: select this Panoptic body id; None -> highest-confidence body
            (matches the single-person pipeline's best_person policy).

    Returns:
        Pose3D in COCO-17 world meters, or None when the file is missing or has
        no bodies.
    """
    path = os.path.join(gt_dir, f"body3DScene_{frame:08d}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    bodies = doc.get("bodies", [])
    if not bodies:
        return None

    if body_id is not None:
        match = [b for b in bodies if b.get("id") == body_id]
        if not match:
            return None
        body = match[0]
    elif len(bodies) == 1:
        body = bodies[0]
    else:
        # Highest mean joint confidence == the single-person target.
        def _mean_conf(b: dict) -> float:
            j = np.asarray(b["joints19"], dtype=float).reshape(19, 4)
            return float(np.clip(j[:, 3], 0.0, None).mean())

        body = max(bodies, key=_mean_conf)

    return gt_pose_from_body(body["joints19"])


def load_gt_raw_body(gt_dir: str, frame: int) -> np.ndarray | None:
    """Highest-confidence body's raw joints19 (19, 4) cm array, or None.

    Kept raw (not remapped) for callers needing the Panoptic BodyCenter pelvis or
    confidences, e.g. root-relative alignment (src/eval/runner.py).
    """
    path = os.path.join(gt_dir, f"body3DScene_{frame:08d}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    bodies = doc.get("bodies", [])
    if not bodies:
        return None

    def _mean_conf(b: dict) -> float:
        j = np.asarray(b["joints19"], dtype=float).reshape(19, 4)
        return float(np.clip(j[:, 3], 0.0, None).mean())

    body = max(bodies, key=_mean_conf)
    return np.asarray(body["joints19"], dtype=float).reshape(19, 4)


def available_gt_frames(gt_dir: str) -> list[int]:
    """Sorted HD frame indices that have a GT file in gt_dir."""
    frames = []
    for path in glob.glob(os.path.join(gt_dir, "body3DScene_*.json")):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            frames.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(frames)
