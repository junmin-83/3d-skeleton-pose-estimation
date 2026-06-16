"""Reusable Panoptic evaluation core (driver + experiment scripts share this).

evaluate() runs one configuration over a frame range and returns per-frame metric
rows plus a summary, in either input mode (EVALUATION_PLAN §3-1):

  oracle : project GT 3D into each camera as GT 2D, then run the geometry. No
           rtmlib/video. Optional Gaussian pixel noise stresses 2D error.
  real   : run RTMPose on HD frames (needs rtmlib + .mp4), end-to-end system.

Calibration sensitivity (EVALUATION_PLAN C2): oracle 2D is always generated with
the TRUE calibration (what the sensor sees), while the pipeline reconstructs with
optionally PERTURBED cameras (rot_noise_deg / trans_noise_mm), so the metric
degradation isolates the system's sensitivity to calibration error.

Distances returned in millimeters (report units); pixel errors in pixels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.core.types import CameraParams
from src.eval.metrics import (
    auc3d,
    mpjpe,
    pa_mpjpe,
    pck3d,
    root_relative_mpjpe,
    valid_rate,
)
from src.eval.panoptic_gt import load_gt_frame, load_gt_raw_body, pelvis_coco19
from src.io.sources.panoptic import load_panoptic_hd_cameras
from src.pipeline import Pipeline

GT_SUBDIR = "hdPose3d_stage1_coco19"
METRIC_KEYS = (
    "abs_mpjpe_mm", "rrel_mpjpe_mm", "pa_mpjpe_mm",
    "pck_50mm", "pck_100mm", "auc", "valid_rate", "reproj_rmse_px",
)


def load_cameras(seq_dir: str, cam_names: list[str], views: int | None = None):
    """Selected HD cameras for a sequence, optionally truncated to `views`."""
    seq = Path(seq_dir)
    calib = next(seq.glob("calibration_*.json"))
    cams_all = {c.name: c for c in load_panoptic_hd_cameras(str(calib))}
    names = list(cam_names) if views is None else list(cam_names)[:views]
    return [cams_all[n] for n in names], names


def perturb_cameras(cams, rot_noise_deg: float, trans_noise_mm: float, rng):
    """Cameras with Gaussian extrinsic noise (deg on rotation, mm on translation).

    Returns the cameras unchanged when both noises are zero, so the no-noise path
    is exact (and the oracle round-trip stays at 0 mm).
    """
    if rot_noise_deg <= 0 and trans_noise_mm <= 0:
        return cams
    out = []
    for c in cams:
        d_rot = (Rotation.from_euler("xyz", rng.normal(scale=rot_noise_deg, size=3),
                                     degrees=True).as_matrix()
                 if rot_noise_deg > 0 else np.eye(3))
        d_t = rng.normal(scale=trans_noise_mm * 1e-3, size=3) if trans_noise_mm > 0 else np.zeros(3)
        out.append(CameraParams(name=c.name, K=c.K, dist=c.dist,
                                R=d_rot @ c.R, t=c.t + d_t, image_size=c.image_size))
    return out


def project_distorted(cam, points_world: np.ndarray) -> np.ndarray:
    """Project world points to pixels WITH lens distortion (the detector's space).

    The pipeline undistorts before triangulation, so oracle 2D and reprojection
    error must live in the distorted pixel space; a pinhole projection here would
    be wrongly 'undistorted' downstream (verified: keeps oracle round-trip at 0).
    """
    obj = np.asarray(points_world, float).reshape(-1, 1, 3)
    rvec = cv2.Rodrigues(cam.R)[0]
    uv, _ = cv2.projectPoints(obj, rvec, cam.t.reshape(3, 1), cam.K, cam.dist)
    return uv.reshape(-1, 2)


def make_pipeline(cams, mode: str, ransac: bool, smoothing: bool, fps: float) -> Pipeline:
    config = {
        "triangulation": {
            "score_threshold": 0.0 if mode == "oracle" else 0.4,
            "min_views": 2,
            "ransac": {"enabled": ransac, "reproj_threshold_px": 15.0},
        },
        "depth_fusion": {"enabled": False},
        "smoothing": {"enabled": smoothing, "freq": fps,
                      "min_cutoff": 1.0, "beta": 0.01, "d_cutoff": 1.0},
    }
    return Pipeline(config, cams)


def _oracle_2d(cams, gt_points_m, valid, noise_px, rng):
    v = len(cams)
    kpts = np.zeros((v, 17, 2))
    scores = np.zeros((v, 17))
    for i, cam in enumerate(cams):
        uv = project_distorted(cam, gt_points_m)
        if noise_px > 0:
            uv = uv + rng.normal(scale=noise_px, size=uv.shape)
        kpts[i] = uv
        scores[i] = valid.astype(float)
    return kpts, scores


def _real_2d(detector, frames):
    v = len(frames)
    kpts = np.zeros((v, 17, 2))
    scores = np.zeros((v, 17))
    for i, img in enumerate(frames):
        pose2d = detector.detect_best(img)
        kpts[i], scores[i] = pose2d.keypoints, pose2d.scores
    return kpts, scores


def _reproj_rmse(cams, pose, kpts_obs, valid) -> float:
    m = valid & pose.valid
    if not m.any():
        return float("nan")
    errs = []
    for i, cam in enumerate(cams):
        uv_pred = project_distorted(cam, pose.points[m])
        errs.append(np.sqrt(np.mean(np.sum((uv_pred - kpts_obs[i][m]) ** 2, axis=1))))
    return float(np.mean(errs))


def evaluate(
    seq_dir: str,
    cam_names: list[str],
    start: int,
    num_frames: int,
    mode: str = "oracle",
    views: int | None = None,
    pixel_noise: float = 0.0,
    ransac: bool = True,
    smoothing: bool = True,
    rot_noise_deg: float = 0.0,
    trans_noise_mm: float = 0.0,
    fps: float = 30.0,
    gt_offset: int = 0,
    seed: int = 0,
    detector=None,
    frame_iter=None,
) -> tuple[dict, list[dict]]:
    """Evaluate one configuration; returns (summary, per-frame rows).

    summary[k] = (mean, std, n) over non-NaN frames for each metric key.
    Reconstruction uses cameras perturbed by rot/trans noise (calibration
    sensitivity); oracle 2D and reprojection use the TRUE cameras.
    """
    true_cams, names = load_cameras(seq_dir, cam_names, views)
    rng = np.random.default_rng(seed)
    recon_cams = perturb_cameras(true_cams, rot_noise_deg, trans_noise_mm, rng)
    pipe = make_pipeline(recon_cams, mode, ransac, smoothing, fps)
    gt_dir = str(Path(seq_dir) / GT_SUBDIR)

    if mode == "real" and frame_iter is None:
        from src.io.sources.panoptic import iter_panoptic_hd_frames
        frame_iter = iter_panoptic_hd_frames(Path(seq_dir), names, start, num_frames)

    rows: list[dict] = []
    for f in range(num_frames):
        hd_frame = start + f
        gt_pose = load_gt_frame(gt_dir, hd_frame + gt_offset)
        raw = load_gt_raw_body(gt_dir, hd_frame + gt_offset)
        if gt_pose is None or raw is None:
            if mode == "real":
                next(frame_iter, None)
            continue

        if mode == "oracle":
            kpts, scores = _oracle_2d(true_cams, gt_pose.points, gt_pose.valid, pixel_noise, rng)
        else:
            frames = next(frame_iter, None)
            if frames is None:
                break
            kpts, scores = _real_2d(detector, frames)

        pose = pipe.process(kpts, scores, depth_map=None, timestamp=hd_frame / fps)
        gt_root = pelvis_coco19(raw)
        rows.append({
            "frame": hd_frame,
            "abs_mpjpe_mm": 1000 * mpjpe(pose.points, gt_pose.points, pose.valid, gt_pose.valid),
            "rrel_mpjpe_mm": 1000 * root_relative_mpjpe(
                pose.points, gt_pose.points, pose.valid, gt_pose.valid, gt_root=gt_root),
            "pa_mpjpe_mm": 1000 * pa_mpjpe(pose.points, gt_pose.points, pose.valid, gt_pose.valid),
            "pck_50mm": pck3d(pose.points, gt_pose.points, 0.05, pose.valid, gt_pose.valid),
            "pck_100mm": pck3d(pose.points, gt_pose.points, 0.10, pose.valid, gt_pose.valid),
            "auc": auc3d(pose.points, gt_pose.points, None, pose.valid, gt_pose.valid),
            "valid_rate": valid_rate(pose.valid, gt_pose.valid),
            "reproj_rmse_px": _reproj_rmse(true_cams, pose, kpts, gt_pose.valid),
        })

    if mode == "real" and frame_iter is not None:
        for _ in frame_iter:
            pass

    summary = {k: _agg(rows, k) for k in METRIC_KEYS}
    return summary, rows


def _agg(rows, key):
    vals = np.array([r[key] for r in rows if not np.isnan(r[key])], float)
    if vals.size == 0:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(vals)), float(np.std(vals)), int(vals.size))
