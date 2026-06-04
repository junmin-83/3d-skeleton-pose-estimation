"""실제 공개 멀티뷰 데이터셋(CMU Panoptic Studio)으로 3D 포즈를 복원하는 데모.

사용하는 '실제 데이터':
  - CMU Panoptic의 **실제 31-HD-카메라 calibration** (K/distCoef/R/t, 단위 cm)
  - 데이터셋이 제공하는 **실제 캡처된 사람의 3D 포즈 GT** (hdPose3d, COCO19, 단위 cm)

흐름: 실제 GT 3D를 실제 카메라 calibration으로 각 뷰에 투영(렌즈 왜곡 포함) → (검출
오차 모사용) 노이즈 추가 → 우리 파이프라인(robust 삼각측량)으로 다시 3D 복원 →
GT 대비 복원 오차 + 카메라별 reprojection 오차로 검증한다.

[정직한 한계] Panoptic의 동기화 '이미지/영상'은 무료로 작게 받을 수 없어(영상이 수 GB,
test 샘플엔 이미지 미포함) 이 데모는 RTMPose를 '실제 픽셀'에 돌리는 대신 실제 calibration
으로 GT를 투영한 2D를 입력으로 쓴다. 실제 프레임이 있으면 아래 한 줄만 바꾸면 된다:
    keypoints[v] = RTMPoseDetector(...).detect_best(frame_v).keypoints   # GT 투영 대신

데이터 준비(이미 받았다면 생략):
    calibration_160906_band1.json, hdPose3d_stage1_coco19/body3DScene_*.json
    (mmpose tests/data/panoptic_body3d 에서 raw 다운로드)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.types import COCO_SKELETON, CameraParams  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

# CMU Panoptic COCO19 joint index -> our COCO17 index.
_PANOPTIC19_TO_COCO17 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
_CM_TO_M = 0.01


def load_panoptic_hd_cameras(calib_path: str) -> list[CameraParams]:
    """Panoptic calibration JSON -> our CameraParams (HD cams only, cm -> m)."""
    doc = json.load(open(calib_path, encoding="utf-8"))
    cams = []
    for c in doc["cameras"]:
        if c["type"] != "hd":
            continue
        cams.append(CameraParams(
            name=c["name"],
            K=np.asarray(c["K"], float),
            dist=np.asarray(c["distCoef"], float),          # [k1,k2,p1,p2,k3]
            R=np.asarray(c["R"], float),                    # world -> camera
            t=np.asarray(c["t"], float).reshape(3) * _CM_TO_M,  # cm -> m
            image_size=(int(c["resolution"][0]), int(c["resolution"][1])),
        ))
    return cams


def load_gt_coco17(gt_path: str, body_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load one body's GT, map COCO19 -> COCO17, cm -> m. Returns (17,3) m, (17,) conf."""
    doc = json.load(open(gt_path, encoding="utf-8"))
    joints19 = np.asarray(doc["bodies"][body_idx]["joints19"], float).reshape(19, 4)
    sel = joints19[_PANOPTIC19_TO_COCO17]
    return sel[:, :3] * _CM_TO_M, sel[:, 3]


def project(cam: CameraParams, pts_world: np.ndarray) -> np.ndarray:
    """Project world points (m) into a camera WITH lens distortion -> (N,2) px."""
    rvec, _ = cv2.Rodrigues(cam.R)
    proj, _ = cv2.projectPoints(pts_world.reshape(-1, 1, 3), rvec, cam.t.reshape(3, 1), cam.K, cam.dist)
    return proj.reshape(-1, 2)


def sees_person(cam: CameraParams, pts_world: np.ndarray) -> bool:
    """True if the body is in front of the camera and mostly inside the image."""
    cam_z = (pts_world @ cam.R.T + cam.t)[:, 2]
    if np.median(cam_z) <= 0:
        return False
    uv = project(cam, pts_world)
    w, h = cam.image_size
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return inside.mean() >= 0.8


def select_spread_views(cams: list[CameraParams], pts_world: np.ndarray, n: int) -> list[CameraParams]:
    """Greedy farthest-point pick of n in-frame cameras for a wide triangulation baseline."""
    valid = [c for c in cams if sees_person(c, pts_world)]
    if len(valid) < 2:
        raise RuntimeError(f"only {len(valid)} cameras see the person; need >= 2")
    centers = {c.name: c.center for c in valid}
    chosen = [valid[0]]
    while len(chosen) < min(n, len(valid)):
        best, best_d = None, -1.0
        for c in valid:
            if c in chosen:
                continue
            d = min(np.linalg.norm(centers[c.name] - centers[s.name]) for s in chosen)
            if d > best_d:
                best, best_d = c, d
        chosen.append(best)
    return chosen


def render(recon, gt_pts, out_path):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, j in COCO_SKELETON:
        if recon.valid[i] and recon.valid[j]:
            ax.plot(*[[recon.points[i, a], recon.points[j, a]] for a in range(3)], c="royalblue", lw=2)
    ax.scatter(*recon.points[recon.valid].T, c="royalblue", s=25, label="reconstructed")
    ax.scatter(*gt_pts.T, c="0.6", s=15, marker="x", label="GT (Panoptic)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("CMU Panoptic 3D pose reconstruction"); ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real CMU Panoptic multi-view 3D pose demo.")
    ap.add_argument("--calib", default="data/panoptic/160906_band1/calibration_160906_band1.json")
    ap.add_argument("--gt", default="data/panoptic/160906_band1/body3DScene_00000168.json")
    ap.add_argument("--body", type=int, default=0, help="which person (0..2) in the scene.")
    ap.add_argument("--num-views", type=int, default=4, help="number of real cameras to triangulate from.")
    ap.add_argument("--noise-px", type=float, default=3.0, help="2D detection-noise std (px) added to projections.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/panoptic_pose3d.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cams_all = load_panoptic_hd_cameras(args.calib)
    gt_pts, gt_conf = load_gt_coco17(args.gt, args.body)
    cams = select_spread_views(cams_all, gt_pts, args.num_views)
    print(f"[panoptic] HD cams loaded: {len(cams_all)} | using {len(cams)} views: {[c.name for c in cams]}")

    # Build per-view 2D by projecting the REAL GT through the REAL calibration (+ noise).
    K = gt_pts.shape[0]
    keypoints = np.zeros((len(cams), K, 2))
    scores = np.where(gt_conf > 0.1, 0.9, 0.0)[None, :].repeat(len(cams), axis=0)
    for v, cam in enumerate(cams):
        keypoints[v] = project(cam, gt_pts) + rng.normal(0, args.noise_px, size=(K, 2))

    config = {
        "triangulation": {"score_threshold": 0.3, "min_views": 2,
                          "ransac": {"enabled": True, "reproj_threshold_px": 20.0}},
        "depth_fusion": {"enabled": False},
        "smoothing": {"enabled": False},
    }
    recon = Pipeline(config, cams).process(keypoints, scores, depth_map=None)

    valid = recon.valid & (gt_conf > 0.1)
    err_mm = np.linalg.norm(recon.points[valid] - gt_pts[valid], axis=1) * 1000.0
    print(f"[panoptic] reconstructed {int(recon.valid.sum())}/{K} joints | "
          f"mean 3D error vs GT = {err_mm.mean():.1f} mm  (median {np.median(err_mm):.1f} mm)")
    for v, cam in enumerate(cams):
        re = np.linalg.norm(project(cam, recon.points[valid]) - keypoints[v][valid], axis=1)
        print(f"[panoptic]   view {cam.name}: reprojection RMS = {np.sqrt((re**2).mean()):.2f} px")

    render(recon, gt_pts[gt_conf > 0.1], args.out)
    print(f"[panoptic] 3D reconstruction vs GT -> {args.out}")


if __name__ == "__main__":
    main()
