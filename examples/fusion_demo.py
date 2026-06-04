"""합성 데이터로 RGB 삼각측량 + RGB-D depth **융합('fused')**을 시각화하는 데모.

이 프로젝트의 간판 경로(다중 뷰 confidence-가중 삼각측량 + depth back-projection을
관절별로 융합)는 ``Pipeline.process``의 hybrid 분기이지만, 실영상 데모(②/③-A/③-B)는
각각 2D-only · depth-only · triangulation-only 서브경로만 시연한다. 공개 단일 데이터셋이
"캘리브레이션된 다중 RGB 뷰 + 정렬 depth"를 동시에 주지 않기 때문이다(TUM=단일 RGB-D,
Panoptic=멀티뷰지만 Kinect depth 미디코딩). 그래서 이 데모는 **합성**으로 그 셋업을
구성해 fused 경로를 눈으로 보여준다.

구성: 2 RGB + 1 RGB-D 카메라를 하나의 world 좌표계에 배치하고, 알려진 3D COCO-17
스켈레톤(팔 흔들기 + 좌우 sway)을 각 뷰에 투영해 2D + score를 만들고, RGB-D 카메라용
정렬 depth 맵을 만든 뒤 진짜 ``Pipeline``에 넣는다. 관절 출처를 색으로 구분:

  - **fused**(초록): 삼각측량 ✅ + depth ✅ → confidence 가중 평균
  - **depth**(파랑): RGB 두 뷰에서 가려진(score↓) 관절을 depth가 채움 — 오른손목
  - **triangulation**(빨강): depth 맵에 구멍(hole)이라 RGB 삼각측량만 — 왼발목

학습 없이, rtmlib/GPU/다운로드 없이 오프라인에서 결정론적으로 실행된다.

Usage::

    uv run python examples/fusion_demo.py --num-frames 80
    #   -> output/fusion_pose3d.mp4   ([cam0 2D | cam1 2D | Depth | 3D(출처색)])
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry import project_points  # noqa: E402
from src.core.types import NUM_KEYPOINTS, CameraParams, DepthCameraParams  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.render.skeleton_2d import draw_skeleton_2d, label_panel  # noqa: E402
from src.render.skeleton_3d import render_pose3d_by_source  # noqa: E402
from src.render.video_writer import LazyVideoWriter  # noqa: E402

PW, PH = 300, 240          # per-panel size
_W, _H = 640, 480          # synthetic image size
_K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
_ZERO5 = np.zeros(5)

OCCLUDED_IN_RGB = 10       # right_wrist: low RGB score in both views -> depth fills it
DEPTH_HOLE = 15            # left_ankle: punched out of depth map -> triangulation only

# Base COCO-17 pose in a person-local frame: x=right, y=up, z=toward camera (m).
_BASE = np.array([
    [0.000, 1.62, 0.05],   # 0  nose
    [0.035, 1.66, 0.06],   # 1  left_eye
    [-0.035, 1.66, 0.06],  # 2  right_eye
    [0.075, 1.63, 0.00],   # 3  left_ear
    [-0.075, 1.63, 0.00],  # 4  right_ear
    [0.190, 1.42, 0.00],   # 5  left_shoulder
    [-0.190, 1.42, 0.00],  # 6  right_shoulder
    [0.240, 1.15, 0.00],   # 7  left_elbow
    [-0.240, 1.15, 0.00],  # 8  right_elbow
    [0.260, 0.90, 0.00],   # 9  left_wrist
    [-0.260, 0.90, 0.00],  # 10 right_wrist
    [0.110, 0.92, 0.00],   # 11 left_hip
    [-0.110, 0.92, 0.00],  # 12 right_hip
    [0.120, 0.50, 0.02],   # 13 left_knee
    [-0.120, 0.50, 0.02],  # 14 right_knee
    [0.110, 0.08, 0.00],   # 15 left_ankle
    [-0.110, 0.08, 0.00],  # 16 right_ankle
])


def skeleton_world(frame: int, period: float = 40.0) -> np.ndarray:
    """Animated COCO-17 in world meters: right arm wave + slight body sway."""
    local = _BASE.copy()
    phase = 2.0 * np.pi * frame / period
    a = 0.5 * (1.0 + np.sin(phase))            # wave amount in [0, 1]
    local[8] += np.array([-0.04 * a, 0.30 * a, 0.0])    # right_elbow up
    local[10] += np.array([-0.10 * a, 0.85 * a, 0.10 * a])  # right_wrist up/forward
    sway_x = 0.05 * np.sin(0.5 * phase)
    world = np.empty_like(local)
    world[:, 0] = local[:, 0] + sway_x         # X right
    world[:, 1] = 0.85 - local[:, 1]           # Y down (image-down positive)
    world[:, 2] = 2.8 + local[:, 2]            # Z depth (in front of cameras)
    return world


def build_cameras() -> list[CameraParams]:
    """Two RGB views (0.6 m baseline) + one RGB-D view in a shared world frame."""
    return [
        CameraParams("cam0_rgb", _K, _ZERO5, np.eye(3), np.array([0.0, 0.0, 0.0]), (_W, _H)),
        CameraParams("cam1_rgb", _K, _ZERO5, np.eye(3), np.array([-0.6, 0.0, 0.0]), (_W, _H)),
        DepthCameraParams("cam2_rgbd", _K, _ZERO5, np.eye(3), np.array([-0.3, 0.0, 0.0]),
                          (_W, _H), depth_K=_K),
    ]


def make_depth_map(depth_cam: DepthCameraParams, world: np.ndarray) -> np.ndarray:
    """Aligned metric-depth map for the RGB-D view: stamp each joint's camera Z.

    A small disk is drawn so patch sampling finds it. ``DEPTH_HOLE`` is left
    empty (invalid) to force that joint onto the triangulation-only path.
    """
    depth_map = np.zeros((_H, _W), dtype=np.float32)
    cam_pts = world @ depth_cam.R.T + depth_cam.t          # world -> camera frame
    uv = project_points(depth_cam.P, world)
    for k in range(NUM_KEYPOINTS):
        if k == DEPTH_HOLE:
            continue
        u, v = int(round(uv[k, 0])), int(round(uv[k, 1]))
        if 0 <= u < _W and 0 <= v < _H:
            cv2.circle(depth_map, (u, v), 7, float(cam_pts[k, 2]), -1)
    return depth_map


def detect_synthetic(cameras: list[CameraParams], world: np.ndarray):
    """Project world -> per-view 2D + confidence, emulating an occluded RGB joint."""
    kpts = np.stack([project_points(c.P, world) for c in cameras])   # (3, K, 2)
    scores = np.full((len(cameras), NUM_KEYPOINTS), 0.9)
    scores[0, OCCLUDED_IN_RGB] = 0.1   # right wrist occluded in cam0 (RGB)
    scores[1, OCCLUDED_IN_RGB] = 0.1   # ... and cam1 (RGB); cam2 (depth) still sees it
    return kpts, scores


def depth_panel(depth_map: np.ndarray, kpts, scores, thr, dmin, dmax) -> np.ndarray:
    """Colour-mapped depth + the depth view's 2D skeleton."""
    valid = (depth_map > 0) & (depth_map >= dmin) & (depth_map <= dmax)
    vis = np.full((_H, _W, 3), 30, np.uint8)
    if valid.any():
        norm = np.zeros_like(depth_map)
        norm[valid] = (depth_map[valid] - dmin) / max(dmax - dmin, 1e-6)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cm[~valid] = (30, 30, 30)
        vis = cm
    panel = cv2.resize(vis, (PW, PH))
    return draw_skeleton_2d(panel, kpts, scores, thr, scale=(PW / _W, PH / _H))


def rgb_panel(kpts, scores, thr, title: str) -> np.ndarray:
    """Synthetic RGB frame (dark canvas) with the 2D skeleton overlay."""
    canvas = np.full((PH, PW, 3), 45, np.uint8)
    draw_skeleton_2d(canvas, kpts, scores, thr, scale=(PW / _W, PH / _H))
    return label_panel(canvas, title, PW, font_scale=0.4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic RGB+Depth fusion (hybrid) 3D demo.")
    ap.add_argument("--num-frames", type=int, default=80)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--depth-min", type=float, default=0.2)
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--video", default="output/fusion_pose3d.mp4")
    args = ap.parse_args()

    cameras = build_cameras()
    depth_cam = cameras[2]
    config = {
        "triangulation": {"score_threshold": 0.3, "min_views": 2,
                          "ransac": {"enabled": True, "reproj_threshold_px": 10.0}},
        "depth_fusion": {"enabled": True, "fill_missing": True, "patch_radius_px": 3,
                         "depth_min": args.depth_min, "depth_max": args.depth_max},
        "smoothing": {"enabled": True, "freq": args.fps, "min_cutoff": 1.0,
                      "beta": 0.01, "d_cutoff": 1.0},
    }
    pipe = Pipeline(config, cameras)

    fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    writer = LazyVideoWriter(args.video, args.fps)
    lims = None

    for f in range(args.num_frames):
        world = skeleton_world(f)
        kpts, scores = detect_synthetic(cameras, world)
        depth_map = make_depth_map(depth_cam, world)
        pose = pipe.process(kpts, scores, depth_map=depth_map, timestamp=f / args.fps)
        if lims is None and pose.valid.any():
            c = pose.points[pose.valid].mean(axis=0)
            lims = [(c[a] - 0.9, c[a] + 0.9) for a in range(3)]

        panels = [
            rgb_panel(kpts[0], scores[0], 0.3, "cam0 RGB+2D (wrist occluded)"),
            rgb_panel(kpts[1], scores[1], 0.3, "cam1 RGB+2D (wrist occluded)"),
            label_panel(depth_panel(depth_map, kpts[2], scores[2], 0.3, args.depth_min, args.depth_max),
                        "RGB-D depth + 2D", PW, font_scale=0.4),
            label_panel(render_pose3d_by_source(fig, ax, pose, lims or [(-1, 1)] * 3, (PW, PH),
                                                 view_init=(-75, -90)),
                        f"3D  fused=G depth=B tri=R  f{f}", PW, font_scale=0.4),
        ]
        writer.write(np.hstack(panels))

        if f % 10 == 0:
            src = pose.source
            counts = {s: src.count(s) for s in ("fused", "depth", "triangulation", "missing")}
            print(f"[fusion] frame {f}: fused={counts['fused']} depth={counts['depth']} "
                  f"tri={counts['triangulation']} missing={counts['missing']}")

    if writer.opened:
        writer.release()
        print(f"[fusion] result video ({args.num_frames} frames) -> {args.video}")
    plt.close(fig)


if __name__ == "__main__":
    main()
