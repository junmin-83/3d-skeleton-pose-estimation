"""2 RGB + 1 RGB-D 하이브리드 리그의 3D pose 결과영상(MP4) 데모.

당신의 실제 리그 구조(RGB 2대 + Depth 1대)를 그대로 모사한다:
프레임마다 알려진 3D 스켈레톤을 config의 3개 카메라로 투영해 합성 2D를 만들고,
Depth 카메라용 깊이맵을 합성한 뒤, 전체 파이프라인(삼각측량 -> depth fusion ->
One-Euro 스무딩)을 실제로 실행해 3D를 복원한다. 결과는 4분할 화면으로 합성한다:

    [ cam0 (RGB1) 2D ] [ cam1 (RGB2) 2D ]
    [ cam2 (Depth) 2D ] [ 3D 복원       ]

[한계] 입력 2D/depth는 '합성'이다(공개로 받을 수 있는 2RGB+1Depth 동기화+calibration
영상이 없어서). 실제 영상이 준비되면 각 RGB 뷰는 RTMPoseDetector, depth 뷰는 실제
depth 맵으로 바꾸면 동일 파이프라인이 그대로 동작한다.

Usage::

    uv run python examples/hybrid_3cam_demo.py --frames 120
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

from src.core.types import COCO_SKELETON, DepthCameraParams  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.synthetic import (  # noqa: E402
    synthesize_depth_map,
    synthesize_observations,
    synthesize_sequence,
)
from src.viz.visualize_3d import plot_skeleton_3d  # noqa: E402

PW, PH = 480, 360                       # per-panel size
_XLIM, _YLIM, _ZLIM = (-0.9, 0.9), (-1.0, 1.0), (1.8, 3.2)


def draw_skeleton_2d(canvas, kpts, scores, cam_size, thr, pt_color, bone_color):
    """Draw a COCO skeleton (scaled from camera resolution into the panel)."""
    sx, sy = PW / cam_size[0], PH / cam_size[1]
    pix = (kpts * np.array([sx, sy])).round().astype(int)
    for i, j in COCO_SKELETON:
        if scores[i] >= thr and scores[j] >= thr:
            cv2.line(canvas, tuple(pix[i]), tuple(pix[j]), bone_color, 2)
    for k in range(len(pix)):
        if scores[k] >= thr:
            cv2.circle(canvas, tuple(pix[k]), 4, pt_color, -1)
    return canvas


def label(panel, text):
    cv2.rectangle(panel, (0, 0), (PW, 22), (0, 0, 0), -1)
    cv2.putText(panel, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def depth_panel(depth_map, kpts, scores, cam_size, thr):
    """Colorize the synthetic depth map and overlay the 2D skeleton."""
    valid = depth_map > 0
    vis = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), np.uint8)
    if valid.any():
        lo, hi = depth_map[valid].min(), depth_map[valid].max()
        norm = np.zeros_like(depth_map)
        norm[valid] = (depth_map[valid] - lo) / max(hi - lo, 1e-6)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cm[~valid] = (30, 30, 30)
        vis = cm
    panel = cv2.resize(vis, (PW, PH))
    return draw_skeleton_2d(panel, kpts, scores, cam_size, thr, (255, 255, 255), (200, 200, 200))


def render_3d(fig, ax, pose):
    ax.cla()
    plot_skeleton_3d(pose, ax=ax)
    ax.set_xlim(*_XLIM); ax.set_ylim(*_YLIM); ax.set_zlim(*_ZLIM)
    fig.canvas.draw()
    bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    return cv2.resize(bgr, (PW, PH))


def main() -> None:
    ap = argparse.ArgumentParser(description="2 RGB + 1 RGB-D hybrid 3D pose result video.")
    ap.add_argument("--config", default="config/cameras.yaml")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--jitter", type=float, default=0.0, help="2D/3D noise (m) to exercise smoothing.")
    ap.add_argument("--video", default="output/hybrid_pose3d.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    pipe = Pipeline.from_config(args.config)
    thr = pipe.score_threshold
    rgb_idx = [i for i, c in enumerate(pipe.cameras) if not isinstance(c, DepthCameraParams)]
    depth_idx = pipe.depth_idx
    if depth_idx is None or len(rgb_idx) < 2:
        print("[hybrid] config must have >=2 RGB cameras and 1 RGB-D camera."); sys.exit(1)
    print(f"[hybrid] RGB views: {[pipe.cameras[i].name for i in rgb_idx]} | "
          f"depth view: {pipe.cameras[depth_idx].name} | {args.frames} frames")

    fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    writer = None
    video_path = Path(args.video)

    for f, skeleton in enumerate(synthesize_sequence(args.frames, amplitude=0.25, jitter=args.jitter)):
        kpts, scores = synthesize_observations(skeleton, pipe.cameras)
        depth_map = synthesize_depth_map(skeleton, pipe.cameras[depth_idx], stamp_radius=8)
        pose = pipe.process(kpts, scores, depth_map, timestamp=f / args.fps)

        # Build the four panels.
        panels = []
        for n, i in enumerate(rgb_idx[:2]):
            p = np.full((PH, PW, 3), 245, np.uint8)
            draw_skeleton_2d(p, kpts[i], scores[i], pipe.cameras[i].image_size, thr, (0, 0, 255), (0, 180, 0))
            panels.append(label(p, f"{pipe.cameras[i].name} (RGB{n + 1}) - 2D in"))
        dp = depth_panel(depth_map, kpts[depth_idx], scores[depth_idx], pipe.cameras[depth_idx].image_size, thr)
        panels.append(label(dp, f"{pipe.cameras[depth_idx].name} (Depth) - 2D + depth in"))
        p3d = render_3d(fig, ax, pose)
        panels.append(label(p3d, f"3D reconstruction (frame {f:03d})"))

        top = np.hstack([panels[0], panels[1]])
        bot = np.hstack([panels[2], panels[3]])
        frame = np.vstack([top, bot])

        if writer is None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(frame)

    writer.release()
    plt.close(fig)
    print(f"[hybrid] result video ({args.frames} frames) -> {video_path}")


if __name__ == "__main__":
    main()
