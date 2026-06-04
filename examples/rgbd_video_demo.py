"""단일 RGB-D(컬러 + 정렬 depth)로 depth 기반 3D pose를 뽑아 MP4로 저장하는 데모.

컬러에 RTMPose를 돌려 COCO-17 2D를 얻고, 각 키포인트 위치의 depth를
back_project_depth_keypoints로 back-projection해 3D를 복원한다. 삼각측량 없이
depth만 쓰는 경로(단일 카메라라 world == 카메라 좌표계). 출력:
[컬러+2D | depth 컬러맵 | 3D 스켈레톤] 3분할 MP4.

입력은 둘 중 하나:
  --tum <dir>   TUM RGB-D 포맷 (rgb/*.png + depth/*.png 16bit + rgb.txt/depth.txt)
                예: data/tum/rgbd_dataset_freiburg3_sitting_static
  --realsense   Intel RealSense 라이브 (pyrealsense2 필요; 정렬 depth + intrinsics 자동)

Usage::

    uv run python examples/rgbd_video_demo.py --tum data/tum/rgbd_dataset_freiburg3_sitting_static \
        --num-frames 60 --device cuda
    uv run python examples/rgbd_video_demo.py --realsense --num-frames 300 --device cuda
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

from src.core.types import DepthCameraParams  # noqa: E402
from src.io.sources.realsense import RealSenseSource  # noqa: E402
from src.io.sources.tum import TUMSource  # noqa: E402
from src.io.keypoints_io import export_keypoints  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.pose2d.rtmpose_detector import RTMPoseDetector  # noqa: E402
from src.render.skeleton_2d import draw_skeleton_2d, label_panel  # noqa: E402
from src.render.skeleton_3d import render_pose3d_frame  # noqa: E402
from src.render.video_writer import LazyVideoWriter  # noqa: E402

PW, PH = 320, 240  # 패널 하나 크기


def depth_panel(depth_m, kpts, scores, thr, dmin, dmax):
    valid = (depth_m > 0) & (depth_m >= dmin) & (depth_m <= dmax)
    vis = np.full((*depth_m.shape, 3), 30, np.uint8)
    if valid.any():
        norm = np.zeros_like(depth_m)
        norm[valid] = (depth_m[valid] - dmin) / max(dmax - dmin, 1e-6)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cm[~valid] = (30, 30, 30)
        vis = cm
    panel = cv2.resize(vis, (PW, PH))
    return draw_skeleton_2d(panel, kpts, scores, thr, scale=(PW / 640.0, PH / 480.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Single RGB-D -> depth-based 3D pose video.")
    ap.add_argument("--tum", type=str, default=None, help="TUM RGB-D dataset directory.")
    ap.add_argument("--realsense", action="store_true", help="Intel RealSense live (needs pyrealsense2).")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                    help="inference device (default cuda; auto-falls back to CPU if no GPU).")
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--depth-min", type=float, default=0.3)
    ap.add_argument("--depth-max", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--video", default="output/rgbd_pose3d.mp4")
    ap.add_argument("--keypoints", default="output/rgbd_pose3d.json",
                    help="per-frame 3D keypoints output (points/scores/valid/source).")
    ap.add_argument("--keypoints-format", default="json", choices=["json", "npy"])
    args = ap.parse_args()

    if args.tum:
        frames = TUMSource(args.tum, args.start, args.num_frames)
    elif args.realsense:
        frames = RealSenseSource(args.num_frames)
    else:
        print("[rgbd] specify --tum <dir> or --realsense")
        sys.exit(1)

    detector = RTMPoseDetector(device=args.device, mode=args.mode, score_threshold=0.3)
    # 단일 RGB-D라 world == camera (R=I, t=0). 3D 복원(depth back-projection +
    # 2D confidence gating + One-Euro smoothing)은 Pipeline의 depth 경로가 처리한다.
    # 카메라는 첫 프레임에서 intrinsic K가 나온 뒤 lazy로 만든다(K는 소스마다 고정).
    config = {
        "triangulation": {"score_threshold": 0.3, "min_views": 2},
        "depth_fusion": {"enabled": True, "fill_missing": True, "patch_radius_px": 2,
                         "depth_min": args.depth_min, "depth_max": args.depth_max},
        "smoothing": {"enabled": True, "freq": args.fps, "min_cutoff": 1.0,
                      "beta": 0.01, "d_cutoff": 1.0},
    }
    pipe = None
    fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    writer = LazyVideoWriter(args.video, args.fps)
    lims = None
    scale = (PW / 640.0, PH / 480.0)

    poses: list = []
    n = 0
    for color, depth_m, K in frames:
        if pipe is None:
            h, w = color.shape[:2]
            cam = DepthCameraParams(name="rgbd", K=K, dist=np.zeros(5), R=np.eye(3),
                                    t=np.zeros(3), image_size=(w, h), depth_K=K)
            pipe = Pipeline(config, [cam])
        pose2d = detector.detect_best(color)
        pose = pipe.process(pose2d.keypoints[np.newaxis], pose2d.scores[np.newaxis],
                            depth_map=depth_m, timestamp=n / args.fps)
        poses.append(pose)
        if lims is None and pose.valid.any():
            c = pose.points[pose.valid].mean(axis=0)
            lims = [(c[a] - 0.8, c[a] + 0.8) for a in range(3)]

        rgb_panel = draw_skeleton_2d(cv2.resize(color, (PW, PH)), pose2d.keypoints, pose2d.scores, 0.3, scale=scale)
        cpanel = label_panel(rgb_panel, "RGB + 2D", PW, font_scale=0.42)
        dpanel = label_panel(depth_panel(depth_m, pose2d.keypoints, pose2d.scores, 0.3, args.depth_min, args.depth_max),
                             "Depth", PW, font_scale=0.42)
        p3d = label_panel(render_pose3d_frame(fig, ax, pose, lims or [(-1, 1)] * 3, (PW, PH),
                                              point_size=18, view_init=(-75, -90)),
                          f"3D (depth) f{args.start + n}", PW, font_scale=0.42)
        frame = np.hstack([cpanel, dpanel, p3d])

        writer.write(frame)
        if n % 10 == 0:
            print(f"[rgbd] frame {n}: {int(pose.valid.sum())}/17 joints from depth")
        n += 1

    if n == 0:
        print("[rgbd] no frames processed.")
        sys.exit(1)
    writer.release()
    plt.close(fig)
    print(f"[rgbd] result video ({n} frames) -> {args.video}")
    export_keypoints(poses, args.keypoints, fmt=args.keypoints_format)
    print(f"[rgbd] 3D keypoints ({len(poses)} frames) -> {args.keypoints}")


if __name__ == "__main__":
    main()
