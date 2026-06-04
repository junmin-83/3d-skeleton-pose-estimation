"""실제 MVOR 멀티뷰 RGB-D 영상으로 hybrid **fused** 3D pose를 시각화하는 데모.

``fusion_demo.py``(3-C)가 합성으로 보여준 간판 경로(다중 뷰 confidence-가중
삼각측량 + depth back-projection을 관절별 융합)를, MVOR 데이터셋의 **실픽셀 +
실depth**로 그대로 돌린다. MVOR은 캘리브된 RGB-D 카메라 3대가 한 장면을 보므로,
2대(이상)의 color로 삼각측량 + 1대의 정렬 depth로 fusion → ``source='fused'`` 경로.

각 프레임: 3개 color에 RTMPose(COCO-17) → 진짜 ``Pipeline``(삼각측량+RANSAC →
depth fusion → One-Euro) → ``[cam RGB+2D | cam RGB+2D | depth+2D | 3D(출처색)]``
4분할 MP4. 출처 색은 fusion_demo와 동일(fused=초록, depth=파랑, tri=빨강).

데이터 준비 (이미지 zip은 별도 다운로드):
    wget https://s3.unistra.fr/camma_public/datasets/mvor/camma_mvor_dataset.zip
    # 압축 해제 후 구조:  <root>/day1/cam{1,2,3}/{color,depth}/*.png
    # 주석 JSON:  https://github.com/CAMMA-public/MVOR/raw/master/annotations/camma_mvor_2018.json

Usage::

    uv run python examples/mvor_fusion_demo.py \
        --mvor-root data/mvor/camma_mvor_dataset \
        --json data/mvor/camma_mvor_2018.json --num-frames 80 --device cuda
    #   -> output/mvor_fusion_pose3d.mp4
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

from src.core.types import NUM_KEYPOINTS  # noqa: E402
from src.io.sources.mvor import iter_mvor_frames, load_mvor_cameras  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.pose2d.rtmpose_detector import RTMPoseDetector  # noqa: E402
from src.render.skeleton_2d import draw_skeleton_2d, label_panel  # noqa: E402
from src.render.skeleton_3d import render_pose3d_by_source  # noqa: E402
from src.render.video_writer import LazyVideoWriter  # noqa: E402

PW, PH = 320, 240  # per-panel size


def depth_panel(depth_m, kpts, scores, thr, dmin, dmax, scale):
    """Colour-mapped metric depth + the depth view's 2D skeleton overlay."""
    valid = (depth_m > 0) & (depth_m >= dmin) & (depth_m <= dmax)
    vis = np.full((*depth_m.shape, 3), 30, np.uint8)
    if valid.any():
        norm = np.zeros_like(depth_m)
        norm[valid] = (depth_m[valid] - dmin) / max(dmax - dmin, 1e-6)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cm[~valid] = (30, 30, 30)
        vis = cm
    panel = cv2.resize(vis, (PW, PH))
    return draw_skeleton_2d(panel, kpts, scores, thr, scale=scale)


def rgb_panel(color, kpts, scores, thr, title, scale):
    """Resized colour frame with its 2D skeleton + caption."""
    panel = draw_skeleton_2d(cv2.resize(color, (PW, PH)), kpts, scores, thr, scale=scale)
    return label_panel(panel, title, PW, font_scale=0.42)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real MVOR RGB-D hybrid fusion 3D demo.")
    ap.add_argument("--mvor-root", required=True, help="extracted camma_mvor_dataset dir.")
    ap.add_argument("--json", required=True, help="camma_mvor_2018.json path.")
    ap.add_argument("--depth-cam", type=int, default=3, help="1-based cam_id used as depth provider.")
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=80)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--depth-min", type=float, default=0.3)
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--depth-scale", type=float, default=1000.0, help="raw depth -> meters divisor.")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--video", default="output/mvor_fusion_pose3d.mp4")
    args = ap.parse_args()

    cameras = load_mvor_cameras(args.json, depth_cam=args.depth_cam)
    depth_idx = args.depth_cam - 1
    rgb_idx = [i for i in range(len(cameras)) if i != depth_idx]
    config = {
        "triangulation": {"score_threshold": 0.3, "min_views": 2,
                          "ransac": {"enabled": True, "reproj_threshold_px": 15.0}},
        "depth_fusion": {"enabled": True, "fill_missing": True, "patch_radius_px": 3,
                         "depth_min": args.depth_min, "depth_max": args.depth_max},
        "smoothing": {"enabled": True, "freq": args.fps, "min_cutoff": 1.0,
                      "beta": 0.01, "d_cutoff": 1.0},
    }
    pipe = Pipeline(config, cameras)
    detector = RTMPoseDetector(device=args.device, mode=args.mode, score_threshold=0.3)

    fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    writer = LazyVideoWriter(args.video, args.fps)
    lims = None
    scale = (PW / 640.0, PH / 480.0)

    n = 0
    frames = iter_mvor_frames(args.mvor_root, args.json, depth_cam=args.depth_cam,
                              day=args.day, start=args.start, num=args.num_frames,
                              depth_scale=args.depth_scale)
    for frame in frames:
        kpts = np.zeros((len(cameras), NUM_KEYPOINTS, 2))
        scores = np.zeros((len(cameras), NUM_KEYPOINTS))
        for i, color in enumerate(frame.colors):
            pose2d = detector.detect_best(color)
            kpts[i], scores[i] = pose2d.keypoints, pose2d.scores
        pose = pipe.process(kpts, scores, depth_map=frame.depth_m, timestamp=n / args.fps)
        if lims is None and pose.valid.any():
            c = pose.points[pose.valid].mean(axis=0)
            lims = [(c[a] - 1.0, c[a] + 1.0) for a in range(3)]

        panels = [
            rgb_panel(frame.colors[rgb_idx[0]], kpts[rgb_idx[0]], scores[rgb_idx[0]], 0.3,
                      f"cam{rgb_idx[0] + 1} RGB+2D", scale),
            rgb_panel(frame.colors[rgb_idx[1]], kpts[rgb_idx[1]], scores[rgb_idx[1]], 0.3,
                      f"cam{rgb_idx[1] + 1} RGB+2D", scale),
            label_panel(depth_panel(frame.depth_m, kpts[depth_idx], scores[depth_idx], 0.3,
                                    args.depth_min, args.depth_max, scale),
                        f"cam{args.depth_cam} depth+2D", PW, font_scale=0.42),
            label_panel(render_pose3d_by_source(fig, ax, pose, lims or [(-1, 1)] * 3, (PW, PH),
                                                 view_init=(-75, -90)),
                        f"3D fused=G depth=B tri=R f{n}", PW, font_scale=0.42),
        ]
        writer.write(np.hstack(panels))

        if n % 10 == 0:
            src = pose.source
            counts = {s: src.count(s) for s in ("fused", "depth", "triangulation", "missing")}
            print(f"[mvor] frame {n} ({frame.frame_id}): fused={counts['fused']} "
                  f"depth={counts['depth']} tri={counts['triangulation']} missing={counts['missing']}")
        n += 1

    if n == 0:
        print("[mvor] no frames processed — check --mvor-root / --json / --day.")
        sys.exit(1)
    writer.release()
    plt.close(fig)
    print(f"[mvor] result video ({n} frames) -> {args.video}")


if __name__ == "__main__":
    main()
