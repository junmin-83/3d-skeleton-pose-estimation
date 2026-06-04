"""실제 RGB-D 영상에서 Depth를 활용해 3D pose를 추출하고 MP4로 만드는 데모.

단일 RGB-D 카메라(컬러 + 정렬된 depth)에서:
  RTMPose로 컬러 프레임의 COCO-17 2D 키포인트를 검출(실제 픽셀) →
  각 키포인트 위치의 depth를 읽어 back-projection(`fusion.back_project_depth_keypoints`)으로 3D 복원 →
  [컬러+2D | depth 컬러맵 | 3D 스켈레톤] 3분할 MP4 저장.
삼각측량 없이 **depth 정보만으로** 3D를 만드는 경로다(단일 카메라 → world = 카메라 좌표계).

지원 입력:
  --tum <dir>   TUM RGB-D 포맷 (rgb/*.png + depth/*.png(16bit) + rgb.txt/depth.txt)
                예: data/tum/rgbd_dataset_freiburg3_sitting_static
  --realsense   Intel RealSense 라이브(pyrealsense2 필요; 컬러에 정렬된 depth + intrinsics 자동)

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

from src.core.types import NUM_KEYPOINTS, Pose3D  # noqa: E402
from src.fusion.depth_fusion import back_project_depth_keypoints  # noqa: E402
from src.pose2d.rtmpose_detector import RTMPoseDetector  # noqa: E402
from src.render.skeleton_2d import draw_skeleton_2d, label_panel  # noqa: E402
from src.render.skeleton_3d import render_pose3d_frame  # noqa: E402
from src.render.video_writer import LazyVideoWriter  # noqa: E402
from src.smoothing.one_euro import PoseSmoother  # noqa: E402

PW, PH = 320, 240  # per-panel size
# TUM freiburg3 RGB intrinsics + depth scale (raw / scale = meters).
_TUM_FR3 = dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6, depth_scale=5000.0)


def _read_tum_assoc(tum_dir: Path) -> list[tuple[str, str]]:
    """Associate each RGB frame with the nearest-timestamp depth frame."""
    def load(name):
        out = []
        for line in (tum_dir / name).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, fname = line.split()
            out.append((float(ts), fname))
        return out

    rgb, depth = load("rgb.txt"), load("depth.txt")
    dts = np.array([t for t, _ in depth])
    pairs = []
    for ts, rgb_f in rgb:
        j = int(np.argmin(np.abs(dts - ts)))
        if abs(dts[j] - ts) <= 0.02:  # 20 ms
            pairs.append((rgb_f, depth[j][1]))
    return pairs


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


def _realsense_frames(num):
    """Yield (color_bgr, depth_meters, K) from a live Intel RealSense camera."""
    import pyrealsense2 as rs  # lazy: only needed for --realsense

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)  # depth -> color alignment
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
    try:
        for _ in range(num):
            frames = align.process(pipe.wait_for_frames())
            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            if not cf or not df:  # startup / dropped frame
                continue
            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale
            yield color, depth, K
    finally:
        pipe.stop()


def _tum_frames(tum_dir, start, num):
    pairs = _read_tum_assoc(tum_dir)[start:start + num]
    K = np.array([[_TUM_FR3["fx"], 0, _TUM_FR3["cx"]],
                  [0, _TUM_FR3["fy"], _TUM_FR3["cy"]], [0, 0, 1]])
    for rgb_f, depth_f in pairs:
        color = cv2.imread(str(tum_dir / rgb_f))
        raw = cv2.imread(str(tum_dir / depth_f), cv2.IMREAD_UNCHANGED)
        if color is None or raw is None:
            continue
        yield color, raw.astype(np.float32) / _TUM_FR3["depth_scale"], K


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
    args = ap.parse_args()

    if args.tum:
        frames = _tum_frames(Path(args.tum), args.start, args.num_frames)
    elif args.realsense:
        frames = _realsense_frames(args.num_frames)
    else:
        print("[rgbd] specify --tum <dir> or --realsense")
        sys.exit(1)

    detector = RTMPoseDetector(device=args.device, mode=args.mode, score_threshold=0.3)
    smoother = PoseSmoother(NUM_KEYPOINTS, freq=args.fps, min_cutoff=1.0, beta=0.01, d_cutoff=1.0)
    fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    writer = LazyVideoWriter(args.video, args.fps)
    lims = None
    scale = (PW / 640.0, PH / 480.0)

    n = 0
    for color, depth_m, K in frames:
        pose2d = detector.detect_best(color)
        pts3d, valid = back_project_depth_keypoints(
            pose2d.keypoints, depth_m, K, np.eye(3), np.zeros(3),
            patch_radius=2, depth_min=args.depth_min, depth_max=args.depth_max,
        )
        # Gate by 2D confidence: a low-score (e.g. undetected) joint whose pixel
        # happens to land on valid background depth must NOT yield a 3D point
        # (prevents a "ghost skeleton" when no person is present).
        valid = valid & (pose2d.scores >= detector.score_threshold)
        pose = smoother.update(Pose3D(pts3d, pose2d.scores, valid, ["depth"] * NUM_KEYPOINTS),
                               timestamp=n / args.fps)
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


if __name__ == "__main__":
    main()
