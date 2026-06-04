"""CMU Panoptic HD 멀티뷰 영상으로 3D pose 결과영상을 만드는 데모 (RTMPose-on-pixels).

HD 비디오 여러 대(단일 인물 시퀀스 권장, 예: 171204_pose1)를 받아서, 프레임마다 각 뷰에
RTMPose를 돌려 2D를 얻고 Panoptic calibration으로 삼각측량해 3D를 복원한다. 출력은
[입력 뷰들 + 3D 복원]을 붙인 MP4.

필요한 데이터(panoptic-toolbox나 curl로 직접 받기; README 2-e 참고):
  <seq>/calibration_<seq>.json
  <seq>/hdVideos/hd_00_<NN>.mp4              (HD RGB, 카메라당 ~2.8GB)
  <seq>/hdPose3d_stage1_coco19/body3DScene_*.json   (선택: GT 비교용)

주의:
  - 단일 인물 시퀀스만. 다인 장면은 뷰 간 인물 매칭(association)이 더 필요한데
    이 파이프라인은 single-person(best_person)만 지원한다.
  - depth(Kinect)는 .dat 디코딩/동기/정렬이 필요해서 여기선 HD RGB 삼각측량만 한다
    (depth fusion off). depth까지 쓰려면 kcalibration + KINECTNODE depthdata.dat를
    디코딩해 정렬된 depth맵(미터)을 pipeline.process(depth_map=...)로 넘기면 된다.

Usage::

    uv run python examples/panoptic_video_demo.py \
        --seq-dir data/panoptic/171204_pose1 \
        --cams 00_03,00_12,00_23 --start 500 --num-frames 60 --device cuda
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

from src.io.sources.panoptic import iter_panoptic_hd_frames, load_panoptic_hd_cameras  # noqa: E402
from src.io.keypoints_io import export_keypoints  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.pose2d.rtmpose_detector import RTMPoseDetector  # noqa: E402
from src.render.skeleton_2d import draw_skeleton_2d, label_panel  # noqa: E402
from src.render.skeleton_3d import render_pose3d_frame  # noqa: E402
from src.render.video_writer import LazyVideoWriter  # noqa: E402

PW, PH = 360, 240


def _limits_from(pose, pad=0.6):
    p = pose.points[pose.valid]
    c = p.mean(axis=0) if len(p) else np.array([0.0, 0.0, 0.0])
    return [(c[a] - pad, c[a] + pad) for a in range(3)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Real CMU Panoptic HD multi-view 3D pose video.")
    ap.add_argument("--seq-dir", required=True, help="sequence dir with calibration_*.json + hdVideos/.")
    ap.add_argument("--cams", default="00_03,00_12,00_23", help="comma-separated HD camera names.")
    ap.add_argument("--start", type=int, default=0, help="start HD frame index.")
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                    help="inference device (default cuda; auto-falls back to CPU if no GPU).")
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--video", default="output/panoptic_video_pose3d.mp4")
    ap.add_argument("--keypoints", default="output/panoptic_pose3d.json",
                    help="per-frame 3D keypoints output (points/scores/valid/source).")
    ap.add_argument("--keypoints-format", default="json", choices=["json", "npy"])
    args = ap.parse_args()

    seq = Path(args.seq_dir)
    calib = next(seq.glob("calibration_*.json"))
    cams_all = {c.name: c for c in load_panoptic_hd_cameras(str(calib))}
    names = [n.strip() for n in args.cams.split(",")]
    cams = [cams_all[n] for n in names]
    try:
        frame_iter = iter_panoptic_hd_frames(seq, names, args.start, args.num_frames)
    except FileNotFoundError as exc:
        print(f"[panoptic-vid] {exc}")
        sys.exit(1)
    print(f"[panoptic-vid] {len(cams)} views {names} | device={args.device} | "
          f"frames {args.start}..{args.start + args.num_frames}")

    detector = RTMPoseDetector(device=args.device, mode=args.mode, score_threshold=0.4)
    config = {"triangulation": {"score_threshold": 0.4, "min_views": 2,
                                "ransac": {"enabled": True, "reproj_threshold_px": 15.0}},
              "depth_fusion": {"enabled": False},
              "smoothing": {"enabled": True, "freq": args.fps, "min_cutoff": 1.0, "beta": 0.01, "d_cutoff": 1.0}}
    pipe = Pipeline(config, cams)

    fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    writer = LazyVideoWriter(args.video, args.fps)
    lims = None
    poses: list = []

    for f, frames in enumerate(frame_iter):
        kpts = np.zeros((len(cams), 17, 2))
        scores = np.zeros((len(cams), 17))
        for v, img in enumerate(frames):
            pose2d = detector.detect_best(img)
            kpts[v], scores[v] = pose2d.keypoints, pose2d.scores
        pose = pipe.process(kpts, scores, depth_map=None, timestamp=f / args.fps)
        poses.append(pose)
        if lims is None and pose.valid.any():
            lims = _limits_from(pose)

        panels = []
        for v, n in enumerate(names):
            p = cv2.resize(frames[v], (PW, PH))
            cam_w, cam_h = cams[v].image_size
            draw_skeleton_2d(p, kpts[v], scores[v], 0.4, scale=(PW / cam_w, PH / cam_h))
            panels.append(label_panel(p, f"HD {n} (RGB) 2D", PW, font_scale=0.45))
        panels.append(label_panel(
            render_pose3d_frame(fig, ax, pose, lims or _limits_from(pose), (PW, PH), point_size=20),
            f"3D reconstruction f{args.start + f}", PW, font_scale=0.45))
        frame = np.hstack(panels)

        writer.write(frame)
        if f % 10 == 0:
            print(f"[panoptic-vid] frame {f}: {int(pose.valid.sum())}/17 joints reconstructed")

    if writer.opened:
        writer.release()
        print(f"[panoptic-vid] result video -> {args.video}")
    plt.close(fig)
    export_keypoints(poses, args.keypoints, fmt=args.keypoints_format)
    print(f"[panoptic-vid] 3D keypoints ({len(poses)} frames) -> {args.keypoints}")


if __name__ == "__main__":
    main()
