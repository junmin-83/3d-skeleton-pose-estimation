"""실제 CMU Panoptic HD 영상으로 3D pose 결과영상을 만드는 데모 (RTMPose-on-pixels).

다운로드한 Panoptic HD 비디오 N대(권장 단일 인물 시퀀스, 예: 171204_pose1)를 입력으로:
프레임마다 각 뷰에 RTMPose를 돌려 2D를 얻고(실제 픽셀!), 실제 Panoptic calibration으로
삼각측량해 3D를 복원한 뒤, [입력 뷰들 + 3D 복원]을 합성한 MP4를 만든다.

전제 데이터(panoptic-toolbox 또는 직접 curl로 받기; README의 2-e 참고):
  <seq>/calibration_<seq>.json
  <seq>/hdVideos/hd_00_<NN>.mp4              (HD RGB, 카메라당 ~2.8GB)
  <seq>/hdPose3d_stage1_coco19/body3DScene_*.json   (선택: GT 비교용)

주의:
  - 단일 인물 시퀀스를 쓰세요. 다인 장면은 뷰 간 인물 매칭(association)이 추가로 필요하며
    이 파이프라인은 single-person(best_person)만 지원합니다.
  - depth(Kinect)는 .dat 원시 포맷 디코딩/동기/정렬이 필요해 여기선 HD RGB 삼각측량만
    수행합니다(depth fusion off). depth까지 쓰려면 kcalibration + KINECTNODE depthdata.dat를
    디코딩해 aligned depth맵(미터)을 pipeline.process(depth_map=...)로 넘기면 됩니다.

Usage::

    uv run python examples/panoptic_video_demo.py \
        --seq-dir data/panoptic/171204_pose1 \
        --cams 00_03,00_12,00_23 --start 500 --num-frames 60 --device cuda
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
from src.pose2d.rtmpose_detector import RTMPoseDetector  # noqa: E402

PW, PH = 360, 240
_CM_TO_M = 0.01  # CMU Panoptic calibration is in centimeters


def load_panoptic_hd_cameras(calib_path: str) -> list[CameraParams]:
    """CMU Panoptic calibration JSON -> CameraParams (HD cams only, cm -> m)."""
    doc = json.load(open(calib_path, encoding="utf-8"))
    cams = []
    for c in doc["cameras"]:
        if c["type"] != "hd":
            continue
        cams.append(CameraParams(
            name=c["name"],
            K=np.asarray(c["K"], float),
            dist=np.asarray(c["distCoef"], float),
            R=np.asarray(c["R"], float),
            t=np.asarray(c["t"], float).reshape(3) * _CM_TO_M,
            image_size=(int(c["resolution"][0]), int(c["resolution"][1])),
        ))
    return cams


def draw_2d(canvas, kpts, scores, cam_size, thr):
    sx, sy = PW / cam_size[0], PH / cam_size[1]
    pix = (kpts * np.array([sx, sy])).round().astype(int)
    for i, j in COCO_SKELETON:
        if scores[i] >= thr and scores[j] >= thr:
            cv2.line(canvas, tuple(pix[i]), tuple(pix[j]), (0, 180, 0), 2)
    for k in range(len(pix)):
        if scores[k] >= thr:
            cv2.circle(canvas, tuple(pix[k]), 3, (0, 0, 255), -1)
    return canvas


def label(panel, text):
    cv2.rectangle(panel, (0, 0), (PW, 20), (0, 0, 0), -1)
    cv2.putText(panel, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def render_3d(fig, ax, pose, lims):
    ax.cla()
    pts, valid = pose.points, pose.valid
    for i, j in COCO_SKELETON:
        if valid[i] and valid[j]:
            ax.plot(*[[pts[i, a], pts[j, a]] for a in range(3)], c="royalblue", lw=2)
    if valid.any():
        ax.scatter(*pts[valid].T, c="royalblue", s=20)
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])
    ax.set_xlabel("X(m)")
    ax.set_ylabel("Y(m)")
    ax.set_zlabel("Z(m)")
    fig.canvas.draw()
    bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    return cv2.resize(bgr, (PW, PH))


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
    args = ap.parse_args()

    seq = Path(args.seq_dir)
    calib = next(seq.glob("calibration_*.json"))
    cams_all = {c.name: c for c in load_panoptic_hd_cameras(str(calib))}
    names = [n.strip() for n in args.cams.split(",")]
    cams = [cams_all[n] for n in names]
    caps = [cv2.VideoCapture(str(seq / "hdVideos" / f"hd_00_{n.split('_')[1]}.mp4")) for n in names]
    for cap, n in zip(caps, names):
        if not cap.isOpened():
            print(f"[panoptic-vid] cannot open hdVideos/hd_00_{n.split('_')[1]}.mp4")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
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
    writer, lims = None, None
    video_path = Path(args.video)

    for f in range(args.num_frames):
        frames = []
        for cap in caps:
            ok, img = cap.read()
            frames.append(img if ok else None)
        if any(im is None for im in frames):
            print(f"[panoptic-vid] stream ended at frame {f}")
            break

        kpts = np.zeros((len(cams), 17, 2))
        scores = np.zeros((len(cams), 17))
        for v, img in enumerate(frames):
            pose2d = detector.detect_best(img)
            kpts[v], scores[v] = pose2d.keypoints, pose2d.scores
        pose = pipe.process(kpts, scores, depth_map=None, timestamp=f / args.fps)
        if lims is None and pose.valid.any():
            lims = _limits_from(pose)

        panels = []
        for v, n in enumerate(names):
            p = cv2.resize(frames[v], (PW, PH))
            draw_2d(p, kpts[v], scores[v], cams[v].image_size, 0.4)
            panels.append(label(p, f"HD {n} (RGB) 2D"))
        panels.append(label(render_3d(fig, ax, pose, lims or _limits_from(pose)),
                            f"3D reconstruction f{args.start + f}"))
        frame = np.hstack(panels)

        if writer is None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(frame)
        if f % 10 == 0:
            print(f"[panoptic-vid] frame {f}: {int(pose.valid.sum())}/17 joints reconstructed")

    for cap in caps:
        cap.release()
    if writer is not None:
        writer.release()
        print(f"[panoptic-vid] result video -> {video_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
