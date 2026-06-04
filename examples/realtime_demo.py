"""Real-time-style demo: extract the 17 COCO keypoints per frame with RTMPose.

Runs the project's ``RTMPoseDetector`` over a frame source and reports per-frame
timing (FPS), prints the 17 keypoints of the last frame, and writes an annotated
**MP4 video of every frame** (plus a last-frame still image). The frame source is
either a live webcam (``--camera <idx>``) or a static image looped ``--frames``
times to emulate a stream (default).

Requires ``rtmlib`` + ``onnxruntime`` (see requirements.txt); rtmlib downloads
the RTMPose ONNX model on first run. Exits with a clear message if unavailable.

Usage::

    uv run python examples/realtime_demo.py --frames 30           # looped image
    uv run python examples/realtime_demo.py --camera 0 --frames 100  # live webcam
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make ``src`` importable when this script is run directly from examples/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.types import COCO_17_KEYPOINTS, COCO_SKELETON  # noqa: E402
from src.pose2d.rtmpose_detector import RTMPoseDetector  # noqa: E402


def draw_pose(image: np.ndarray, pose, score_thr: float) -> np.ndarray:
    """Overlay the COCO-17 skeleton + keypoints on a copy of ``image``."""
    out = image.copy()
    kpts, scores = pose.keypoints, pose.scores
    for i, j in COCO_SKELETON:
        if scores[i] >= score_thr and scores[j] >= score_thr:
            p1 = tuple(np.round(kpts[i]).astype(int))
            p2 = tuple(np.round(kpts[j]).astype(int))
            cv2.line(out, p1, p2, (0, 255, 0), 2)
    for k in range(len(kpts)):
        if scores[k] >= score_thr:
            cv2.circle(out, tuple(np.round(kpts[k]).astype(int)), 4, (0, 0, 255), -1)
    return out


def frame_source(args: argparse.Namespace):
    """Yield BGR frames from a webcam (live) or a looped static image."""
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera device {args.camera}")
        try:
            for _ in range(args.frames):
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
    else:
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"image not found: {args.image}")
        for _ in range(args.frames):
            yield image


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Real-time COCO-17 keypoint extraction demo.")
    ap.add_argument("--image", default="data/demo/person.jpg", help="static image (looped) source.")
    ap.add_argument("--camera", type=int, default=None, help="webcam device index for true live mode.")
    ap.add_argument("--frames", type=int, default=30, help="number of frames to process.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="inference device.")
    ap.add_argument("--mode", default="lightweight", help="rtmlib pose mode: lightweight|balanced|performance.")
    ap.add_argument("--score-thr", type=float, default=0.3,
                    help="keypoint confidence threshold; raise it to drop low-confidence (often wrong) joints.")
    ap.add_argument("--out", default="output/realtime_keypoints.png", help="last-frame annotated image path.")
    ap.add_argument("--video", default="output/realtime_keypoints.mp4", help="annotated MP4 video path (all frames).")
    ap.add_argument("--fps", type=float, default=30.0, help="playback frame rate for the output MP4.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[demo] loading RTMPose (device={args.device}, mode={args.mode}); "
          "first run downloads the ONNX model ...")
    try:
        detector = RTMPoseDetector(device=args.device, mode=args.mode, score_threshold=args.score_thr)
    except ImportError as exc:
        print(f"[demo] {exc}")
        sys.exit(1)

    times: list[float] = []
    last_pose = last_annotated = None
    writer = None
    writer_failed = False
    video_path = Path(args.video)
    for idx, frame in enumerate(frame_source(args)):
        t0 = time.perf_counter()
        pose = detector.detect_best(frame)        # -> Pose2D with 17 COCO keypoints
        dt = time.perf_counter() - t0
        times.append(dt)

        annotated = draw_pose(frame, pose, detector.score_threshold)
        if writer is None and not writer_failed:  # lazy-init once the frame size is known
            video_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = annotated.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
            )
            if not writer.isOpened():
                writer.release()
                writer, writer_failed = None, True
                print(f"[demo] WARNING: cannot open MP4 writer for {video_path}; video skipped.")
        if writer is not None:
            writer.write(annotated)
        last_pose, last_annotated = pose, annotated

        n_valid = int((pose.scores >= detector.score_threshold).sum())
        print(f"[demo] frame {idx:02d}: {n_valid}/17 keypoints  "
              f"{1.0 / dt:5.1f} FPS ({dt * 1000:4.0f} ms)")

    if last_pose is None:
        print("[demo] no frames processed.")
        sys.exit(1)

    if writer is not None:
        writer.release()
        print(f"[demo] annotated video ({len(times)} frames @ {args.fps:g} fps) -> {video_path}")

    # First frame includes model warm-up; report steady-state throughput.
    steady = times[1:] or times
    avg_fps = len(steady) / sum(steady)

    print("\n[demo] === 17 COCO keypoints (last frame) ===")
    for k, name in enumerate(COCO_17_KEYPOINTS):
        u, v = last_pose.keypoints[k]
        print(f"  {k:2d}  {name:15s} (u={u:7.1f}, v={v:7.1f})  score={last_pose.scores[k]:.3f}")
    print(f"\n[demo] steady-state {avg_fps:.1f} FPS over {len(steady)} frames "
          f"(device={args.device}, warm-up frame excluded)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), last_annotated)
    print(f"[demo] last-frame image -> {out_path}")


if __name__ == "__main__":
    main()
