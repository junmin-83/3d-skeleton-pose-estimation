"""RTMPose 2D 데모: 프레임마다 COCO-17 키포인트를 뽑는다.

프레임 소스(웹캠 --camera 또는 정적 이미지를 --frames번 반복)에 RTMPoseDetector를
돌려서 프레임별 FPS를 찍고, 마지막 프레임의 17개 키포인트를 출력하고, 전체 프레임을
annotated MP4 + 마지막 프레임 PNG로 저장한다.

rtmlib + onnxruntime 필요(GPU 권장, CPU도 동작). 첫 실행 때 RTMPose ONNX 모델을
받아온다. 없으면 메시지 찍고 종료.

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

# examples/에서 직접 실행할 때 src import 되게.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.types import COCO_17_KEYPOINTS  # noqa: E402
from src.pose2d.rtmpose_detector import RTMPoseDetector  # noqa: E402
from src.render.skeleton_2d import draw_skeleton_2d  # noqa: E402
from src.render.video_writer import LazyVideoWriter  # noqa: E402


def frame_source(args: argparse.Namespace):
    """웹캠(라이브) 또는 반복되는 정적 이미지에서 BGR 프레임을 yield."""
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
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                    help="inference device (default cuda; auto-falls back to CPU if no GPU).")
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
    writer = LazyVideoWriter(args.video, args.fps)
    for idx, frame in enumerate(frame_source(args)):
        t0 = time.perf_counter()
        pose = detector.detect_best(frame)        # Pose2D, COCO 17개 키포인트
        dt = time.perf_counter() - t0
        times.append(dt)

        annotated = draw_skeleton_2d(
            frame, pose.keypoints, pose.scores, detector.score_threshold,
            copy=True, line_color=(0, 255, 0), point_radius=4,
        )
        writer.write(annotated)
        last_pose, last_annotated = pose, annotated

        n_valid = int((pose.scores >= detector.score_threshold).sum())
        print(f"[demo] frame {idx:02d}: {n_valid}/17 keypoints  "
              f"{1.0 / dt:5.1f} FPS ({dt * 1000:4.0f} ms)")

    if last_pose is None:
        print("[demo] no frames processed.")
        sys.exit(1)

    if writer.opened:
        writer.release()
        print(f"[demo] annotated video ({len(times)} frames @ {args.fps:g} fps) -> {args.video}")

    # 첫 프레임은 모델 워밍업이 섞이니 빼고 정상 상태 처리량을 본다.
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
