"""3D 포즈 추정 결과를 '영상(MP4)'으로 보여주는 데모.

전체 파이프라인(삼각측량 -> depth fusion -> One-Euro 스무딩)을 프레임 시퀀스에
대해 실제로 실행하고, 복원된 3D 스켈레톤을 프레임마다 matplotlib 3D로 렌더링해
MP4로 저장한다.

[중요] 실제 영상에서의 3D 복원은 **calibration된 2대 이상의 동기화된 뷰**가 반드시
필요하다(삼각측량이 depth 모호성을 풀려면 멀티뷰가 필수). 단일 뷰 클립으로는
기하학적으로 불가능하므로, 이 데모는 알려진 3D 스켈레톤을 config의 카메라들로
투영해 만든 **합성 멀티뷰 2D**로 파이프라인을 구동한다. 실제 영상으로 돌리려면
calibration된 멀티 카메라 녹화를 준비해 각 뷰를 RTMPoseDetector -> pipeline.process
로 넣으면 된다(코드 경로는 동일).

Usage::

    uv run python examples/pose3d_video.py --frames 90 --jitter 0.0
    uv run python examples/pose3d_video.py --frames 120 --jitter 0.01   # 노이즈+스무딩 효과
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless 렌더링
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Pipeline  # noqa: E402
from src.synthetic import (  # noqa: E402
    synthesize_depth_map,
    synthesize_observations,
    synthesize_sequence,
)
from src.viz.visualize_3d import plot_skeleton_3d  # noqa: E402

# 영상이 프레임마다 튀지 않도록 3D 축 범위를 고정한다 (meter, world=cam0 frame).
_XLIM, _YLIM, _ZLIM = (-0.9, 0.9), (-1.0, 1.0), (1.8, 3.2)


def fig_to_bgr(fig) -> np.ndarray:
    """matplotlib figure -> OpenCV BGR 프레임."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D pose estimation rendered to an MP4.")
    ap.add_argument("--config", default="config/cameras.yaml", help="camera/pipeline config.")
    ap.add_argument("--frames", type=int, default=90, help="number of frames.")
    ap.add_argument("--jitter", type=float, default=0.0, help="2D/3D noise (m) to exercise smoothing.")
    ap.add_argument("--video", default="output/pose3d_demo.mp4", help="output MP4 path.")
    ap.add_argument("--png", default="output/pose3d_demo_frame.png", help="sample mid-frame PNG path.")
    ap.add_argument("--fps", type=float, default=30.0, help="output video frame rate.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline = Pipeline.from_config(args.config)
    depth_cam = pipeline.cameras[pipeline.depth_idx] if pipeline.depth_idx is not None else None
    print(f"[3d-demo] {len(pipeline.cameras)} views, depth={'yes' if depth_cam else 'no'}; "
          f"running {args.frames} frames ...")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    writer = None
    video_path = Path(args.video)
    errors: list[float] = []
    mid = args.frames // 2

    for idx, skeleton in enumerate(synthesize_sequence(args.frames, amplitude=0.3, jitter=args.jitter)):
        keypoints, scores = synthesize_observations(skeleton, pipeline.cameras)
        depth_map = synthesize_depth_map(skeleton, depth_cam) if depth_cam is not None else None
        pose = pipeline.process(keypoints, scores, depth_map, timestamp=idx / args.fps)

        if pose.valid.any():
            errors.append(float(np.linalg.norm(pose.points[pose.valid] - skeleton[pose.valid], axis=1).mean()))

        ax.cla()
        plot_skeleton_3d(pose, ax=ax, title=f"3D pose - frame {idx:03d}")
        ax.set_xlim(*_XLIM)
        ax.set_ylim(*_YLIM)
        ax.set_zlim(*_ZLIM)
        frame = fig_to_bgr(fig)

        if writer is None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
            )
            if not writer.isOpened():
                print(f"[3d-demo] ERROR: cannot open MP4 writer for {video_path}")
                sys.exit(1)
        writer.write(frame)

        if idx == mid:
            cv2.imwrite(args.png, frame)

    writer.release()
    plt.close(fig)
    mean_err = float(np.mean(errors)) if errors else float("nan")
    print(f"[3d-demo] mean 3D error = {mean_err:.3e} m over {len(errors)} frames")
    print(f"[3d-demo] video -> {video_path}")
    print(f"[3d-demo] sample frame -> {args.png}")


if __name__ == "__main__":
    main()
