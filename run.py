"""3D 스켈레톤 pose 파이프라인 진입점 (라이브 / 녹화).

설정된 카메라 뷰마다 RTMPose 2D 검출을 돌려 3D로 삼각측량하고(depth 소스가 연결돼
있으면 depth fusion 추가), One-Euro 스무딩을 거쳐 프레임별 3D 키포인트를 내보낸다.
rtmlib + onnxruntime와 config/cameras.yaml의 카메라 소스/캘리브레이션이 필요하다.

    uv run python run.py --config config/cameras.yaml
    uv run python run.py --live                  # 웹캠 등을 실시간 창으로 표시 (q/Esc 종료)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import Pipeline
from src.io.keypoints_io import export_keypoints
from src.render.skeleton_3d import save_skeleton_png


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D skeleton pose estimation (live/recorded).")
    parser.add_argument("--config", default="config/cameras.yaml", help="camera/pipeline config YAML.")
    parser.add_argument("--output", default=None, help="override output file path.")
    parser.add_argument("--viz", action="store_true", help="save a PNG of the first frame's skeleton.")
    parser.add_argument("--live", action="store_true",
                        help="show a live window (2D overlays + 3D); press q or Esc to quit.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after N frames (caps an otherwise endless webcam loop).")
    return parser.parse_args()


def run_live(pipeline: Pipeline, show: bool = False, max_frames: int | None = None) -> list:
    """설정된 소스에서 동기화된 프레임을 읽어 3D를 복원한다.

    show=True면 프레임마다 라이브 창(2D 오버레이 + 3D)을 띄우고 q/Esc로 멈춘다.
    max_frames가 주어지면 그 수만큼 처리한 뒤 멈춘다(웹캠 무한 루프 방지).
    """
    from src.io.frame_reader import CameraSpec, MultiViewFrameReader  # cv2 capture 필요해서 로컬 import

    specs = [CameraSpec(cam.name, getattr(cam, "source", idx) or idx)
             for idx, cam in enumerate(pipeline.cameras)]
    poses = []
    fps = float(pipeline.config.get("smoothing", {}).get("freq", 30.0))

    view = None
    if show:
        from src.render.live_view import LiveView  # cv2 GUI 필요해서 로컬 import
        view = LiveView(pipeline.cameras, score_thr=pipeline.score_threshold)

    try:
        with MultiViewFrameReader(specs) as reader:
            for frameset in reader:
                keypoints, scores = pipeline.detect_2d(frameset)
                # DepthFrameSource 백엔드가 연결돼 있으면 여기서 정렬된 depth맵을 넘겨준다.
                # 없으면 삼각측량만 돈다.
                pose = pipeline.process(keypoints, scores, depth_map=None,
                                        timestamp=frameset.index / fps)
                poses.append(pose)
                if view is not None and not view.show(frameset, keypoints, scores, pose):
                    break
                if max_frames is not None and len(poses) >= max_frames:
                    break
    finally:
        if view is not None:
            view.close()
    return poses


def main() -> None:
    args = _parse_args()
    pipeline = Pipeline.from_config(args.config)

    poses = run_live(pipeline, show=args.live, max_frames=args.max_frames)
    if not poses:
        print("[run] no frames processed.")
        return

    out_cfg = pipeline.config.get("output", {})
    fmt = out_cfg.get("format", "json")
    out_dir = Path(out_cfg.get("path", "output/"))
    out_path = Path(args.output) if args.output else out_dir / f"poses_3d.{fmt}"
    export_keypoints(poses, str(out_path), fmt=fmt)
    print(f"[run] exported {len(poses)} frame(s) -> {out_path}")

    if args.viz:
        png_path = out_dir / "skeleton_frame0.png"
        save_skeleton_png(poses[0], str(png_path))
        print(f"[run] saved visualization -> {png_path}")


if __name__ == "__main__":
    main()
