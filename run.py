"""Entry point for the 3D skeleton pose estimation pipeline (live / recorded).

Runs RTMPose 2D detection on each configured camera view, triangulates to 3D
(+ depth fusion when a depth source is wired into the pipeline), applies One-Euro
smoothing, and exports per-frame 3D keypoints. Requires rtmlib + onnxruntime and
the camera sources / calibration configured in ``config/cameras.yaml``.

    uv run python run.py --config config/cameras.yaml
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
    return parser.parse_args()


def run_live(pipeline: Pipeline) -> list:
    """Read synchronized frames from the configured sources and reconstruct 3D."""
    from src.io.frame_reader import CameraSpec, MultiViewFrameReader  # local: needs cv2 capture

    specs = [CameraSpec(cam.name, getattr(cam, "source", idx) or idx)
             for idx, cam in enumerate(pipeline.cameras)]
    poses = []
    fps = float(pipeline.config.get("smoothing", {}).get("freq", 30.0))
    with MultiViewFrameReader(specs) as reader:
        for frameset in reader:
            keypoints, scores = pipeline.detect_2d(frameset)
            # A wired DepthFrameSource backend would supply the aligned depth map
            # here; without one this runs triangulation-only.
            pose = pipeline.process(keypoints, scores, depth_map=None,
                                    timestamp=frameset.index / fps)
            poses.append(pose)
    return poses


def main() -> None:
    args = _parse_args()
    pipeline = Pipeline.from_config(args.config)

    poses = run_live(pipeline)
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
