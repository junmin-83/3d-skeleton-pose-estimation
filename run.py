"""Entry point for the 3D skeleton pose estimation pipeline.

Examples
--------
Offline end-to-end demo (no rtmlib / GPU / cameras needed)::

    uv run python run.py --config config/cameras.yaml --synthetic --frames 30 --viz

Live / recorded run (requires rtmlib + onnxruntime and configured sources)::

    uv run python run.py --config config/cameras.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.pipeline import Pipeline
from src.synthetic import synthesize_depth_map, synthesize_observations, synthesize_sequence
from src.viz.visualize_3d import export_keypoints, save_skeleton_png


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D skeleton pose estimation pipeline.")
    parser.add_argument("--config", default="config/cameras.yaml", help="camera/pipeline config YAML.")
    parser.add_argument("--synthetic", action="store_true",
                        help="run the offline synthetic demo (no rtmlib/GPU/cameras).")
    parser.add_argument("--frames", type=int, default=1, help="number of synthetic frames.")
    parser.add_argument("--output", default=None, help="override output file path.")
    parser.add_argument("--viz", action="store_true", help="save a PNG of the first frame's skeleton.")
    return parser.parse_args()


def run_synthetic(pipeline: Pipeline, num_frames: int) -> list:
    """Project a known skeleton into the configured rig and reconstruct it."""
    depth_cam = pipeline.cameras[pipeline.depth_idx] if pipeline.depth_idx is not None else None
    poses, truths = [], []
    fps = float(pipeline.config.get("smoothing", {}).get("freq", 30.0))
    seed = pipeline.config.get("seed")
    for frame_idx, skeleton in enumerate(synthesize_sequence(num_frames, seed=seed)):
        keypoints, scores = synthesize_observations(skeleton, pipeline.cameras)
        depth_map = synthesize_depth_map(skeleton, depth_cam) if depth_cam is not None else None
        pose = pipeline.process(keypoints, scores, depth_map, timestamp=frame_idx / fps)
        poses.append(pose)
        truths.append(skeleton)
    # Report reconstruction accuracy on the first frame (smoother passes it through raw).
    if poses:
        err = np.linalg.norm(poses[0].points - truths[0], axis=1)
        valid = poses[0].valid
        print(f"[synthetic] frames={len(poses)} keypoints={valid.size} "
              f"valid(frame0)={int(valid.sum())}/{valid.size} "
              f"mean3D_err(frame0)={float(err[valid].mean()):.6e} m")
    return poses


def run_live(pipeline: Pipeline) -> list:
    """Live/recorded run. Requires rtmlib + configured frame/depth sources."""
    from src.io.frame_reader import CameraSpec, MultiViewFrameReader  # local: needs cv2 capture

    specs = [CameraSpec(cam.name, getattr(cam, "source", idx) or idx)
             for idx, cam in enumerate(pipeline.cameras)]
    poses = []
    fps = float(pipeline.config.get("smoothing", {}).get("freq", 30.0))
    with MultiViewFrameReader(specs) as reader:
        for frameset in reader:
            keypoints, scores = pipeline.detect_2d(frameset)
            # Depth acquisition is SDK-specific; a real DepthFrameSource backend
            # would supply the aligned depth map here. Without one we run
            # triangulation-only.
            pose = pipeline.process(keypoints, scores, depth_map=None,
                                    timestamp=frameset.index / fps)
            poses.append(pose)
    return poses


def main() -> None:
    args = _parse_args()
    pipeline = Pipeline.from_config(args.config)

    seed = pipeline.config.get("seed")
    if seed is not None:
        np.random.seed(int(seed))  # reproducibility (SPEC §2)

    poses = run_synthetic(pipeline, args.frames) if args.synthetic else run_live(pipeline)
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
