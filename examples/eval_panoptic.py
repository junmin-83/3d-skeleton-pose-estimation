"""CMU Panoptic 3D-accuracy evaluation driver (용역보고서 정량평가용).

Thin CLI over src/eval/runner.evaluate(). Computes the EVALUATION_PLAN metrics
(Absolute / Root-relative / PA-MPJPE, PCK3D/AUC, valid rate, reprojection RMSE)
over Panoptic frames vs hdPose3d_stage1_coco19 GT, in two input modes that
separate the borrowed 2D detector from the system's own geometry (§3-1):

  --mode oracle : project GT 3D into each camera as GT 2D, then run the geometry
                  (no rtmlib/video). --pixel-noise adds Gaussian px noise.
  --mode real   : run RTMPose on the HD videos (needs rtmlib + .mp4), end-to-end.

Ablation: --no-ransac, --no-smoothing, --views. Calibration sensitivity:
--rot-noise-deg, --trans-noise-mm. Batch experiments live in examples/run_*.py.

Example::

    uv run python examples/eval_panoptic.py --seq-dir data/panoptic/171204_pose1 \
        --cams 00_03,00_12,00_23 --start 500 --num-frames 60 --mode oracle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.runner import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Panoptic 3D pose accuracy evaluation.")
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--cams", default="00_03,00_12,00_23")
    ap.add_argument("--start", type=int, default=500)
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--mode", choices=["oracle", "real"], default="oracle")
    ap.add_argument("--views", type=int, default=None, help="use first N cameras (2 vs 3 ablation).")
    ap.add_argument("--pixel-noise", type=float, default=0.0, help="oracle: Gaussian px noise std.")
    ap.add_argument("--no-ransac", action="store_true")
    ap.add_argument("--no-smoothing", action="store_true")
    ap.add_argument("--rot-noise-deg", type=float, default=0.0, help="extrinsic rotation noise (deg).")
    ap.add_argument("--trans-noise-mm", type=float, default=0.0, help="extrinsic translation noise (mm).")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--mode-detector", default="balanced")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--gt-offset", type=int, default=0, help="HD frame -> GT file index offset.")
    ap.add_argument("--out", default="output/panoptic_eval.json")
    args = ap.parse_args()

    names = [n.strip() for n in args.cams.split(",")]
    detector = None
    if args.mode == "real":
        from src.pose2d.rtmpose_detector import RTMPoseDetector
        sel = names[: args.views] if args.views else names
        detector = RTMPoseDetector(device=args.device, mode=args.mode_detector, score_threshold=0.4)
        _ = sel  # detector is view-agnostic; frame_iter built inside evaluate()

    summary, rows = evaluate(
        seq_dir=args.seq_dir, cam_names=names, start=args.start, num_frames=args.num_frames,
        mode=args.mode, views=args.views, pixel_noise=args.pixel_noise,
        ransac=not args.no_ransac, smoothing=not args.no_smoothing,
        rot_noise_deg=args.rot_noise_deg, trans_noise_mm=args.trans_noise_mm,
        fps=args.fps, gt_offset=args.gt_offset, detector=detector,
    )

    if not rows:
        print("[eval] no frames scored (check --start/--num-frames vs GT range).")
        sys.exit(1)

    n_views = len(names[: args.views]) if args.views else len(names)
    print(f"\n=== Panoptic eval | mode={args.mode} views={n_views} "
          f"ransac={not args.no_ransac} smooth={not args.no_smoothing} "
          f"noise={args.pixel_noise}px rot={args.rot_noise_deg}deg trans={args.trans_noise_mm}mm "
          f"| {len(rows)} frames ===")
    for k, (mean, std, n) in summary.items():
        print(f"  {k:16s}: {mean:8.2f} +/- {std:6.2f}  (n={n})")

    meta = {"mode": args.mode, "views": n_views, "cams": names,
            "ransac": not args.no_ransac, "smoothing": not args.no_smoothing,
            "pixel_noise": args.pixel_noise, "rot_noise_deg": args.rot_noise_deg,
            "trans_noise_mm": args.trans_noise_mm, "frames_scored": len(rows),
            "start": args.start, "num_frames": args.num_frames}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "summary": summary, "frames": rows}, indent=2))
    print(f"[eval] per-frame + summary -> {out}")


if __name__ == "__main__":
    main()
