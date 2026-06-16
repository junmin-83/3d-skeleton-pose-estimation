"""Batch experiments for the 용역보고서 quantitative sections (EVALUATION_PLAN §4-6).

Blocks (select with --only, comma-separated; default: all):
  ablation_oracle  : §2 ablation under controlled 2D noise (ransac/smooth/views/noise)
  calib            : §4 calibration sensitivity (extrinsic rot/trans noise sweep)
  multiseq_oracle  : §1 cross-sequence mean ± 95% CI (oracle, needs only calib+GT)
  ablation_real    : §2 end-to-end ablation with real RTMPose (needs videos+rtmlib)
  multiseq_real    : §1 single real sequence, bootstrap CI + temporal-segment CI
  runtime          : §3 per-stage ms/frame + FPS

Each block writes output/experiments/<block>.json and prints a markdown table.
Oracle blocks are fast (no rtmlib/video); real blocks run RTMPose on CPU/GPU.

    uv run python examples/run_experiments.py --only ablation_oracle,calib,multiseq_oracle
    uv run python examples/run_experiments.py --only ablation_real,multiseq_real,runtime --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.panoptic_gt import available_gt_frames  # noqa: E402
from src.eval.runner import evaluate, load_cameras  # noqa: E402
from src.eval.stats import bootstrap_ci, mean_ci95_t  # noqa: E402

SEQ = "data/panoptic/171204_pose1"
CAMS = ["00_03", "00_12", "00_23"]
START = 500
OUTDIR = Path("output/experiments")


def _seqmean(rows, key):
    v = np.array([r[key] for r in rows if not np.isnan(r[key])], float)
    return float(v.mean()) if v.size else float("nan")


def _md_table(headers, rows):
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return f"{line}\n{sep}\n{body}"


def _save(name, payload):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{name}.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------- ablation_oracle
def block_ablation_oracle(num_frames=60):
    configs = [
        ("full (ransac+smooth)",  dict(ransac=True,  smoothing=True,  views=None, pixel_noise=2.0)),
        ("no RANSAC",             dict(ransac=False, smoothing=True,  views=None, pixel_noise=2.0)),
        ("no smoothing",          dict(ransac=True,  smoothing=False, views=None, pixel_noise=2.0)),
        ("no RANSAC, no smooth",  dict(ransac=False, smoothing=False, views=None, pixel_noise=2.0)),
        ("2 views",               dict(ransac=True,  smoothing=True,  views=2,    pixel_noise=2.0)),
        ("3 views",               dict(ransac=True,  smoothing=True,  views=3,    pixel_noise=2.0)),
        ("noise 0px",             dict(ransac=True,  smoothing=False, views=None, pixel_noise=0.0)),
        ("noise 1px",             dict(ransac=True,  smoothing=False, views=None, pixel_noise=1.0)),
        ("noise 2px",             dict(ransac=True,  smoothing=False, views=None, pixel_noise=2.0)),
        ("noise 4px",             dict(ransac=True,  smoothing=False, views=None, pixel_noise=4.0)),
    ]
    table, data = [], []
    for label, kw in configs:
        s, _ = evaluate(SEQ, CAMS, START, num_frames, mode="oracle", **kw)
        table.append([label, f"{s['abs_mpjpe_mm'][0]:.2f}", f"{s['pa_mpjpe_mm'][0]:.2f}",
                      f"{s['pck_50mm'][0]:.3f}", f"{s['reproj_rmse_px'][0]:.2f}"])
        data.append({"config": label, "kw": kw, "summary": s})
    md = "### §2 Ablation (oracle 2D, " + str(num_frames) + " frames)\n\n" + _md_table(
        ["config", "abs MPJPE (mm)", "PA-MPJPE (mm)", "PCK@50mm", "reproj (px)"], table)
    _save("ablation_oracle", data)
    return md


# ------------------------------------------------------------------------- calib
def block_calib(num_frames=60):
    rows, data = [], []
    for rot in [0.0, 0.25, 0.5, 1.0, 2.0]:
        s, _ = evaluate(SEQ, CAMS, START, num_frames, mode="oracle",
                        smoothing=False, pixel_noise=0.0, rot_noise_deg=rot)
        rows.append([f"rot {rot}°", f"{s['abs_mpjpe_mm'][0]:.2f}", f"{s['reproj_rmse_px'][0]:.2f}"])
        data.append({"rot_deg": rot, "summary": s})
    for tr in [0.0, 5.0, 10.0, 20.0, 50.0]:
        s, _ = evaluate(SEQ, CAMS, START, num_frames, mode="oracle",
                        smoothing=False, pixel_noise=0.0, trans_noise_mm=tr)
        rows.append([f"trans {tr}mm", f"{s['abs_mpjpe_mm'][0]:.2f}", f"{s['reproj_rmse_px'][0]:.2f}"])
        data.append({"trans_mm": tr, "summary": s})
    md = ("### §4 Calibration sensitivity (oracle, no pixel noise, " + str(num_frames) +
          " frames)\n\n" + _md_table(["perturbation", "abs MPJPE (mm)", "reproj (px)"], rows))
    _save("calib", data)
    return md


# --------------------------------------------------------------- multiseq_oracle
def _discover_sequences():
    seqs = []
    for d in sorted(Path("data/panoptic").glob("*")):
        gt = d / "hdPose3d_stage1_coco19"
        calib = list(d.glob("calibration_*.json"))
        if gt.is_dir() and calib:
            frames = available_gt_frames(str(gt))
            if frames:
                seqs.append((str(d), frames))
    return seqs


def block_multiseq_oracle(num_frames=60):
    seqs = _discover_sequences()
    per_seq, data = [], []
    for seq_dir, frames in seqs:
        start = frames[len(frames) // 4]  # avoid edges
        try:
            s, rows = evaluate(seq_dir, CAMS, start, num_frames, mode="oracle",
                               smoothing=False, pixel_noise=2.0)
        except (KeyError, StopIteration):
            continue
        if not rows:
            continue
        m = _seqmean(rows, "abs_mpjpe_mm")
        per_seq.append((Path(seq_dir).name, m, len(rows)))
        data.append({"seq": Path(seq_dir).name, "start": start, "abs_mpjpe_mm_mean": m,
                     "frames": len(rows)})
    means = [m for _, m, _ in per_seq]
    mean, half, std, n = mean_ci95_t(means)
    rows_md = [[name, f"{m:.2f}", fr] for name, m, fr in per_seq]
    rows_md.append(["**mean ± 95% CI**", f"**{mean:.2f} ± {half:.2f}**", f"n={n} seqs"])
    md = ("### §1 Cross-sequence accuracy (oracle, 2px noise, " + str(num_frames) +
          " frames/seq)\n\nPer-sequence mean abs MPJPE, aggregated across sequences "
          "(independent units):\n\n" +
          _md_table(["sequence", "abs MPJPE (mm)", "frames"], rows_md))
    _save("multiseq_oracle", {"per_seq": data, "ci": {"mean": mean, "half": half, "std": std, "n": n}})
    return md


# ----------------------------------------------------------------- ablation_real
def block_ablation_real(detector, num_frames=40):
    configs = [
        ("full (ransac+smooth)", dict(ransac=True,  smoothing=True,  views=None)),
        ("no RANSAC",            dict(ransac=False, smoothing=True,  views=None)),
        ("no smoothing",         dict(ransac=True,  smoothing=False, views=None)),
        ("2 views",              dict(ransac=True,  smoothing=True,  views=2)),
    ]
    table, data = [], []
    for label, kw in configs:
        s, _ = evaluate(SEQ, CAMS, START, num_frames, mode="real", detector=detector, **kw)
        table.append([label, f"{s['abs_mpjpe_mm'][0]:.2f}", f"{s['pa_mpjpe_mm'][0]:.2f}",
                      f"{s['rrel_mpjpe_mm'][0]:.2f}", f"{s['pck_50mm'][0]:.3f}",
                      f"{s['reproj_rmse_px'][0]:.2f}"])
        data.append({"config": label, "kw": kw, "summary": s})
    md = ("### §2 Ablation (real RTMPose end-to-end, " + str(num_frames) + " frames)\n\n" +
          _md_table(["config", "abs MPJPE (mm)", "PA-MPJPE (mm)", "root-rel (mm)",
                     "PCK@50mm", "reproj (px)"], table))
    _save("ablation_real", data)
    return md


# ----------------------------------------------------------------- multiseq_real
def block_multiseq_real(detector, num_frames=120):
    s, rows = evaluate(SEQ, CAMS, START, num_frames, mode="real", detector=detector,
                       ransac=True, smoothing=True)
    vals = [r["abs_mpjpe_mm"] for r in rows]
    bmean, blo, bhi, bn = bootstrap_ci(vals)
    # temporal-segment pseudo-sequences (EVALUATION_PLAN §4: frames autocorrelated)
    seg = max(1, len(rows) // 4)
    seg_means = [_seqmean(rows[i:i + seg], "abs_mpjpe_mm") for i in range(0, len(rows), seg)]
    smean, shalf, sstd, sn = mean_ci95_t(seg_means)
    md = ("### §1 Single real sequence — uncertainty (RTMPose, " + str(len(rows)) + " frames)\n\n" +
          _md_table(["estimator", "abs MPJPE (mm)", "95% interval", "n"],
                    [["frame bootstrap", f"{bmean:.2f}", f"[{blo:.2f}, {bhi:.2f}]", f"{bn} frames"],
                     ["segment t-interval", f"{smean:.2f}", f"± {shalf:.2f}", f"{sn} segments"]]) +
          "\n\n> Single sequence with video available; frame bootstrap is optimistic "
          "(autocorrelated). Cross-sequence CI requires more sequences' HD videos.")
    _save("multiseq_real", {"summary": s, "bootstrap": [bmean, blo, bhi, bn],
                            "segment_ci": [smean, shalf, sstd, sn]})
    return md


# ----------------------------------------------------------------------- runtime
def block_runtime(detector, device, iters=200, det_frames=20):

    from src.fusion.depth_fusion import back_project_depth_keypoints, fuse
    from src.smoothing.one_euro import PoseSmoother
    from src.triangulation.robust import triangulate_robust
    from src.eval.runner import project_distorted
    from src.eval.panoptic_gt import load_gt_frame
    from src.core.types import NUM_KEYPOINTS

    cams, names = load_cameras(SEQ, CAMS)
    proj = [c.P for c in cams]
    gt = load_gt_frame(str(Path(SEQ) / "hdPose3d_stage1_coco19"), START)
    kpts = np.stack([project_distorted(c, gt.points) for c in cams])
    scores = np.tile((gt.valid).astype(float), (len(cams), 1))

    def timeit(fn, n):
        fn()  # warm-up
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return 1000.0 * (time.perf_counter() - t0) / n

    t_tri = timeit(lambda: triangulate_robust(kpts, scores, proj, 0.0, 2, True, 15.0), iters)
    depth = np.full((1080, 1920), 3.0)
    dcam = cams[0]
    t_fuse = timeit(lambda: fuse(
        triangulate_robust(kpts, scores, proj, 0.0, 2, True, 15.0),
        *back_project_depth_keypoints(kpts[0], depth, dcam.K, dcam.R, dcam.t),
        scores[0]), iters)
    sm = PoseSmoother(NUM_KEYPOINTS, 30.0, 1.0, 0.01, 1.0)
    pose = triangulate_robust(kpts, scores, proj, 0.0, 2, True, 15.0)
    t_sm = timeit(lambda: sm.update(pose, 0.0), iters)

    # 2D detection (real): time detector over det_frames real HD frames
    t_det = float("nan")
    if detector is not None:
        from src.io.sources.panoptic import iter_panoptic_hd_frames
        it = iter_panoptic_hd_frames(Path(SEQ), names, START, det_frames)
        n = 0
        t0 = time.perf_counter()
        for frames in it:
            for img in frames:
                detector.detect_best(img)
            n += 1
        t_det = 1000.0 * (time.perf_counter() - t0) / max(n, 1)  # per frameset (all views)

    stages = [
        ("2D detection (RTMPose, all views)", t_det),
        ("triangulation (robust DLT+RANSAC)", t_tri),
        ("depth fusion (back-proj+fuse)", t_fuse),
        ("One-Euro smoothing", t_sm),
    ]
    geom_total = t_tri + t_fuse + t_sm
    total = (t_det if not np.isnan(t_det) else 0) + geom_total
    rows = [[name, f"{ms:.3f}"] for name, ms in stages]
    rows.append(["**geometry subtotal**", f"**{geom_total:.3f}**"])
    rows.append(["**end-to-end total**", f"**{total:.3f}**"])
    fps = 1000.0 / total if total > 0 else float("nan")
    md = (f"### §3 Runtime per stage (device={device}, {iters} iters; 2D over {det_frames} frames)\n\n"
          + _md_table(["stage", "ms / frame"], rows) +
          f"\n\n- End-to-end ≈ **{fps:.1f} FPS** ({device}). "
          "Geometry stages are device-independent (NumPy/SciPy); 2D detection dominates and "
          "is the stage GPU accelerates. README reports ~8× detection speedup on GPU "
          "(RTX 4050: CPU ~9.6 → GPU ~78 FPS) — re-measure on the target GPU for the report.")
    _save("runtime", {"stages": dict(stages), "geom_total": geom_total, "total": total,
                      "fps": fps, "device": device})
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", help="comma list of blocks or 'all'.")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="output/experiments/RESULTS_FRAGMENT.md")
    args = ap.parse_args()

    # Windows consoles default to cp949 here and choke on em-dashes in the
    # markdown; force UTF-8 so printing tables never crashes the run.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    all_blocks = ["ablation_oracle", "calib", "multiseq_oracle",
                  "ablation_real", "multiseq_real", "runtime"]
    blocks = all_blocks if args.only == "all" else [b.strip() for b in args.only.split(",")]
    needs_detector = any(b in blocks for b in ("ablation_real", "multiseq_real", "runtime"))

    detector = None
    if needs_detector:
        from src.pose2d.rtmpose_detector import RTMPoseDetector
        detector = RTMPoseDetector(device=args.device, mode="balanced", score_threshold=0.4)

    out_md = []
    for b in blocks:
        print(f"\n[run] block: {b} ...")
        if b == "ablation_oracle":
            md = block_ablation_oracle()
        elif b == "calib":
            md = block_calib()
        elif b == "multiseq_oracle":
            md = block_multiseq_oracle()
        elif b == "ablation_real":
            md = block_ablation_real(detector)
        elif b == "multiseq_real":
            md = block_multiseq_real(detector)
        elif b == "runtime":
            md = block_runtime(detector, args.device)
        else:
            print(f"[run] unknown block {b}, skipping")
            continue
        print(md)
        out_md.append(md)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n---\n\n".join(out_md), encoding="utf-8")
    print(f"\n[run] markdown fragment -> {out}")


if __name__ == "__main__":
    main()
