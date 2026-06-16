"""Generate report figures (docs/figures/*.png) for the 용역보고서.

Figure groups (--figs, comma-separated; default all):
  charts : from output/experiments/*.json (fast) — noise/views/calibration/CI plots
  qual   : real RTMPose pass — predicted vs GT 3D skeleton overlay (one frame)
  jitter : real RTMPose pass — one joint's coordinate, raw vs One-Euro smoothed

'charts' is instant; 'qual'/'jitter' run one real detection pass (needs videos+rtmlib).

    uv run python examples/make_report_figures.py --figs charts
    uv run python examples/make_report_figures.py --figs qual,jitter --device cuda
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

# Korean labels need a CJK-capable font; Malgun Gothic ships with Windows.
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.types import COCO_17_KEYPOINTS, COCO_SKELETON, NUM_KEYPOINTS  # noqa: E402
from src.eval.panoptic_gt import load_gt_frame  # noqa: E402
from src.eval.runner import _real_2d, load_cameras, make_pipeline, project_distorted  # noqa: E402
from src.render.skeleton_2d import draw_skeleton_2d, label_panel  # noqa: E402
from src.smoothing.one_euro import PoseSmoother  # noqa: E402

SEQ = "data/panoptic/171204_pose1"
CAMS = ["00_03", "00_12", "00_23"]
START = 500
EXP = Path("output/experiments")
FIGDIR = Path("docs/figures")


def _load(name):
    return json.loads((EXP / f"{name}.json").read_text())


def fig_pipeline():
    """Vector pipeline flowchart (§2.1) — saved as both SVG and PNG.

    Color-codes each stage by ownership (차용 / 고유 / 고유 핵심); Korean text is
    emitted as vector paths in the SVG so it renders without font dependencies.
    """
    from matplotlib.patches import FancyBboxPatch, Patch

    stages = [
        ("입력: RGB 2대 + RGB-D 1대 (동기화 프레임)", "", "input"),
        ("① 2D 키포인트 검출 (RTMPose, COCO-17)", "src/pose2d/rtmpose_detector.py", "borrow"),
        ("② 렌즈 왜곡 보정 (undistort)", "src/pipeline.py", "own"),
        ("③ 신뢰도 가중 DLT 삼각측량 (+RANSAC robust)", "src/triangulation/{dlt,robust}.py", "core"),
        ("④ depth back-projection 융합", "src/fusion/depth_fusion.py", "core"),
        ("⑤ One-Euro 시간적 스무딩", "src/smoothing/one_euro.py", "own"),
        ("출력: 3D 스켈레톤 (world 좌표, meter, COCO-17)", "", "output"),
    ]
    fill = {"input": "#e9e9e9", "borrow": "#f3b06b", "own": "#9ecae1",
            "core": "#2f6fb0", "output": "#bfe3bd"}
    tag = {"borrow": "차용", "own": "고유", "core": "고유 핵심"}

    n, H, gap, W, x0 = len(stages), 1.0, 0.55, 8.8, 0.6
    prev = plt.rcParams.get("svg.fonttype")
    plt.rcParams["svg.fonttype"] = "path"  # Korean -> vector paths in the SVG
    fig, ax = plt.subplots(figsize=(7.4, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n * (H + gap) + 0.3)
    ax.axis("off")

    for i, (title, mod, kind) in enumerate(stages):
        y = (n - 1 - i) * (H + gap) + 0.3
        ax.add_patch(FancyBboxPatch(
            (x0, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.5, edgecolor="#333333", facecolor=fill[kind]))
        tc = "white" if kind == "core" else "#111111"
        ax.text(x0 + 0.28, y + H * (0.62 if mod else 0.5), title,
                fontsize=11.5, va="center", ha="left", color=tc, fontweight="bold")
        if mod:
            ax.text(x0 + 0.28, y + H * 0.25, mod, fontsize=8.5, va="center",
                    ha="left", color=tc, family="monospace")
        if kind in tag:
            ax.text(x0 + W - 0.18, y + H * 0.5, f"[{tag[kind]}]", fontsize=9.5,
                    va="center", ha="right", color=tc, style="italic")
        if i < n - 1:
            cx = x0 + W / 2
            ax.annotate("", xy=(cx, y - gap + 0.04), xytext=(cx, y - 0.04),
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333333"))

    legend = [Patch(facecolor=fill["borrow"], edgecolor="#333", label="차용 모델 (사전학습)"),
              Patch(facecolor=fill["own"], edgecolor="#333", label="시스템 고유"),
              Patch(facecolor=fill["core"], edgecolor="#333", label="시스템 고유 핵심")]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=9, frameon=False)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGDIR / "fig_pipeline.svg"))
    fig.savefig(str(FIGDIR / "fig_pipeline.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams["svg.fonttype"] = prev or "path"
    print("[fig] -> docs/figures/fig_pipeline.svg + .png")


# --------------------------------------------------------------------- charts
def fig_noise_and_views():
    data = _load("ablation_oracle")
    by = {d["config"]: d["summary"] for d in data}
    noise_x = [0, 1, 2, 4]
    noise_y = [by[f"noise {n}px"]["abs_mpjpe_mm"][0] for n in noise_x]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(noise_x, noise_y, "o-", color="#1f77b4", lw=2)
    ax1.set_xlabel("2D pixel noise (px)")
    ax1.set_ylabel("abs MPJPE (mm)")
    ax1.set_title("(a) 2D noise → 3D error (oracle)")
    ax1.grid(alpha=0.3)

    views = ["2 views", "3 views"]
    vy = [by[v]["abs_mpjpe_mm"][0] for v in views]
    ax2.bar(views, vy, color=["#ff7f0e", "#2ca02c"], width=0.5)
    for i, v in enumerate(vy):
        ax2.text(i, v + 0.1, f"{v:.2f}", ha="center")
    ax2.set_ylabel("abs MPJPE (mm)")
    ax2.set_title("(b) views (oracle, 2px noise)")
    fig.tight_layout()
    _save(fig, "fig_noise_views.png")


def fig_calibration():
    data = _load("calib")
    rot = [(d["rot_deg"], d["summary"]["abs_mpjpe_mm"][0]) for d in data if "rot_deg" in d]
    tr = [(d["trans_mm"], d["summary"]["abs_mpjpe_mm"][0]) for d in data if "trans_mm" in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot([x for x, _ in rot], [y for _, y in rot], "o-", color="#d62728", lw=2)
    ax1.set_xlabel("extrinsic rotation noise (deg)")
    ax1.set_ylabel("abs MPJPE (mm)")
    ax1.set_title("(a) rotation sensitivity")
    ax1.grid(alpha=0.3)
    ax2.plot([x for x, _ in tr], [y for _, y in tr], "s-", color="#9467bd", lw=2)
    ax2.set_xlabel("extrinsic translation noise (mm)")
    ax2.set_ylabel("abs MPJPE (mm)")
    ax2.set_title("(b) translation sensitivity")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_calibration.png")


def fig_multiseq():
    data = _load("multiseq_oracle")
    seqs = [d["seq"] for d in data["per_seq"]]
    vals = [d["abs_mpjpe_mm_mean"] for d in data["per_seq"]]
    ci = data["ci"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(seqs, vals, color="#17becf", width=0.55)
    ax.axhline(ci["mean"], color="k", ls="--", lw=1,
               label=f"mean {ci['mean']:.2f} ± {ci['half']:.2f} (95% CI)")
    ax.fill_between([-0.5, len(seqs) - 0.5], ci["mean"] - ci["half"], ci["mean"] + ci["half"],
                    color="k", alpha=0.08)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("abs MPJPE (mm)")
    ax.set_title("Cross-sequence accuracy (oracle, 2px noise, n=4)")
    ax.set_xticklabels(seqs, rotation=15, ha="right")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig_multiseq.png")


def fig_ablation_real():
    """Two aligned horizontal-bar panels: abs MPJPE (lower=better) and PCK@50mm
    (higher=better). The 2-view config is highlighted to carry the main story."""
    data = _load("ablation_real")
    labels = [d["config"] for d in data]
    abs_v = [d["summary"]["abs_mpjpe_mm"][0] for d in data]
    pck = [d["summary"]["pck_50mm"][0] for d in data]
    y = np.arange(len(labels))[::-1]
    colors = ["#d62728" if "view" in lab else "#4c78a8" for lab in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    ax1.barh(y, abs_v, color=colors)
    for yi, v in zip(y, abs_v):
        ax1.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=10)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("abs MPJPE (mm)  ← 낮을수록 우수")
    ax1.set_xlim(0, max(abs_v) * 1.18)
    ax1.set_title("(a) 절대정확도")

    ax2.barh(y, pck, color=colors)
    for yi, v in zip(y, pck):
        ax2.text(v - 0.02, yi, f"{v:.3f}", va="center", ha="right",
                 color="white", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("PCK@50mm  →  높을수록 우수")
    ax2.set_title("(b) 50mm 이내 관절 비율")

    fig.suptitle("End-to-end ablation (real RTMPose) — 빨강: 2뷰", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_ablation_real.png")


# ---------------------------------------------------------------- real pass
def _real_trajectories(detector, num_frames, device):
    """Run real detection once; return list of (raw_pose, gt_pose) over frames."""
    from src.io.sources.panoptic import iter_panoptic_hd_frames
    cams, names = load_cameras(SEQ, CAMS)
    pipe = make_pipeline(cams, "real", ransac=True, smoothing=False, fps=30.0)
    gt_dir = str(Path(SEQ) / "hdPose3d_stage1_coco19")
    it = iter_panoptic_hd_frames(Path(SEQ), names, START, num_frames)
    out = []
    for f, frames in enumerate(it):
        gt = load_gt_frame(gt_dir, START + f)
        if gt is None:
            continue
        kpts, scores = _real_2d(detector, frames)
        raw = pipe.process(kpts, scores, depth_map=None, timestamp=(START + f) / 30.0)
        out.append((raw, gt))
    return out


def _capture_frame(detector, offset):
    """Decode one HD frameset at START+offset, detect 2D, reconstruct 3D, load GT.

    Returns (frames[list BGR], kpts (V,17,2), scores (V,17), pred Pose3D, gt Pose3D,
    cams, names).
    """
    from src.io.sources.panoptic import iter_panoptic_hd_frames
    cams, names = load_cameras(SEQ, CAMS)
    pipe = make_pipeline(cams, "real", ransac=True, smoothing=False, fps=30.0)
    gt_dir = str(Path(SEQ) / "hdPose3d_stage1_coco19")
    it = iter_panoptic_hd_frames(Path(SEQ), names, START, offset + 1)
    frames = None
    for f, fr in enumerate(it):
        frames = fr
        if f >= offset:
            break
    for _ in it:
        pass
    kpts, scores = _real_2d(detector, frames)
    pred = pipe.process(kpts, scores, depth_map=None, timestamp=(START + offset) / 30.0)
    gt = load_gt_frame(gt_dir, START + offset)
    return frames, kpts, scores, pred, gt, cams, names


def _scan_front_frame(detector, span=300, step=6):
    """Scan a frame range and return the absolute frame index whose face is most
    front-facing (lowest reconstructed nose+eye error vs GT, full body valid).

    The default qual frame is back-facing (subject away from cam), so face joints
    err high. A front-facing frame makes the face well-localised — a contrasting
    'good' qualitative case.
    """
    from src.io.sources.panoptic import iter_panoptic_hd_frames
    cams, names = load_cameras(SEQ, CAMS)
    pipe = make_pipeline(cams, "real", ransac=True, smoothing=False, fps=30.0)
    gt_dir = str(Path(SEQ) / "hdPose3d_stage1_coco19")
    face = [0, 1, 2]  # nose, left_eye, right_eye
    best_frame, best_err = None, float("inf")
    for f, frames in enumerate(iter_panoptic_hd_frames(Path(SEQ), names, START, span)):
        if f % step:
            continue
        gt = load_gt_frame(gt_dir, START + f)
        if gt is None:
            continue
        kpts, scores = _real_2d(detector, frames)
        pred = pipe.process(kpts, scores, depth_map=None, timestamp=(START + f) / 30.0)
        m = pred.valid & gt.valid
        if not m.all():
            continue
        err = np.linalg.norm((pred.points - gt.points), axis=1) * 1000
        face_err = float(np.mean(err[face]))
        if face_err > 25.0:        # require a genuinely front-facing head
            continue
        total = float(np.mean(err))  # then pick the cleanest overall frame
        if total < best_err:
            best_err, best_frame = total, START + f
    if best_frame is not None:
        print(f"[front] best front frame {best_frame}, total error {best_err:.1f} mm")
    return best_frame


def _person_bbox(uv, mask, shape, pad=0.35):
    """Padded (x0,y0,x1,y1) around the valid points uv within image shape."""
    pts = uv[mask]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    x0 = int(max(0, x0 - pad * w))
    x1 = int(min(shape[1], x1 + pad * w))
    y0 = int(max(0, y0 - pad * h))
    y1 = int(min(shape[0], y1 + pad * h))
    return x0, y0, x1, y1


def fig_multiview_capture(cap, name="fig_multiview_capture.png"):
    """Real HD frames (3 views) with the detected COCO-17 skeleton overlaid,
    cropped to the subject. The 'video capture -> COCO-17 prediction' visual."""
    frames, kpts, scores, _, _, _, names = cap
    panels, target_h = [], 460
    for v, (img, n) in enumerate(zip(frames, names)):
        over = draw_skeleton_2d(img.copy(), kpts[v], scores[v], 0.3,
                                line_thickness=3, point_radius=6)
        mask = scores[v] >= 0.3
        if mask.sum() >= 2:
            x0, y0, x1, y1 = _person_bbox(kpts[v], mask, img.shape)
            over = over[y0:y1, x0:x1]
        s = target_h / over.shape[0]
        over = cv2.resize(over, (int(over.shape[1] * s), target_h))
        panels.append(label_panel(over, f"HD {n}", over.shape[1], font_scale=0.6))
    sep = np.full((target_h, 10, 3), 255, np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    _imsave(out, name)


def _draw_uv_skeleton(img, uv, valid, color, thick=2, rad=4):
    pix = np.round(uv).astype(int)
    for i, j in COCO_SKELETON:
        if valid[i] and valid[j]:
            cv2.line(img, tuple(pix[i]), tuple(pix[j]), color, thick)
    for k in range(len(pix)):
        if valid[k]:
            cv2.circle(img, tuple(pix[k]), rad, color, -1)
    return img


def fig_reproj_on_image(cap, name="fig_reproj_on_image.png", view=0):
    """Predicted 3D and GT 3D reprojected onto a real HD view, over the actual
    frame — shows reconstruction accuracy in intuitive image space."""
    frames, kpts, scores, pred, gt, cams, names = cap
    img = frames[view].copy()
    uv_pred = project_distorted(cams[view], pred.points)
    uv_gt = project_distorted(cams[view], gt.points)
    _draw_uv_skeleton(img, uv_gt, gt.valid, (0, 200, 0), thick=2, rad=5)        # GT green
    _draw_uv_skeleton(img, uv_pred, pred.valid, (0, 0, 255), thick=2, rad=4)    # pred red
    mask = gt.valid
    if mask.sum() >= 2:
        x0, y0, x1, y1 = _person_bbox(uv_gt, mask, img.shape, pad=0.4)
        img = img[y0:y1, x0:x1]
    s = 560 / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * s), 560))
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, "GT=green  pred=red", (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    _imsave(img, name)


def _imsave(img, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGDIR / name), img)
    print(f"[fig] -> docs/figures/{name}")


def fig_ortho_planes(cap, **kwargs):
    """Two clean spatial overlaps (front Z–Y, side X–Y) + per-joint error bars.

    The top-down (X–Z) plane is intentionally omitted: for a standing subject it
    projects every joint onto a small, tangled horizontal footprint dominated by
    the weakly-constrained depth axis, so it reads as 'mismatch' even though the
    3D error is roughly uniform (tens of mm) in every direction. A per-joint
    error bar conveys that third dimension quantitatively and honestly instead.
    """
    name = kwargs.get("name", "fig_ortho_planes.png")
    pred, gt = cap[3], cap[4]
    m = pred.valid & gt.valid
    err_mm = np.linalg.norm(pred.points - gt.points, axis=1) * 1000
    mean_err = float(np.mean(err_mm[m]))

    def draw(ax, pts, valid, color, label, a, b):
        for i, j in COCO_SKELETON:
            if valid[i] and valid[j]:
                ax.plot([pts[i, a], pts[j, a]], [pts[i, b], pts[j, b]], c=color, lw=2.0)
        ax.scatter(pts[valid, a], pts[valid, b], c=color, s=18, label=label, zorder=5)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (title, a, b) in zip(axes[:2], [("정면 (Z–Y)", 2, 1), ("측면 (X–Y)", 0, 1)]):
        draw(ax, gt.points, gt.valid, "#2ca02c", "GT", a, b)
        draw(ax, pred.points, pred.valid, "#d62728", "예측(real)", a, b)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.invert_yaxis()
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 3: per-joint error (mm) — the quantitative 'third dimension'.
    ax = axes[2]
    idx = np.where(m)[0]
    names = [COCO_17_KEYPOINTS[i].replace("left", "L").replace("right", "R") for i in idx]
    vals = err_mm[idx]
    y = np.arange(len(idx))[::-1]
    colors = ["#2ca02c" if v <= 30 else "#e08214" if v <= 60 else "#d62728" for v in vals]
    ax.barh(y, vals, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v:.0f}", va="center", fontsize=7)
    ax.axvline(mean_err, color="k", ls="--", lw=1, label=f"평균 {mean_err:.1f} mm")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("관절별 오차 (mm)")
    ax.set_title("관절별 오차")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, max(vals) * 1.2)

    fig.suptitle(f"예측 vs GT 3D — 정면·측면 overlay + 관절별 오차 (mean {mean_err:.1f} mm)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, name)


def fig_jitter(traj):
    joint = COCO_17_KEYPOINTS.index("right_wrist")
    raw_traj = np.array([p.points[joint] for p, _ in traj])
    valid = np.array([bool(p.valid[joint]) for p, _ in traj])
    # offline One-Euro on the raw poses (re-run smoother on raw sequence)
    sm = PoseSmoother(NUM_KEYPOINTS, 30.0, 1.0, 0.01, 1.0)
    sm_traj = []
    for k, (raw, _) in enumerate(traj):
        sm_traj.append(sm.update(raw, (START + k) / 30.0).points[joint])
    sm_traj = np.array(sm_traj)

    axis = int(np.nanargmax(np.nanstd(raw_traj[valid], axis=0)))  # most variable axis
    t = np.arange(len(traj))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t[valid], raw_traj[valid, axis], "o-", ms=3, color="#888", label="raw (no smoothing)")
    ax.plot(t[valid], sm_traj[valid, axis], "-", lw=2, color="#1f77b4", label="One-Euro smoothed")
    ax.set_xlabel("frame")
    ax.set_ylabel(f"right wrist {'XYZ'[axis]} (m)")
    ax.set_title("Temporal jitter: One-Euro smoothing effect (real)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_jitter.png")


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGDIR / name), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> docs/figures/{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default="charts,qual,jitter")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--num-frames", type=int, default=90)
    ap.add_argument("--qual-offset", type=int, default=40,
                    help="frame offset from START for the image-space figures.")
    args = ap.parse_args()
    figs = [f.strip() for f in args.figs.split(",")]

    if "charts" in figs:
        fig_pipeline()
        fig_noise_and_views()
        fig_calibration()
        fig_multiseq()
        fig_ablation_real()

    need_real = any(f in figs for f in ("qual", "jitter", "capture", "front"))
    if need_real:
        from src.pose2d.rtmpose_detector import RTMPoseDetector
        detector = RTMPoseDetector(device=args.device, mode="balanced", score_threshold=0.4)
        if "jitter" in figs:
            traj = _real_trajectories(detector, args.num_frames, args.device)
            fig_jitter(traj)
        if "qual" in figs or "capture" in figs:
            # one representative, well-visible frame for the image-space figures
            cap = _capture_frame(detector, args.qual_offset)
            fig_multiview_capture(cap)
            fig_reproj_on_image(cap)
            fig_ortho_planes(cap)
        if "front" in figs:
            off = _scan_front_frame(detector)
            if off is None:
                print("[front] no front-facing frame found in scan range.")
            else:
                cap = _capture_frame(detector, off - START)
                face_view = int(np.argmax(cap[2][:, [0, 1, 2]].sum(axis=1)))
                print(f"[front] frame {off}, best face-view {cap[6][face_view]}")
                fig_ortho_planes(cap, name="fig_ortho_planes_front.png")
                fig_reproj_on_image(cap, name="fig_reproj_front.png", view=face_view)


if __name__ == "__main__":
    main()
