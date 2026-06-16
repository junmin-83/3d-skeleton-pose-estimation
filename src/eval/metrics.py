"""3D pose accuracy metrics (MPJPE family, PCK3D, AUC).

Everything works on COCO-17 arrays in the world frame, meters (the pipeline's
native units; see core/types.py). Metrics take a per-joint ``valid`` mask so that
only joints reconstructed by the system AND present in the GT are scored, and
coverage (valid rate) is reported separately rather than mixed into accuracy
(docs/EVALUATION_PLAN.md A5: hiding coverage inflates accuracy).

Alignment protocols (EVALUATION_PLAN §3-3), each a different question:
  - mpjpe            : no alignment           -> absolute accuracy (A1)
  - root_relative    : subtract pelvis        -> joint layout, global pos removed (A2)
  - pa_mpjpe         : Procrustes (R, t, s)   -> structural accuracy (A3)

Distances come back in meters; multiply by 1000 for the mm figures the report
quotes.
"""

from __future__ import annotations

import numpy as np

# COCO-17 hip indices; pelvis (root) is their midpoint since COCO-17 has no
# dedicated pelvis joint (EVALUATION_PLAN appendix A interpolation rule).
_LEFT_HIP = 11
_RIGHT_HIP = 12


def pelvis_from_coco17(points: np.ndarray) -> np.ndarray:
    """Pelvis (root) as the mid-hip of a COCO-17 array: (left_hip+right_hip)/2.

    Args:
        points: (17, 3) world coords, meters.

    Returns:
        (3,) pelvis point, meters.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return 0.5 * (pts[_LEFT_HIP] + pts[_RIGHT_HIP])


def _common_valid(
    pred_valid: np.ndarray | None,
    gt_valid: np.ndarray | None,
    n: int,
) -> np.ndarray:
    """Joints scored only where both prediction and GT are valid."""
    mask = np.ones(n, dtype=bool)
    if pred_valid is not None:
        mask &= np.asarray(pred_valid, dtype=bool).reshape(-1)
    if gt_valid is not None:
        mask &= np.asarray(gt_valid, dtype=bool).reshape(-1)
    return mask


def per_joint_error(
    pred: np.ndarray,
    gt: np.ndarray,
) -> np.ndarray:
    """Per-joint Euclidean distance (K,) in input units (meters)."""
    pred = np.asarray(pred, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt, dtype=float).reshape(-1, 3)
    return np.linalg.norm(pred - gt, axis=1)


def mpjpe(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_valid: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
) -> float:
    """Absolute MPJPE (no alignment), meters. NaN if no jointly-valid joint.

    Mean over jointly-valid joints of the Euclidean prediction-GT distance.
    This is the absolute-accuracy metric (A1): keep predictions and GT in the
    same world frame with NO per-frame alignment.
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt, dtype=float).reshape(-1, 3)
    mask = _common_valid(pred_valid, gt_valid, pred.shape[0])
    if not mask.any():
        return float("nan")
    return float(np.mean(per_joint_error(pred[mask], gt[mask])))


def root_relative_mpjpe(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_valid: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
    pred_root: np.ndarray | None = None,
    gt_root: np.ndarray | None = None,
) -> float:
    """Root-relative MPJPE (A2): subtract each pose's pelvis, then MPJPE, meters.

    Removes global translation so only joint layout is scored. By default the
    pelvis is the COCO-17 mid-hip of each pose; pass pred_root/gt_root to use an
    external root (e.g. GT BodyCenter). The pelvis itself is excluded from the
    average (it is ~0 by construction and would dilute the error).
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt, dtype=float).reshape(-1, 3)
    p_root = pelvis_from_coco17(pred) if pred_root is None else np.asarray(pred_root, float).reshape(3)
    g_root = pelvis_from_coco17(gt) if gt_root is None else np.asarray(gt_root, float).reshape(3)

    mask = _common_valid(pred_valid, gt_valid, pred.shape[0])
    # Exclude the two hip joints from the scored set: they define the root, so
    # scoring them flatters the metric.
    mask[_LEFT_HIP] = False
    mask[_RIGHT_HIP] = False
    if not mask.any():
        return float("nan")
    return float(np.mean(per_joint_error(pred[mask] - p_root, gt[mask] - g_root)))


def procrustes_align(
    pred: np.ndarray,
    gt: np.ndarray,
) -> np.ndarray:
    """Umeyama similarity (rotation+uniform scale+translation) of pred onto gt.

    Solves min over (s, R, t) of || s R pred + t - gt ||^2 (with proper rotation,
    det(R)=+1) and returns the transformed prediction. Used for PA-MPJPE (A3),
    which removes the global similarity so only pose *shape* is scored. Caller
    must pass only the jointly-valid joints (>= 3 for a meaningful fit).

    Args:
        pred, gt: (N, 3) corresponding points, same order.

    Returns:
        (N, 3) aligned prediction.
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt, dtype=float).reshape(-1, 3)
    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    p0 = pred - mu_p
    g0 = gt - mu_g

    cov = g0.T @ p0 / pred.shape[0]
    u, d, vt = np.linalg.svd(cov)
    # Reflection guard: force a proper rotation (det = +1).
    s = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        s[-1] = -1.0
    rot = u @ np.diag(s) @ vt

    var_p = (p0 ** 2).sum() / pred.shape[0]
    scale = (d * s).sum() / var_p if var_p > 0 else 1.0
    trans = mu_g - scale * rot @ mu_p
    return (scale * (pred @ rot.T)) + trans


def pa_mpjpe(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_valid: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
) -> float:
    """Procrustes-aligned MPJPE (A3 / Protocol #2), meters.

    Align the jointly-valid predicted joints onto GT with a similarity transform,
    then take the MPJPE. Removes calibration/scale/orientation error, isolating
    structural accuracy. NaN if fewer than 3 jointly-valid joints (under-determined).
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt, dtype=float).reshape(-1, 3)
    mask = _common_valid(pred_valid, gt_valid, pred.shape[0])
    if mask.sum() < 3:
        return float("nan")
    aligned = procrustes_align(pred[mask], gt[mask])
    return float(np.mean(per_joint_error(aligned, gt[mask])))


def pck3d(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float,
    pred_valid: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
) -> float:
    """Fraction of jointly-valid joints within ``threshold`` meters (A4).

    NaN if no jointly-valid joint. Threshold is in meters (e.g. 0.05 = 50 mm).
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt, dtype=float).reshape(-1, 3)
    mask = _common_valid(pred_valid, gt_valid, pred.shape[0])
    if not mask.any():
        return float("nan")
    err = per_joint_error(pred[mask], gt[mask])
    return float(np.mean(err <= float(threshold)))


def auc3d(
    pred: np.ndarray,
    gt: np.ndarray,
    thresholds: np.ndarray | None = None,
    pred_valid: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
) -> float:
    """Area under the PCK3D curve over ``thresholds`` (A4), mean of PCK values.

    Default thresholds sweep 0..150 mm in 5 mm steps (a common AUC range). NaN if
    no jointly-valid joint.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.15, 31)  # 0..150 mm, 5 mm steps
    vals = [pck3d(pred, gt, float(thr), pred_valid, gt_valid) for thr in thresholds]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def valid_rate(
    pred_valid: np.ndarray,
    gt_valid: np.ndarray | None = None,
) -> float:
    """Coverage (A5): fraction of GT-present joints the system reconstructed.

    Denominator is GT-valid joints (or all joints if gt_valid is None); numerator
    is joints valid in both. Report this ALONGSIDE accuracy, never folded in.
    """
    pred_valid = np.asarray(pred_valid, dtype=bool).reshape(-1)
    if gt_valid is None:
        return float(np.mean(pred_valid))
    gt_valid = np.asarray(gt_valid, dtype=bool).reshape(-1)
    denom = int(gt_valid.sum())
    if denom == 0:
        return float("nan")
    return float(np.sum(pred_valid & gt_valid) / denom)
