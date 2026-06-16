"""Metric tests on synthetic poses (no data / network).

Checks the alignment protocols behave per their definitions:
  - mpjpe is the raw mean distance and is sensitive to global translation,
  - root_relative removes translation,
  - pa_mpjpe removes rotation+scale+translation (a similarity-transformed pose
    scores ~0),
  - pck3d / auc count joints under a threshold,
  - valid masks restrict scoring to jointly-valid joints,
  - valid_rate reports coverage against GT.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.eval.metrics import (
    auc3d,
    mpjpe,
    pa_mpjpe,
    pck3d,
    pelvis_from_coco17,
    procrustes_align,
    root_relative_mpjpe,
    valid_rate,
)

rng = np.random.default_rng(0)


def _random_pose():
    return rng.normal(size=(17, 3))


def test_mpjpe_zero_on_identical():
    p = _random_pose()
    assert mpjpe(p, p) == pytest.approx(0.0, abs=1e-12)


def test_mpjpe_constant_offset_equals_offset_norm():
    p = _random_pose()
    offset = np.array([0.1, -0.2, 0.05])
    err = mpjpe(p + offset, p)
    assert err == pytest.approx(np.linalg.norm(offset), abs=1e-12)


def test_root_relative_kills_global_translation():
    p = _random_pose()
    offset = np.array([1.0, 2.0, -3.0])
    # Whole-body translation must not affect a root-relative metric.
    assert root_relative_mpjpe(p + offset, p) == pytest.approx(0.0, abs=1e-12)


def test_pa_mpjpe_invariant_to_similarity():
    p = _random_pose()
    rot = Rotation.from_euler("xyz", [30, -20, 45], degrees=True).as_matrix()
    scale = 1.7
    trans = np.array([0.5, -1.0, 2.0])
    transformed = scale * (p @ rot.T) + trans
    # A pure similarity of the GT should align back to ~0 error.
    assert pa_mpjpe(transformed, p) == pytest.approx(0.0, abs=1e-9)


def test_pa_mpjpe_nonzero_on_nonrigid_deformation():
    p = _random_pose()
    deformed = p.copy()
    deformed[5] += np.array([0.5, 0.0, 0.0])  # bend one joint only
    assert pa_mpjpe(deformed, p) > 1e-3


def test_procrustes_recovers_known_transform():
    p = _random_pose()
    rot = Rotation.from_euler("z", 25, degrees=True).as_matrix()
    transformed = 2.0 * (p @ rot.T) + np.array([1.0, 1.0, 1.0])
    aligned = procrustes_align(transformed, p)
    assert np.allclose(aligned, p, atol=1e-9)


def test_pck3d_counts_joints_under_threshold():
    p = _random_pose()
    pred = p.copy()
    pred[0] += np.array([1.0, 0.0, 0.0])   # 1.0 m off
    pred[1] += np.array([0.01, 0.0, 0.0])  # 0.01 m off
    # threshold 0.05 m: 16 of 17 within (only joint 0 fails)
    assert pck3d(pred, p, 0.05) == pytest.approx(16 / 17)


def test_auc_between_zero_and_one():
    p = _random_pose()
    pred = p + rng.normal(scale=0.02, size=p.shape)
    a = auc3d(pred, p)
    assert 0.0 <= a <= 1.0


def test_valid_mask_restricts_scoring():
    p = _random_pose()
    pred = p.copy()
    pred[3] += np.array([10.0, 0.0, 0.0])  # huge error on joint 3
    valid = np.ones(17, dtype=bool)
    valid[3] = False
    # Excluding joint 3 -> error ~0 despite the outlier.
    assert mpjpe(pred, p, pred_valid=valid) == pytest.approx(0.0, abs=1e-12)


def test_mpjpe_nan_when_no_common_valid():
    p = _random_pose()
    none_valid = np.zeros(17, dtype=bool)
    assert np.isnan(mpjpe(p, p, pred_valid=none_valid))


def test_valid_rate_coverage():
    pred_valid = np.array([True] * 10 + [False] * 7)
    gt_valid = np.array([True] * 14 + [False] * 3)
    # common = first 10 true; denom = 14 GT-valid -> 10/14
    assert valid_rate(pred_valid, gt_valid) == pytest.approx(10 / 14)


def test_pelvis_is_midhip():
    p = _random_pose()
    expected = 0.5 * (p[11] + p[12])
    assert np.allclose(pelvis_from_coco17(p), expected)
