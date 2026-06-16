"""Confidence-interval helpers for the evaluation report.

EVALUATION_PLAN §4: the basic aggregation unit is the sequence (adjacent frames
are strongly correlated, so a frame-count N overstates confidence). Use
mean_ci95_t across sequence means; use bootstrap_ci for a single sequence's
frame distribution, flagged as frame-level (autocorrelated) in the report.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci95_t(values) -> tuple[float, float, float, int]:
    """Mean and 95% t-interval half-width over `values`.

    Returns (mean, ci_half_width, std, n). For n<2 the half-width is NaN.
    Use across SEQUENCE means (independent units).
    """
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    n = v.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else float("nan")
    if n < 2:
        return (mean, float("nan"), std, n)
    half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    return (mean, half, std, n)


def bootstrap_ci(values, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float, int]:
    """Mean and 95% percentile-bootstrap interval over `values`.

    Returns (mean, lo, hi, n). For a single sequence's per-frame values this is a
    frame-level interval; report it as such (frames are autocorrelated, so it is
    optimistic relative to a true between-sequence interval).
    """
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    n = v.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(v.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), n)
