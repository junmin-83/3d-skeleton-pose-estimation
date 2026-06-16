"""CI-helper tests (mean_ci95_t, bootstrap_ci)."""

import numpy as np

from src.eval.stats import bootstrap_ci, mean_ci95_t


def test_mean_ci95_t_basic():
    mean, half, std, n = mean_ci95_t([10.0, 12.0, 14.0, 16.0])
    assert mean == 13.0
    assert n == 4
    assert half > 0
    # t(0.975, 3)=3.182; std(ddof=1)=2.582; half=3.182*2.582/2 ≈ 4.108
    assert abs(half - 4.108) < 0.02


def test_mean_ci95_t_single_value_nan_half():
    mean, half, std, n = mean_ci95_t([5.0])
    assert mean == 5.0
    assert n == 1
    assert np.isnan(half)


def test_mean_ci95_t_drops_nan():
    mean, half, std, n = mean_ci95_t([1.0, np.nan, 3.0])
    assert n == 2
    assert mean == 2.0


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    data = rng.normal(loc=50.0, scale=5.0, size=200)
    mean, lo, hi, n = bootstrap_ci(data, n_boot=2000)
    assert n == 200
    assert lo < mean < hi
    assert abs(mean - 50.0) < 2.0


def test_bootstrap_ci_empty():
    mean, lo, hi, n = bootstrap_ci([])
    assert n == 0
    assert np.isnan(mean)
