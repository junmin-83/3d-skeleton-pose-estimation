"""Tests for the One-Euro temporal smoothing filter.

Covers:
  - Constant signal convergence
  - Jitter reduction on a noisy sine wave
  - Monotonic step response
  - Variable-dt (timestamp) path does not crash and reduces variance
  - PoseSmoother: lower variance on valid keypoints; invalid keypoints
    pass through unchanged
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# Make the project src importable without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.smoothing.one_euro import OneEuroFilter, PoseSmoother
from src.core.types import NUM_KEYPOINTS, Pose3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pose3d(points: np.ndarray, valid: np.ndarray | None = None) -> Pose3D:
    K = points.shape[0]
    if valid is None:
        valid = np.ones(K, dtype=bool)
    return Pose3D(
        points=points.copy(),
        scores=np.ones(K, dtype=float),
        valid=valid,
        source=["test"] * K,
    )


# ---------------------------------------------------------------------------
# OneEuroFilter tests
# ---------------------------------------------------------------------------

class TestOneEuroFilterConstant:
    """Feeding a constant signal many times should converge to that constant."""

    @pytest.mark.parametrize("value", [0.0, 1.0, -3.14, 100.0])
    def test_converges_to_constant(self, value: float) -> None:
        filt = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.0)
        result = value
        for _ in range(300):
            result = filt(value)
        assert abs(result - value) < 1e-3, (
            f"Expected convergence to {value}, got {result}"
        )


class TestOneEuroFilterJitterReduction:
    """Smoothed signal should have lower frame-to-frame jitter than noisy input."""

    def test_variance_reduced(self) -> None:
        rng = np.random.default_rng(42)
        freq = 30.0
        n = 500
        t = np.arange(n) / freq
        clean = np.sin(2 * math.pi * 0.5 * t)          # 0.5 Hz sine
        noisy = clean + rng.normal(0.0, 0.1, size=n)   # sigma=0.1 jitter

        filt = OneEuroFilter(freq=freq, min_cutoff=1.5, beta=0.007)
        smoothed = np.array([filt(x) for x in noisy])

        # Frame-to-frame variance (first differences) directly measures jitter.
        # Skip warm-up samples so initialisation transient doesn't skew results.
        warm = 30
        var_noisy = float(np.var(np.diff(noisy[warm:])))
        var_smoothed = float(np.var(np.diff(smoothed[warm:])))

        assert var_smoothed < var_noisy, (
            f"Smoothed jitter {var_smoothed:.6f} should be < noisy jitter "
            f"{var_noisy:.6f}"
        )


class TestOneEuroFilterStepResponse:
    """After a step, the output should move monotonically toward the new level."""

    def test_step_response_monotonic(self) -> None:
        filt = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.5)

        # Warm up at 0.
        for _ in range(100):
            filt(0.0)

        # Apply step to 1.0 and collect 60 samples.
        outputs = [filt(1.0) for _ in range(60)]

        # Every sample should be >= the previous (monotonically non-decreasing).
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i - 1] - 1e-9, (
                f"Step response not monotonic at index {i}: "
                f"{outputs[i-1]:.6f} -> {outputs[i]:.6f}"
            )

        # Final value should be close to 1.0.
        assert abs(outputs[-1] - 1.0) < 0.05, (
            f"Step response did not converge: final = {outputs[-1]:.6f}"
        )


class TestOneEuroFilterVariableDt:
    """Irregular timestamps must not crash and should still reduce variance."""

    def test_variable_dt_no_crash_and_smooths(self) -> None:
        rng = np.random.default_rng(7)
        n = 200
        # Irregular inter-sample intervals around 33 ms.
        dt_arr = 0.033 + rng.uniform(-0.010, 0.010, size=n)
        timestamps = np.concatenate([[0.0], np.cumsum(dt_arr)])  # length n+1

        clean = np.sin(2 * math.pi * 0.5 * timestamps)
        noisy = clean + rng.normal(0.0, 0.1, size=len(timestamps))

        filt = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.007)
        smoothed = np.array([filt(x, t) for x, t in zip(noisy, timestamps)])

        warm = 20
        var_noisy = float(np.var(np.diff(noisy[warm:])))
        var_smoothed = float(np.var(np.diff(smoothed[warm:])))

        assert var_smoothed < var_noisy, (
            f"Variable-dt smoothed jitter {var_smoothed:.6f} not < noisy jitter "
            f"{var_noisy:.6f}"
        )


# ---------------------------------------------------------------------------
# PoseSmoother tests
# ---------------------------------------------------------------------------

class TestPoseSmoother:
    """PoseSmoother reduces variance on valid keypoints; invalid pass through."""

    def _make_noisy_sequence(
        self,
        rng: np.random.Generator,
        n_frames: int,
        K: int = NUM_KEYPOINTS,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Return (noisy_frames list, clean_trajectory array shape (n,K,3))."""
        t = np.arange(n_frames) / 30.0
        # Slow sine on each axis with different phases.
        clean = np.stack(
            [
                np.outer(np.sin(2 * math.pi * 0.3 * t + ph), np.ones(3))
                for ph in np.linspace(0, math.pi, K)
            ],
            axis=1,
        )  # shape (n, K, 3)

        noise = rng.normal(0.0, 0.05, size=(n_frames, K, 3))
        noisy = clean + noise
        return [noisy[i] for i in range(n_frames)], clean

    def test_variance_reduced_on_valid(self) -> None:
        rng = np.random.default_rng(123)
        n = 300
        noisy_frames, clean = self._make_noisy_sequence(rng, n)

        smoother = PoseSmoother(num_keypoints=NUM_KEYPOINTS, freq=30.0,
                                min_cutoff=1.5, beta=0.007)

        smoothed_pts = []
        for pts in noisy_frames:
            pose = _make_pose3d(pts)
            out = smoother.update(pose)
            smoothed_pts.append(out.points.copy())

        smoothed_arr = np.stack(smoothed_pts)   # (n, K, 3)
        noisy_arr = np.stack(noisy_frames)      # (n, K, 3)

        warm = 30
        # Pick keypoint 5 (left_shoulder) as representative.
        # Use frame-to-frame (diff) variance to measure jitter reduction.
        k = 5
        var_noisy = float(np.var(np.diff(noisy_arr[warm:, k, :], axis=0)))
        var_smoothed = float(np.var(np.diff(smoothed_arr[warm:, k, :], axis=0)))

        assert var_smoothed < var_noisy, (
            f"PoseSmoother did not reduce jitter for keypoint {k}: "
            f"smoothed={var_smoothed:.6f}, noisy={var_noisy:.6f}"
        )

    def test_invalid_keypoint_passthrough(self) -> None:
        """An invalid keypoint must be returned verbatim; valid ones are smoothed."""
        rng = np.random.default_rng(99)
        K = NUM_KEYPOINTS
        smoother = PoseSmoother(num_keypoints=K, freq=30.0)

        INVALID_KP = 3  # left_ear — will be set invalid every frame.
        # Unique sentinel value to verify exact pass-through.
        SENTINEL = np.array([9.99, 8.88, 7.77])

        noisy_frames, _ = self._make_noisy_sequence(rng, 100)
        for pts in noisy_frames:
            pts[INVALID_KP] = SENTINEL.copy()
            valid = np.ones(K, dtype=bool)
            valid[INVALID_KP] = False
            pose = _make_pose3d(pts, valid=valid)
            out = smoother.update(pose)

            # Invalid keypoint must match sentinel exactly.
            np.testing.assert_array_equal(
                out.points[INVALID_KP],
                SENTINEL,
                err_msg=f"Invalid keypoint {INVALID_KP} was not passed through unchanged.",
            )
            # valid flag must be forwarded.
            assert not out.valid[INVALID_KP], (
                "valid flag for invalid keypoint was changed."
            )


def test_rejects_nonpositive_cutoff():
    """Non-positive freq/cutoff must be rejected (guards 1/(2*pi*cutoff))."""
    from src.smoothing.one_euro import OneEuroFilter

    for kwargs in ({"min_cutoff": 0.0}, {"freq": 0.0}, {"d_cutoff": -1.0}):
        with pytest.raises(ValueError):
            OneEuroFilter(**kwargs)
