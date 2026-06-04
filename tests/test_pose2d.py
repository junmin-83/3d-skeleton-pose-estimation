"""Tests for src/pose2d/rtmpose_detector.py.

All tests pass offline (no rtmlib / onnxruntime required) except the
detector-integration test, which is skipped when rtmlib is absent.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from src.pose2d.rtmpose_detector import (
    RTMPoseDetector,
    apply_score_threshold,
    best_person,
    normalize_rtmlib_output,
    resolve_device,
)
from src.core.types import NUM_KEYPOINTS, Pose2D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_K = NUM_KEYPOINTS  # 17


def _kp(n: int = 1, k: int = _K) -> np.ndarray:
    """Return a float array of shape (n, k, 2) with unique values."""
    return np.arange(n * k * 2, dtype=float).reshape(n, k, 2)


def _sc(n: int = 1, k: int = _K, fill: float = 0.5) -> np.ndarray:
    return np.full((n, k), fill, dtype=float)


# ---------------------------------------------------------------------------
# normalize_rtmlib_output
# ---------------------------------------------------------------------------

class TestNormalizeRtmlibOutput:
    def test_single_person_promoted_to_batch(self):
        kp_in = np.ones((_K, 2), dtype=float)
        sc_in = np.full(_K, 0.8, dtype=float)
        kp_out, sc_out = normalize_rtmlib_output(kp_in, sc_in)
        assert kp_out.shape == (1, _K, 2), f"Expected (1,{_K},2), got {kp_out.shape}"
        assert sc_out.shape == (1, _K),    f"Expected (1,{_K}),   got {sc_out.shape}"

    def test_batched_shape_unchanged(self):
        kp_in = np.ones((3, _K, 2), dtype=float)
        sc_in = np.full((3, _K), 0.6, dtype=float)
        kp_out, sc_out = normalize_rtmlib_output(kp_in, sc_in)
        assert kp_out.shape == (3, _K, 2)
        assert sc_out.shape == (3, _K)

    def test_scores_clamped_above_one(self):
        sc_in = np.full((_K,), 1.5, dtype=float)
        _, sc_out = normalize_rtmlib_output(np.zeros((_K, 2)), sc_in)
        assert sc_out.max() <= 1.0

    def test_scores_clamped_below_zero(self):
        sc_in = np.full((_K,), -0.3, dtype=float)
        _, sc_out = normalize_rtmlib_output(np.zeros((_K, 2)), sc_in)
        assert sc_out.min() >= 0.0

    def test_scores_mixed_clamping(self):
        sc_in = np.linspace(-0.5, 1.5, _K)
        _, sc_out = normalize_rtmlib_output(np.zeros((_K, 2)), sc_in)
        assert sc_out.min() >= 0.0
        assert sc_out.max() <= 1.0

    def test_wrong_k_raises_assertion(self):
        with pytest.raises(AssertionError):
            normalize_rtmlib_output(np.ones((10, 2)), np.ones(10))  # K=10 != 17

    def test_wrong_k_batched_raises_assertion(self):
        with pytest.raises(AssertionError):
            normalize_rtmlib_output(np.ones((2, 10, 2)), np.ones((2, 10)))

    def test_output_dtype_is_float(self):
        kp_in = np.ones((_K, 2), dtype=np.float32)
        sc_in = np.ones(_K, dtype=np.float32)
        kp_out, sc_out = normalize_rtmlib_output(kp_in, sc_in)
        assert kp_out.dtype == float
        assert sc_out.dtype == float

    def test_values_preserved_within_range(self):
        kp_in = np.arange(_K * 2, dtype=float).reshape(_K, 2)
        sc_in = np.linspace(0.0, 1.0, _K)
        kp_out, sc_out = normalize_rtmlib_output(kp_in, sc_in)
        np.testing.assert_array_equal(kp_out[0], kp_in)
        np.testing.assert_allclose(sc_out[0], sc_in)


# ---------------------------------------------------------------------------
# best_person
# ---------------------------------------------------------------------------

class TestBestPerson:
    def test_returns_higher_scoring_person(self):
        kp = _kp(n=2)
        sc = np.zeros((2, _K), dtype=float)
        sc[0] = 0.3   # lower mean
        sc[1] = 0.9   # higher mean — should be selected
        result = best_person(kp, sc)
        assert isinstance(result, Pose2D)
        np.testing.assert_array_equal(result.keypoints, kp[1])
        np.testing.assert_array_equal(result.scores, sc[1])

    def test_result_shapes(self):
        kp = _kp(n=3)
        sc = _sc(n=3)
        result = best_person(kp, sc)
        assert result.keypoints.shape == (_K, 2)
        assert result.scores.shape == (_K,)

    def test_single_person_returned(self):
        kp = _kp(n=1)
        sc = _sc(n=1, fill=0.7)
        result = best_person(kp, sc)
        np.testing.assert_array_equal(result.keypoints, kp[0])
        np.testing.assert_array_equal(result.scores, sc[0])

    def test_empty_returns_zeros(self):
        kp = np.empty((0, _K, 2), dtype=float)
        sc = np.empty((0, _K), dtype=float)
        result = best_person(kp, sc)
        assert isinstance(result, Pose2D)
        assert result.keypoints.shape == (_K, 2)
        assert result.scores.shape == (_K,)
        np.testing.assert_array_equal(result.keypoints, np.zeros((_K, 2)))
        np.testing.assert_array_equal(result.scores, np.zeros(_K))

    def test_result_is_copy_not_view(self):
        """Mutating the returned Pose2D should not affect the source array."""
        kp = _kp(n=2)
        sc = _sc(n=2)
        original_val = float(kp[0, 0, 0])
        result = best_person(kp, sc)
        result.keypoints[0, 0] = -9999.0
        # source array for person 0 must be untouched
        assert kp[0, 0, 0] == original_val

    def test_tiebreak_picks_first_argmax(self):
        """When two persons have equal mean score, argmax returns the first."""
        kp = _kp(n=2)
        sc = np.full((2, _K), 0.5, dtype=float)
        result = best_person(kp, sc)
        np.testing.assert_array_equal(result.keypoints, kp[0])


# ---------------------------------------------------------------------------
# RTMPoseDetector — ImportError when rtmlib absent
# ---------------------------------------------------------------------------

# Detect whether rtmlib is actually installed in this environment.
_rtmlib_available = importlib.util.find_spec("rtmlib") is not None


class TestDetectorImport:
    def test_raises_import_error_when_rtmlib_absent(self):
        """When rtmlib is not installed, RTMPoseDetector.__init__ must raise
        ImportError with a helpful install message."""
        if _rtmlib_available:
            pytest.skip("rtmlib is installed; ImportError path cannot be triggered")
        with pytest.raises(ImportError, match="rtmlib"):
            RTMPoseDetector(device="cpu")

    @pytest.mark.skipif(
        not _rtmlib_available,
        reason="rtmlib not installed; skipping live-detector smoke test",
    )
    def test_detector_builds_when_rtmlib_present(self):
        """If rtmlib is installed this smoke-test verifies the detector
        constructs without error (no image inference needed)."""
        det = RTMPoseDetector(device="cpu", mode="lightweight")
        assert hasattr(det, "_body")
        assert det.score_threshold == 0.3


# ---------------------------------------------------------------------------
# apply_score_threshold — occluded-joint down-weighting (SPEC §5.4)
# ---------------------------------------------------------------------------

class TestApplyScoreThreshold:
    def test_zeros_below_threshold(self):
        sc = np.array([0.1, 0.4, 0.9, 0.29, 0.3])
        out = apply_score_threshold(sc, 0.3)
        np.testing.assert_array_equal(out, [0.0, 0.4, 0.9, 0.0, 0.3])

    def test_returns_copy_source_untouched(self):
        sc = np.array([0.1, 0.9])
        out = apply_score_threshold(sc, 0.5)
        assert out is not sc
        np.testing.assert_array_equal(sc, [0.1, 0.9])


# ---------------------------------------------------------------------------
# resolve_device — GPU default with graceful CPU fallback
# ---------------------------------------------------------------------------

class TestResolveDevice:
    _GPU = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    _CPU = ["CPUExecutionProvider"]

    def test_cuda_kept_when_cuda_provider_present(self):
        assert resolve_device("cuda", self._GPU) == "cuda"

    def test_cuda_falls_back_when_no_cuda_provider(self):
        assert resolve_device("cuda", self._CPU) == "cpu"

    def test_cuda_falls_back_on_empty_providers(self):
        assert resolve_device("cuda", []) == "cpu"

    def test_cpu_always_stays_cpu_even_with_gpu(self):
        assert resolve_device("cpu", self._GPU) == "cpu"

    def test_case_insensitive_request(self):
        assert resolve_device("CUDA", self._GPU) == "cuda"
