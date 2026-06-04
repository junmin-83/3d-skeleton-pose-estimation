"""2D pose detection via rtmlib (RTMPose ONNX).

Wraps rtmlib's Body detector. The pure helpers (normalize_rtmlib_output,
best_person) are importable/testable without rtmlib.

Pixels are (u, v), matching src/core/types.py. Keypoints are COCO-17 in a fixed
order across every view; cross-view triangulation correspondence depends on that
(SPEC.md §4-2, §5 point 2). rtmlib Body returns COCO-17 in this order regardless
of backbone.
"""

from __future__ import annotations

import os

import numpy as np

from src.core.types import NUM_KEYPOINTS, Pose2D

# rtmlib caches ONNX models under $TORCH_HOME/hub/checkpoints. Default to a
# project-local ./models (relative to cwd, so run from the project root); set
# TORCH_HOME to override. Must happen at import, before rtmlib loads in __init__.
os.environ.setdefault("TORCH_HOME", "./models")


def normalize_rtmlib_output(
    keypoints: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int = NUM_KEYPOINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce rtmlib output to batched float arrays, scores clamped to [0, 1].

    rtmlib returns either single-person (K,2)/(K,) or batched (N,K,2)/(N,K);
    both come back as batched (N,K,2) float64 and (N,K) float64. Asserts the
    keypoint count matches num_keypoints (default 17 for COCO-17).
    """
    kp = np.asarray(keypoints, dtype=float)
    sc = np.asarray(scores, dtype=float)

    # Promote single-person arrays to batched
    if kp.ndim == 2:          # (K, 2) -> (1, K, 2)
        kp = kp[np.newaxis]
    if sc.ndim == 1:          # (K,)   -> (1, K)
        sc = sc[np.newaxis]

    assert kp.shape[1] == num_keypoints, (
        f"Expected {num_keypoints} keypoints in dim 1, got {kp.shape[1]}. "
        f"keypoints shape: {kp.shape}"
    )
    assert sc.shape[1] == num_keypoints, (
        f"Expected {num_keypoints} keypoints in dim 1, got {sc.shape[1]}. "
        f"scores shape: {sc.shape}"
    )

    sc = np.clip(sc, 0.0, 1.0)
    return kp, sc


def best_person(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> Pose2D:
    """Return the highest-mean-score person as a Pose2D (zero-filled if N==0).

    keypoints (N,K,2) pixels (u,v), scores (N,K) in [0,1]. The N==0 fallback
    uses NUM_KEYPOINTS for the zero-filled shapes.
    """
    n = keypoints.shape[0]
    if n == 0:
        k = scores.shape[1] if scores.ndim == 2 else NUM_KEYPOINTS
        return Pose2D(
            keypoints=np.zeros((k, 2), dtype=float),
            scores=np.zeros(k, dtype=float),
        )

    mean_scores = scores.mean(axis=1)          # (N,)
    best_idx = int(np.argmax(mean_scores))
    return Pose2D(
        keypoints=keypoints[best_idx].copy(),  # (K, 2)
        scores=scores[best_idx].copy(),        # (K,)
    )


def apply_score_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Zero out keypoint scores below threshold (returns a copy).

    Sub-threshold joints get score 0 so downstream triangulation/fusion weighting
    drops them automatically (occlusion handling, SPEC §5.4).
    """
    out = np.array(scores, dtype=float, copy=True)
    out[out < float(threshold)] = 0.0
    return out


def resolve_device(requested: str, available_providers: list[str]) -> str:
    """Resolve the inference device, falling back to CPU when CUDA is missing.

    "cuda" is the default, but a CPU-only onnxruntime (default uv sync env, or no
    NVIDIA GPU) exposes no CUDA provider; there we drop to "cpu" so inference runs
    instead of failing or spamming an onnxruntime warning. "cpu" is honoured as-is.

    Args:
        requested: "cuda" or "cpu".
        available_providers: onnxruntime.get_available_providers() output.

    Returns:
        "cuda" only when a CUDA-capable provider is present, else "cpu".
    """
    if requested.lower() == "cuda" and any("CUDA" in p for p in available_providers):
        return "cuda"
    return "cpu"


class RTMPoseDetector:
    """rtmlib.Body wrapper for single/multi-person COCO-17 detection.

    rtmlib and onnxruntime(-gpu) are imported lazily in __init__ so importing
    this module doesn't fail where they aren't installed (e.g. offline CI).

    Output is COCO-17 in the canonical order from src.core.types.COCO_17_KEYPOINTS,
    identical across views (required for cross-view triangulation correspondence).

    Args:
        device: "cuda" (onnxruntime-gpu) or "cpu".
        mode: rtmlib Body mode ("balanced", "performance", "lite").
        backend: rtmlib backend, always "onnxruntime" here.
        score_threshold: min mean-score; kept as metadata for callers.
        det_model: detection model path/name for rtmlib.Body, or None for default.
        pose_input_size: (width, height) for the pose model, or None for default.
    """

    def __init__(
        self,
        device: str = "cuda",
        mode: str = "balanced",
        backend: str = "onnxruntime",
        score_threshold: float = 0.3,
        det_model: str | None = None,
        pose_input_size: tuple[int, int] | None = None,
    ) -> None:
        try:
            import rtmlib  # noqa: F401 - availability check
        except ImportError as exc:
            raise ImportError(
                "rtmlib not installed. "
                "Run: uv pip install rtmlib onnxruntime-gpu  "
                "(or onnxruntime for CPU)"
            ) from exc

        import rtmlib as _rtmlib

        if device == "cuda":
            # Load CUDA/cuDNN DLLs from the nvidia-*-cu12 wheels so onnxruntime-gpu's
            # CUDAExecutionProvider can init (no-op on CPU-only onnxruntime), then
            # drop to CPU if no CUDA provider, so the default device="cuda" never
            # hard-fails.
            try:
                import onnxruntime as _ort
                _ort.preload_dlls()
                device = resolve_device(device, _ort.get_available_providers())
            except Exception:
                device = "cpu"
            if device != "cuda":
                print(
                    "[RTMPoseDetector] CUDA provider unavailable -> using CPU. "
                    "For GPU acceleration see README '(참고) GPU 가속 (NVIDIA CUDA)'."
                )

        self.score_threshold = score_threshold
        self.device = device

        body_kwargs: dict = dict(mode=mode, backend=backend, device=device)
        if det_model is not None:
            body_kwargs["det"] = det_model
        if pose_input_size is not None:
            body_kwargs["pose_input_size"] = pose_input_size

        self._body: _rtmlib.Body = _rtmlib.Body(**body_kwargs)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run RTMPose on one BGR image (cv2.imread or camera frame).

        Returns keypoints (N,K,2) pixels (u,v) and scores (N,K) in [0,1].
        """
        raw_kp, raw_sc = self._body(image)
        return normalize_rtmlib_output(raw_kp, raw_sc)

    def detect_best(self, image: np.ndarray) -> Pose2D:
        """Detect on a BGR image, returning only the best person (COCO-17, (u,v))."""
        kp, sc = self.detect(image)
        pose = best_person(kp, sc)
        # Apply the detection threshold: sub-threshold joints get score 0 so
        # triangulation/fusion drop them (occluded-joint handling).
        return Pose2D(pose.keypoints, apply_score_threshold(pose.scores, self.score_threshold))
