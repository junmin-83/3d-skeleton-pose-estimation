"""2D pose detection via rtmlib (RTMPose ONNX).

This module wraps rtmlib's Body detector and provides helpers that are
testable without rtmlib installed (normalize_rtmlib_output, best_person).

Coordinate convention: all pixel outputs are (u, v) order, matching the
project-wide convention in src/core/types.py.

Keypoint order: COCO-17, fixed across every view — this is a hard requirement
for cross-view triangulation correspondence (see SPEC.md §4-2 and §5 point 2).
rtmlib Body with any RTMPose backbone returns COCO-17 in this identical order.
"""

from __future__ import annotations

import os

import numpy as np

from src.core.types import NUM_KEYPOINTS, Pose2D

# rtmlib caches RTMPose ONNX models under $TORCH_HOME/hub/checkpoints. Default it
# to a project-local ./models dir (relative to the current working directory, so
# run from the project root). Set the TORCH_HOME env var to override. This runs
# at import time, before rtmlib is lazily imported in RTMPoseDetector.__init__.
os.environ.setdefault("TORCH_HOME", "./models")


# ---------------------------------------------------------------------------
# Pure helpers — importable and testable with NO rtmlib dependency
# ---------------------------------------------------------------------------

def normalize_rtmlib_output(
    keypoints: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int = NUM_KEYPOINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce rtmlib output to batched float arrays.

    rtmlib may return single-person arrays of shape ``(K, 2)`` / ``(K,)`` or
    batched ``(N, K, 2)`` / ``(N, K)``.  This function normalises either form
    to the batched shape and clamps scores to ``[0, 1]``.

    Parameters
    ----------
    keypoints:
        Raw keypoint array from rtmlib — shape ``(K, 2)`` or ``(N, K, 2)``.
    scores:
        Raw score array from rtmlib — shape ``(K,)`` or ``(N, K)``.
    num_keypoints:
        Expected number of keypoints (default: 17 for COCO-17).

    Returns
    -------
    kp_out : np.ndarray, shape ``(N, K, 2)``, dtype float64
    sc_out : np.ndarray, shape ``(N, K)``,   dtype float64

    Raises
    ------
    AssertionError
        If the last keypoint dimension does not equal ``num_keypoints``.
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
    """Select the person with the highest mean keypoint score.

    Parameters
    ----------
    keypoints : np.ndarray, shape ``(N, K, 2)``
        Batched pixel coords (u, v) for N persons.
    scores : np.ndarray, shape ``(N, K)``
        Confidence scores in ``[0, 1]`` for each keypoint.

    Returns
    -------
    Pose2D
        The single best person's pose.  If ``N == 0`` returns a zero-filled
        ``Pose2D`` with shape ``(K, 2)`` / ``(K,)`` using ``NUM_KEYPOINTS``.
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
    """Zero keypoint scores below ``threshold``.

    Keypoints whose 2D confidence falls below the detection threshold are
    down-weighted to 0 so the downstream triangulation/fusion weighting drops
    them automatically (occlusion handling, SPEC §5.4). Returns a copy.
    """
    out = np.array(scores, dtype=float, copy=True)
    out[out < float(threshold)] = 0.0
    return out


def resolve_device(requested: str, available_providers: list[str]) -> str:
    """Resolve the effective inference device, falling back to CPU.

    ``"cuda"`` is the project default, but a CPU-only ``onnxruntime`` (the
    default ``uv sync`` environment, or a machine with no NVIDIA GPU) exposes no
    CUDA execution provider.  In that case ``"cuda"`` quietly falls back to
    ``"cpu"`` so inference still runs instead of failing / spamming an
    onnxruntime warning.  ``"cpu"`` is always honoured as-is.

    Parameters
    ----------
    requested:
        Device requested by the caller (``"cuda"`` or ``"cpu"``).
    available_providers:
        ``onnxruntime.get_available_providers()`` output.

    Returns
    -------
    str
        ``"cuda"`` only when a CUDA-capable provider is present, else ``"cpu"``.
    """
    if requested.lower() == "cuda" and any("CUDA" in p for p in available_providers):
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# RTMPoseDetector — rtmlib imported lazily inside __init__
# ---------------------------------------------------------------------------

class RTMPoseDetector:
    """Wraps rtmlib.Body for single- or multi-person COCO-17 pose detection.

    rtmlib and onnxruntime(-gpu) are imported *lazily* so that importing this
    module does not fail in environments where they are not installed (e.g.
    offline CI).

    Keypoint order guarantee
    ------------------------
    rtmlib Body with any RTMPose backbone outputs COCO-17 keypoints in the
    canonical order defined in ``src.core.types.COCO_17_KEYPOINTS``.  This
    ordering is *identical* across all views, which is a hard requirement for
    cross-view triangulation correspondence.

    Parameters
    ----------
    device : {'cuda', 'cpu'}
        Inference device.  Use ``'cuda'`` for onnxruntime-gpu.
    mode : str
        rtmlib Body mode, e.g. ``'balanced'``, ``'performance'``, ``'lite'``.
    backend : str
        rtmlib inference backend — always ``'onnxruntime'`` in this project.
    score_threshold : float
        Minimum mean-score threshold; kept as metadata for callers.
    det_model : str | None
        Path or name of the detection model passed to rtmlib.Body.  ``None``
        uses rtmlib's default.
    pose_input_size : tuple[int, int] | None
        ``(width, height)`` input size for the pose model.  ``None`` uses
        rtmlib's default.
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
            import rtmlib  # noqa: F401 — just check availability first
        except ImportError as exc:
            raise ImportError(
                "rtmlib not installed. "
                "Run: uv pip install rtmlib onnxruntime-gpu  "
                "(or onnxruntime for CPU)"
            ) from exc

        import rtmlib as _rtmlib

        if device == "cuda":
            # Load CUDA/cuDNN DLLs from the nvidia-*-cu12 pip wheels so
            # onnxruntime-gpu's CUDAExecutionProvider can initialize (harmless
            # no-op on CPU-only onnxruntime), then drop to CPU when no CUDA
            # provider is present so the default device="cuda" never hard-fails.
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

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run RTMPose on a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image as returned by ``cv2.imread`` or a camera frame.

        Returns
        -------
        keypoints : np.ndarray, shape ``(N, K, 2)``
            Pixel coords (u, v) for up to N detected persons.
        scores : np.ndarray, shape ``(N, K)``
            Confidence scores in ``[0, 1]``.
        """
        raw_kp, raw_sc = self._body(image)
        return normalize_rtmlib_output(raw_kp, raw_sc)

    def detect_best(self, image: np.ndarray) -> Pose2D:
        """Detect and return only the highest-confidence person.

        Parameters
        ----------
        image : np.ndarray
            BGR image.

        Returns
        -------
        Pose2D
            Best person's pose (COCO-17 keypoints in (u, v) order).
        """
        kp, sc = self.detect(image)
        pose = best_person(kp, sc)
        # Enforce the configured detection threshold: sub-threshold joints get
        # score 0 so triangulation/fusion drop them (occluded-joint handling).
        return Pose2D(pose.keypoints, apply_score_threshold(pose.scores, self.score_threshold))
