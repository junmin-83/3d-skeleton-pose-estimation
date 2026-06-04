"""One-Euro filter for temporal smoothing of 3D pose keypoints.

Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012).
    1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in
    Interactive Systems. CHI 2012.

Units: 3-D coordinates in meters (world frame), time in seconds.
"""

from __future__ import annotations

import math


from src.core.types import NUM_KEYPOINTS, Pose3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alpha(cutoff: float, freq: float) -> float:
    """Smoothing coefficient for a first-order low-pass filter.

    alpha = 1 / (1 + freq / (2*pi*cutoff))

    Args:
        cutoff: Filter cutoff frequency in Hz.
        freq:   Sampling frequency in Hz (must be > 0).

    Returns:
        alpha in (0, 1].
    """
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * freq)


# ---------------------------------------------------------------------------
# Scalar One-Euro filter
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """One-Euro low-pass filter for a scalar signal.

    The filter adapts its cutoff frequency based on the estimated signal
    velocity: slow motion gets heavy smoothing; fast motion gets less lag.

    Args:
        freq:       Nominal sampling frequency in Hz (used when no timestamp
                    is supplied, or for the very first derivative estimate).
        min_cutoff: Minimum cutoff frequency (Hz) applied to the signal.
                    Lower values → more smoothing at rest.
        beta:       Speed coefficient.  Higher values reduce lag for fast
                    motion at the cost of more jitter.
        d_cutoff:   Cutoff frequency (Hz) for the derivative low-pass filter.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        if freq <= 0.0 or min_cutoff <= 0.0 or d_cutoff <= 0.0:
            raise ValueError("freq, min_cutoff and d_cutoff must all be > 0")
        self._freq = float(freq)
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_hat: float = 0.0
        self._t_prev: float | None = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear filter state so the next call is treated as the first sample."""
        self._x_prev = None
        self._dx_hat = 0.0
        self._t_prev = None

    # ------------------------------------------------------------------
    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """Filter one sample.

        Args:
            x:         Raw scalar value.
            timestamp: Optional wall-clock time in seconds.  When provided the
                       filter derives the actual sampling frequency from
                       ``dt = timestamp - t_prev``.  Ignored on the first call
                       (stored for next time).

        Returns:
            Smoothed scalar value.
        """
        # Determine effective sampling frequency for this step.
        freq = self._freq
        if timestamp is not None and self._t_prev is not None:
            dt = timestamp - self._t_prev
            if dt > 0.0:
                freq = 1.0 / dt
        if timestamp is not None:
            self._t_prev = timestamp

        # First sample: initialise and return raw value.
        if self._x_prev is None:
            self._x_prev = x
            self._dx_hat = 0.0
            return x

        # --- Derivative low-pass ---
        dx = (x - self._x_prev) * freq
        a_d = _alpha(self._d_cutoff, freq)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat

        # --- Adaptive cutoff for the signal ---
        # Clamp to a small positive value so a negative beta can never drive the
        # cutoff <= 0 (which would blow up the 1/(2*pi*cutoff) time constant).
        cutoff = max(self._min_cutoff + self._beta * abs(self._dx_hat), 1e-6)

        # --- Signal low-pass ---
        a = _alpha(cutoff, freq)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat

    # Alias so both calling conventions work.
    def filter(self, x: float, timestamp: float | None = None) -> float:
        """Alias for ``__call__``."""
        return self(x, timestamp)


# ---------------------------------------------------------------------------
# Per-keypoint pose smoother
# ---------------------------------------------------------------------------

class PoseSmoother:
    """Applies one independent One-Euro filter per (keypoint, axis) coordinate.

    A (K, 3) grid of :class:`OneEuroFilter` instances is created at
    construction.  On each frame only *valid* keypoints are updated; invalid
    keypoints (``pose3d.valid[k] == False``) are passed through unchanged and
    their filter state is left unmodified so it can pick up cleanly when the
    keypoint becomes visible again.

    Args:
        num_keypoints: Number of keypoints K (default: 17 for COCO-17).
        freq:          Nominal sampling frequency in Hz.
        min_cutoff:    One-Euro min_cutoff parameter (Hz).
        beta:          One-Euro beta (speed coefficient).
        d_cutoff:      One-Euro derivative low-pass cutoff (Hz).
    """

    def __init__(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._num_keypoints = num_keypoints
        # Grid shape: (K, 3) — one filter per coordinate.
        self._filters: list[list[OneEuroFilter]] = [
            [
                OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
                for _ in range(3)
            ]
            for _ in range(num_keypoints)
        ]

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all filter states."""
        for row in self._filters:
            for f in row:
                f.reset()

    # ------------------------------------------------------------------
    def update(self, pose3d: Pose3D, timestamp: float | None = None) -> Pose3D:
        """Smooth one frame of 3-D pose.

        Valid keypoints are filtered; invalid ones are copied through as-is
        without touching their filter state.

        Args:
            pose3d:    Input pose (K, 3) in world-frame meters.
            timestamp: Optional wall-clock time in seconds.

        Returns:
            New :class:`Pose3D` with smoothed ``points``; ``scores``,
            ``valid``, and ``source`` are forwarded unchanged.
        """
        K = self._num_keypoints
        smoothed = pose3d.points.copy()

        for k in range(K):
            if not pose3d.valid[k]:
                # Pass through; do NOT update filter state for this joint.
                continue
            for axis in range(3):
                smoothed[k, axis] = self._filters[k][axis](
                    pose3d.points[k, axis], timestamp
                )

        return Pose3D(
            points=smoothed,
            scores=pose3d.scores.copy(),
            valid=pose3d.valid.copy(),
            source=list(pose3d.source),
        )
