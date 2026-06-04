"""One-Euro filter for temporal smoothing of 3D pose keypoints.

Casiez, Roussel & Vogel, "1€ Filter", CHI 2012.
Units: 3D coords in meters (world frame), time in seconds.
"""

from __future__ import annotations

import math


from src.core.types import NUM_KEYPOINTS, Pose3D


def _alpha(cutoff: float, freq: float) -> float:
    """First-order low-pass smoothing coefficient, alpha in (0, 1].

    alpha = 1 / (1 + freq / (2*pi*cutoff)). cutoff and freq in Hz, freq > 0.
    """
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * freq)


class OneEuroFilter:
    """One-Euro low-pass filter for a scalar signal.

    Cutoff adapts to estimated velocity: slow motion gets heavy smoothing, fast
    motion gets less lag.

    Args:
        freq: Nominal sampling rate (Hz), used when no timestamp is given or for
            the first derivative estimate.
        min_cutoff: Signal min cutoff (Hz); lower means more smoothing at rest.
        beta: Speed coefficient; higher cuts lag on fast motion but adds jitter.
        d_cutoff: Cutoff (Hz) for the derivative low-pass.
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

    def reset(self) -> None:
        """Clear state so the next call is treated as the first sample."""
        self._x_prev = None
        self._dx_hat = 0.0
        self._t_prev = None

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """Filter one sample, returning the smoothed value.

        Args:
            x: Raw scalar value.
            timestamp: Wall-clock time in seconds. When given, the real sampling
                rate is derived from dt = timestamp - t_prev. Ignored on the
                first call (just stored for next time).
        """
        # Effective sampling rate for this step.
        freq = self._freq
        if timestamp is not None and self._t_prev is not None:
            dt = timestamp - self._t_prev
            if dt > 0.0:
                freq = 1.0 / dt
        if timestamp is not None:
            self._t_prev = timestamp

        # First sample: init and return raw.
        if self._x_prev is None:
            self._x_prev = x
            self._dx_hat = 0.0
            return x

        # Derivative low-pass.
        dx = (x - self._x_prev) * freq
        a_d = _alpha(self._d_cutoff, freq)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat

        # Adaptive signal cutoff. Clamp positive so a negative beta can't drive
        # cutoff <= 0 and blow up the 1/(2*pi*cutoff) time constant.
        cutoff = max(self._min_cutoff + self._beta * abs(self._dx_hat), 1e-6)

        # Signal low-pass.
        a = _alpha(cutoff, freq)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat

    def filter(self, x: float, timestamp: float | None = None) -> float:
        """Alias for __call__, so both calling conventions work."""
        return self(x, timestamp)


class PoseSmoother:
    """One independent One-Euro filter per (keypoint, axis) coordinate.

    Builds a (K, 3) grid of OneEuroFilter at construction. Each frame only
    filters valid keypoints; invalid ones (valid[k] == False) pass through
    untouched and keep their filter state, so a joint resumes cleanly when it
    reappears.

    Args:
        num_keypoints: K (default 17 for COCO-17).
        freq: Nominal sampling rate (Hz).
        min_cutoff, beta, d_cutoff: One-Euro params (Hz / speed coeff / Hz).
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
        # (K, 3) grid: one filter per coordinate.
        self._filters: list[list[OneEuroFilter]] = [
            [
                OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
                for _ in range(3)
            ]
            for _ in range(num_keypoints)
        ]

    def reset(self) -> None:
        """Reset all filter states."""
        for row in self._filters:
            for f in row:
                f.reset()

    def update(self, pose3d: Pose3D, timestamp: float | None = None) -> Pose3D:
        """Smooth one frame of 3D pose.

        Valid keypoints get filtered; invalid ones are copied through as-is with
        their filter state untouched.

        Args:
            pose3d: Input pose (K, 3), world-frame meters.
            timestamp: Optional wall-clock time in seconds.

        Returns:
            New Pose3D with smoothed points; scores, valid, source forwarded
            unchanged.
        """
        K = self._num_keypoints
        smoothed = pose3d.points.copy()

        for k in range(K):
            if not pose3d.valid[k]:
                # Pass through; do NOT touch this joint's filter state.
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
