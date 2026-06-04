"""2D skeleton overlays on BGR frames (COCO-17).

Shared by the example demos so keypoint/skeleton drawing lives in one place
instead of being re-implemented per demo. Coordinates are pixel ``(u, v)``.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.types import COCO_SKELETON


def draw_skeleton_2d(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_thr: float,
    *,
    scale: tuple[float, float] | None = None,
    copy: bool = False,
    line_color: tuple[int, int, int] = (0, 180, 0),
    line_thickness: int = 2,
    point_radius: int = 3,
    point_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draw the COCO-17 skeleton + keypoints onto a BGR ``canvas``.

    Only joints/bones whose 2D score meets ``score_thr`` are drawn. ``scale``
    maps source-image pixels onto the canvas (e.g. when the canvas is a resized
    panel); ``None`` draws at native coordinates. With ``copy=True`` the source
    is left untouched and an annotated copy is returned, else ``canvas`` is
    drawn on in place and returned.
    """
    out = canvas.copy() if copy else canvas
    kp = np.asarray(keypoints, dtype=float)
    if scale is not None:
        kp = kp * np.asarray(scale, dtype=float)
    pix = np.round(kp).astype(int)
    sc = np.asarray(scores, dtype=float)
    for i, j in COCO_SKELETON:
        if sc[i] >= score_thr and sc[j] >= score_thr:
            cv2.line(out, tuple(pix[i]), tuple(pix[j]), line_color, line_thickness)
    for k in range(len(pix)):
        if sc[k] >= score_thr:
            cv2.circle(out, tuple(pix[k]), point_radius, point_color, -1)
    return out


def label_panel(
    panel: np.ndarray,
    text: str,
    width: int,
    *,
    font_scale: float = 0.45,
) -> np.ndarray:
    """Draw a black caption bar with ``text`` across the top of ``panel``."""
    cv2.rectangle(panel, (0, 0), (width, 20), (0, 0, 0), -1)
    cv2.putText(panel, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return panel
