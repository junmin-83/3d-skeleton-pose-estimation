"""CMU Panoptic HD access: calibration loader plus multi-view HD frame iterator.

Panoptic is multi-view RGB for triangulation (no depth), so it is not an
RGBDSource. It gives calibrated cameras and synchronized HD video frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from src.core.types import CameraParams

_CM_TO_M = 0.01  # CMU Panoptic calibration is in centimeters


def load_panoptic_hd_cameras(calib_path: str) -> list[CameraParams]:
    """CMU Panoptic calibration JSON -> CameraParams (HD cams only, cm -> m)."""
    doc = json.load(open(calib_path, encoding="utf-8"))
    cams = []
    for c in doc["cameras"]:
        if c["type"] != "hd":
            continue
        cams.append(CameraParams(
            name=c["name"],
            K=np.asarray(c["K"], float),
            dist=np.asarray(c["distCoef"], float),
            R=np.asarray(c["R"], float),
            t=np.asarray(c["t"], float).reshape(3) * _CM_TO_M,
            image_size=(int(c["resolution"][0]), int(c["resolution"][1])),
        ))
    return cams


def iter_panoptic_hd_frames(
    seq_dir: str | Path,
    cam_names: Sequence[str],
    start: int,
    num: int,
) -> Iterator[list[np.ndarray]]:
    """Yield, per timestep, a list of BGR frames (one per camera).

    Opens each camera's hdVideos/hd_00_<n>.mp4 up front (an unreadable video
    raises FileNotFoundError before iteration), seeks to start, then yields up
    to num synchronized frame-lists, stopping early when any stream ends.
    Captures are released when iteration finishes.
    """
    seq = Path(seq_dir)
    caps = [cv2.VideoCapture(str(seq / "hdVideos" / f"hd_00_{n.split('_')[1]}.mp4"))
            for n in cam_names]
    for cap, n in zip(caps, cam_names):
        if not cap.isOpened():
            for c in caps:
                c.release()
            raise FileNotFoundError(f"cannot open hdVideos/hd_00_{n.split('_')[1]}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    def _gen() -> Iterator[list[np.ndarray]]:
        try:
            for _ in range(num):
                frames = []
                for cap in caps:
                    ok, img = cap.read()
                    frames.append(img if ok else None)
                if any(im is None for im in frames):
                    break
                yield frames
        finally:
            for cap in caps:
                cap.release()

    return _gen()
