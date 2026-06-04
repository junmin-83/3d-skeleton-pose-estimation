"""MVOR (Multi-View Operating Room) dataset adapter: calibration + frame iterator.

MVOR is the public dataset that gives the fusion demo's missing combination —
**multiple calibrated RGB views + aligned metric depth in one world frame** —
so the Pipeline's hybrid ``fused`` path can run on real pixels + real depth (not
the synthetic setup of ``examples/fusion_demo.py``). Three RGB-D cameras
(640x480) observe the same scene; one is exposed as the depth provider.

Calibration (``annotations/camma_mvor_2018.json`` -> ``cameras_info.camParams``):
  - ``intrinsics[i]``: ``focallength=[fx,fy]``, ``principalpoint=[cx,cy]``,
    ``imagesize=[w,h]`` (camera position ``i`` == ``cam_id`` ``i+1``).
  - ``extrinsics[i]``: row-major 4x4 in **millimeters** mapping ``cam_i -> cam1``
    (``extrinsics[0]`` is identity). The project frame is WORLD == cam1, and
    ``CameraParams`` extrinsics map WORLD -> camera, so they are
    ``inv(extrinsics[i])`` with translation converted mm -> m. (Verified against
    MVOR's own 3D annotations: ~11-15 px median reprojection across all 3 cams;
    the un-inverted direction gives 45-183 px and is wrong.)

Distortion: MVOR's reference projection is pinhole, so ``dist`` defaults to zero
(reprojection already ~11 px on the reference cam). ``use_distortion=True``
applies the raw coefficients in OpenCV ``[k1,k2,p1,p2,k3]`` order.

Frames: ``day{d}/cam{c}/depth/{frame}.png`` mirrors the colour path; depth is a
16-bit PNG in millimeters (``depth_scale=1000``) aligned to the colour stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.core.types import CameraParams, DepthCameraParams

_MM_TO_M = 1e-3  # MVOR calibration / depth units are millimeters.


def _intrinsic_matrix(intr: dict) -> np.ndarray:
    """Build the (3, 3) pinhole ``K`` from one MVOR ``intrinsics`` entry."""
    (fx, fy), (cx, cy) = intr["focallength"], intr["principalpoint"]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _distortion(intr: dict) -> np.ndarray:
    """Raw MVOR ``[k1,k2,p1,p2,k3]`` distortion coefficients (or zeros)."""
    d = intr.get("distortion")
    return np.asarray(d, dtype=float).reshape(-1) if d else np.zeros(5)


def load_mvor_cameras(
    json_path: str | Path,
    depth_cam: int = 3,
    use_distortion: bool = False,
) -> list[CameraParams]:
    """MVOR ``camma_mvor_2018.json`` -> 3 ``CameraParams`` in the cam1 world frame.

    The camera at ``depth_cam`` (1-based ``cam_id``) is returned as a
    :class:`DepthCameraParams` (its colour ``K`` doubles as the aligned-depth
    intrinsic); the others are plain RGB :class:`CameraParams`. Camera order in
    the returned list is ``cam1, cam2, cam3`` so it aligns with
    :func:`iter_mvor_frames`'s colour order.
    """
    doc = json.load(open(json_path, encoding="utf-8"))
    cp = doc["cameras_info"]["camParams"]
    intrinsics, extrinsics = cp["intrinsics"], cp["extrinsics"]

    cameras: list[CameraParams] = []
    for i, intr in enumerate(intrinsics):
        cam_id = i + 1
        K = _intrinsic_matrix(intr)
        dist = _distortion(intr) if use_distortion else np.zeros(5)
        # extrinsics[i]: cam_i -> cam1 (mm). World == cam1, and CameraParams
        # extrinsics map world -> camera, so invert and convert mm -> m.
        world_to_cam = np.linalg.inv(np.asarray(extrinsics[i], float).reshape(4, 4))
        R = world_to_cam[:3, :3]
        t = world_to_cam[:3, 3] * _MM_TO_M
        w, h = (int(intr["imagesize"][0]), int(intr["imagesize"][1]))
        if cam_id == depth_cam:
            cameras.append(DepthCameraParams(
                f"cam{cam_id}_rgbd", K, dist, R, t, (w, h), depth_K=K))
        else:
            cameras.append(CameraParams(f"cam{cam_id}_rgb", K, dist, R, t, (w, h)))
    return cameras


@dataclass
class MVORFrame:
    """One synchronized MVOR multi-view frame.

    ``colors`` are the per-camera BGR images in ``cam1, cam2, cam3`` order (so
    ``colors[i]`` matches camera ``i`` of :func:`load_mvor_cameras`). ``depth_m``
    is the metric (meters) depth map of the chosen depth camera, aligned to its
    colour stream. ``frame_id`` is the MVOR multi-view triplet id.
    """

    colors: list[np.ndarray]
    depth_m: np.ndarray
    frame_id: str


def _read_triplets(json_path: str | Path, day: int | None) -> list[dict]:
    """Multi-view triplets (optionally for one ``day_id``), in capture order."""
    doc = json.load(open(json_path, encoding="utf-8"))
    triplets = doc["multiview_images"]
    if day is not None:
        triplets = [t for t in triplets if t["images"][0]["day_id"] == day]
    return sorted(triplets, key=lambda t: str(t["id"]))


def iter_mvor_frames(
    dataset_root: str | Path,
    json_path: str | Path,
    depth_cam: int = 3,
    day: int | None = 1,
    start: int = 0,
    num: int | None = None,
    depth_scale: float = 1000.0,
) -> Iterator[MVORFrame]:
    """Yield :class:`MVORFrame` objects from an extracted ``camma_mvor_dataset``.

    Reads the multi-view triplets from ``json_path`` (filtered to ``day``),
    slices ``[start : start+num]``, and for each triplet loads the three colour
    PNGs (``cam1, cam2, cam3``) plus the depth PNG of ``depth_cam`` (``/color/``
    -> ``/depth/`` in the path), converting raw 16-bit depth to meters via
    ``depth_scale``. Triplets with any unreadable colour/depth file are skipped.
    """
    root = Path(dataset_root)
    triplets = _read_triplets(json_path, day)
    end = None if num is None else start + num

    for triplet in triplets[start:end]:
        images = sorted(triplet["images"], key=lambda im: im["cam_id"])
        colors = [cv2.imread(str(root / im["file_name"])) for im in images]
        depth_rel = next(im["file_name"] for im in images if im["cam_id"] == depth_cam)
        depth_raw = cv2.imread(
            str(root / depth_rel.replace("/color/", "/depth/")), cv2.IMREAD_UNCHANGED)
        if any(c is None for c in colors) or depth_raw is None:
            continue
        depth_m = depth_raw.astype(np.float32) / float(depth_scale)
        yield MVORFrame(colors=colors, depth_m=depth_m, frame_id=str(triplet["id"]))
