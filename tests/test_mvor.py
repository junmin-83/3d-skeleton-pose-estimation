"""Tests for the MVOR dataset adapter (src/io/sources/mvor.py).

All offline and synthetic — no MVOR download required. A tiny in-memory
``camma_mvor_2018.json`` plus a few generated PNGs exercise the calibration
loader (intrinsics, inverted extrinsics, mm -> m, depth-camera typing) and the
multi-view frame iterator (colour order, depth scaling, frame slicing).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.core.types import CameraParams, DepthCameraParams
from src.io.sources.mvor import iter_mvor_frames, load_mvor_cameras


def _intr(fx, fy, cx, cy, w=640, h=480):
    return {"focallength": [fx, fy], "principalpoint": [cx, cy],
            "imagesize": [w, h], "distortion": [0.1, -0.2, 0.0, 0.0, 0.3]}


def _flat(mat: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(mat).reshape(-1)]


def _write_json(tmp: Path, extrinsics) -> Path:
    """A minimal MVOR annotation JSON with 3 cameras + one triplet."""
    cam_params = {
        "intrinsics": [_intr(500, 501, 320, 240), _intr(510, 511, 321, 241),
                       _intr(520, 521, 322, 242)],
        "extrinsics": extrinsics,
        "ids": ["a", "b", "c"],
        "firstCamToRoomRef": _flat(np.eye(4)),
    }
    doc = {"cameras_info": {"camParams": cam_params}, "multiview_images": []}
    path = tmp / "camma_mvor_2018.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_load_cameras_intrinsics_and_world_frame(tmp_path):
    # cam2 is a pure +x translation of 600 mm expressed as cam2 -> cam1.
    e2 = np.eye(4)
    e2[:3, 3] = [600.0, 0.0, 0.0]
    json_path = _write_json(tmp_path, [_flat(np.eye(4)), _flat(e2), _flat(np.eye(4))])

    cams = load_mvor_cameras(json_path, depth_cam=3)
    assert [c.name for c in cams] == ["cam1_rgb", "cam2_rgb", "cam3_rgbd"]

    # Intrinsics map straight through focallength/principalpoint.
    np.testing.assert_allclose(cams[0].K, [[500, 0, 320], [0, 501, 240], [0, 0, 1]])

    # World == cam1 -> reference camera has identity extrinsics.
    np.testing.assert_allclose(cams[0].R, np.eye(3))
    np.testing.assert_allclose(cams[0].t, np.zeros(3), atol=1e-12)

    # extrinsics map cam_i -> cam1, so world->cam is the inverse; 600 mm -> 0.6 m,
    # and the inverse of a +x shift is a -x shift.
    np.testing.assert_allclose(cams[1].R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(cams[1].t, [-0.6, 0.0, 0.0], atol=1e-9)


def test_depth_camera_typing_and_selection(tmp_path):
    json_path = _write_json(tmp_path, [_flat(np.eye(4))] * 3)

    cams = load_mvor_cameras(json_path, depth_cam=2)
    assert isinstance(cams[1], DepthCameraParams)
    assert not isinstance(cams[0], DepthCameraParams)
    assert isinstance(cams[0], CameraParams)
    # Aligned depth: depth_K defaults to the colour K.
    np.testing.assert_allclose(cams[1].depth_K, cams[1].K)


def test_distortion_opt_in(tmp_path):
    json_path = _write_json(tmp_path, [_flat(np.eye(4))] * 3)
    assert np.allclose(load_mvor_cameras(json_path)[0].dist, 0.0)
    raw = load_mvor_cameras(json_path, use_distortion=True)[0].dist
    np.testing.assert_allclose(raw, [0.1, -0.2, 0.0, 0.0, 0.3])


def _make_dataset(tmp_path: Path, n_frames: int, depth_raw: int) -> Path:
    """Write a tiny camma_mvor_dataset tree + matching multiview_images JSON."""
    root = tmp_path / "camma_mvor_dataset"
    triplets = []
    for f in range(n_frames):
        images = []
        for cam in (1, 2, 3):
            rel = f"day1/cam{cam}/color/{f:06d}.png"
            color_p = root / rel
            depth_p = root / rel.replace("/color/", "/depth/")
            color_p.parent.mkdir(parents=True, exist_ok=True)
            depth_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(color_p), np.full((480, 640, 3), cam * 30, np.uint8))
            cv2.imwrite(str(depth_p), np.full((480, 640), depth_raw, np.uint16))
            images.append({"cam_id": cam, "day_id": 1, "file_name": rel,
                           "id": int(f"100{cam}{f:06d}")})
        triplets.append({"id": f"t{f}", "images": images})

    doc = {"cameras_info": {"camParams": {
        "intrinsics": [_intr(500, 500, 320, 240)] * 3,
        "extrinsics": [_flat(np.eye(4))] * 3, "ids": ["a", "b", "c"]}},
        "multiview_images": triplets}
    json_path = tmp_path / "camma_mvor_2018.json"
    json_path.write_text(json.dumps(doc), encoding="utf-8")
    return root


def test_iter_frames_colour_order_and_depth_scale(tmp_path):
    root = _make_dataset(tmp_path, n_frames=3, depth_raw=2500)
    json_path = tmp_path / "camma_mvor_2018.json"

    frames = list(iter_mvor_frames(root, json_path, depth_cam=3, depth_scale=1000.0))
    assert len(frames) == 3

    fr = frames[0]
    assert len(fr.colors) == 3
    # Colours are returned in cam1, cam2, cam3 order (fill values 30, 60, 90).
    assert [int(c[0, 0, 0]) for c in fr.colors] == [30, 60, 90]
    # 16-bit raw 2500 / 1000 -> 2.5 m, float32, full color resolution.
    assert fr.depth_m.dtype == np.float32
    np.testing.assert_allclose(fr.depth_m[0, 0], 2.5)
    assert fr.depth_m.shape == (480, 640)


def test_iter_frames_start_and_num_slicing(tmp_path):
    root = _make_dataset(tmp_path, n_frames=5, depth_raw=1000)
    json_path = tmp_path / "camma_mvor_2018.json"

    frames = list(iter_mvor_frames(root, json_path, start=1, num=2))
    assert [fr.frame_id for fr in frames] == ["t1", "t2"]
