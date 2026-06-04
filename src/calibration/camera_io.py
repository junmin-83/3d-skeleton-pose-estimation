"""Camera-parameter serialization to/from the config/cameras.yaml schema.

Round-trips intrinsics, WORLD -> CAMERA extrinsics, image size, source, and
(for rgbd cameras) the depth fields. Lengths in meters; pixels are (u, v).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.core.types import CameraParams, DepthCameraParams


def _camera_to_dict(cam: CameraParams) -> dict:
    """Serialize one camera to the config/cameras.yaml schema."""
    is_rgbd = isinstance(cam, DepthCameraParams)
    entry: dict = {
        "name": cam.name,
        "type": "rgbd" if is_rgbd else "rgb",
        "K": cam.K.tolist(),
        "dist": cam.dist.tolist(),
        "R": cam.R.tolist(),
        "t": cam.t.tolist(),
        "image_size": [int(cam.image_size[0]), int(cam.image_size[1])],
        "source": getattr(cam, "source", None),
    }
    if is_rgbd:
        entry["depth_K"] = np.asarray(cam.depth_K, float).reshape(3, 3).tolist()
        entry["depth_scale"] = float(cam.depth_scale)
        d2c_R = cam.depth_to_color_R if cam.depth_to_color_R is not None else np.eye(3)
        d2c_t = cam.depth_to_color_t if cam.depth_to_color_t is not None else np.zeros(3)
        entry["depth_to_color_R"] = np.asarray(d2c_R, float).reshape(3, 3).tolist()
        entry["depth_to_color_t"] = np.asarray(d2c_t, float).reshape(3).tolist()
    return entry


def save_cameras_yaml(
    cameras: list[CameraParams],
    path: str | Path,
    units: str = "meter",
    world_frame: str = "reference_camera",
    reference: str = "cam0",
) -> None:
    """Write cameras to path in the config/cameras.yaml schema.

    Top-level keys: units, world, cameras. Each camera carries name, type, K,
    dist, R, t, image_size, source, plus the depth fields (depth_K,
    depth_scale, depth_to_color_R, depth_to_color_t) for rgbd cameras.
    """
    doc = {
        "units": {"length": units},
        "world": {"frame": world_frame, "reference_camera": reference},
        "cameras": [_camera_to_dict(cam) for cam in cameras],
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=None)


def load_cameras_yaml(path: str | Path) -> list[CameraParams]:
    """Load cameras from a config/cameras.yaml-schema file.

    Returns:
        list of CameraParams; rgbd cameras come back as DepthCameraParams with
        their depth fields restored.
    """
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    cameras: list[CameraParams] = []
    for entry in doc.get("cameras", []):
        cam_type = entry.get("type", "rgb")
        image_size = tuple(int(v) for v in entry["image_size"])
        if cam_type == "rgbd":
            cam: CameraParams = DepthCameraParams(
                name=entry["name"],
                K=np.asarray(entry["K"], float),
                dist=np.asarray(entry["dist"], float),
                R=np.asarray(entry["R"], float),
                t=np.asarray(entry["t"], float),
                image_size=image_size,
                depth_K=np.asarray(entry["depth_K"], float),
                depth_scale=float(entry["depth_scale"]),
                depth_to_color_R=np.asarray(entry["depth_to_color_R"], float),
                depth_to_color_t=np.asarray(entry["depth_to_color_t"], float),
            )
        else:
            cam = CameraParams(
                name=entry["name"],
                K=np.asarray(entry["K"], float),
                dist=np.asarray(entry["dist"], float),
                R=np.asarray(entry["R"], float),
                t=np.asarray(entry["t"], float),
                image_size=image_size,
            )
        # source isn't a dataclass field; attach it so the round-trip preserves it.
        cam.source = entry.get("source")
        cameras.append(cam)
    return cameras
