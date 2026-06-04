"""Serialize 3D pose sequences to/from JSON and NPY.

Split from the 3D plotting code (src/render/skeleton_3d.py) so result I/O
doesn't pull in matplotlib. Units: meters, world frame, COCO-17.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.core.types import Pose3D


def export_keypoints(
    poses: "Pose3D | Sequence[Pose3D]",
    path: str,
    fmt: str = "json",
) -> None:
    """Export one or more poses (one per frame) to "json" or "npy".

    For npy, points go to <path> (float64, (N, K, 3)) and a sidecar
    <path>.npz holds scores (N, K), valid (N, K), source (N, K).

    JSON is an array of dicts: {"points": K x 3, "scores": K, "valid": K bool,
    "source": K str}.
    """
    # One pose or many; normalize to a list.
    if isinstance(poses, Pose3D):
        pose_list: list[Pose3D] = [poses]
    else:
        pose_list = list(poses)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        records = []
        for p in pose_list:
            records.append(
                {
                    "points": p.points.tolist(),
                    "scores": p.scores.tolist(),
                    "valid": p.valid.tolist(),
                    "source": list(p.source),
                }
            )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)

    elif fmt == "npy":
        points_arr = np.stack([p.points for p in pose_list], axis=0)  # (N, K, 3)
        scores_arr = np.stack([p.scores for p in pose_list], axis=0)  # (N, K)
        valid_arr = np.stack([p.valid for p in pose_list], axis=0)    # (N, K)
        source_arr = np.array(
            [p.source for p in pose_list], dtype=object
        )  # (N, K) or (N, len(source))

        np.save(str(out_path), points_arr)
        sidecar = out_path.with_suffix(".npy.npz")
        np.savez(
            str(sidecar),
            scores=scores_arr,
            valid=valid_arr,
            source=source_arr,
        )
    else:
        raise ValueError(f"Unsupported export format: {fmt!r}. Use 'json' or 'npy'.")


def load_keypoints(path: str, fmt: str = "json") -> list[Pose3D]:
    """Load poses (one Pose3D per frame) written by export_keypoints; fmt is "json" or "npy"."""
    in_path = Path(path)

    if fmt == "json":
        with open(in_path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        poses = []
        for rec in records:
            poses.append(
                Pose3D(
                    points=np.array(rec["points"], dtype=float),
                    scores=np.array(rec["scores"], dtype=float),
                    valid=np.array(rec["valid"], dtype=bool),
                    source=list(rec.get("source", [])),
                )
            )
        return poses

    elif fmt == "npy":
        points_arr = np.load(str(in_path))          # (N, K, 3)
        sidecar = in_path.with_suffix(".npy.npz")
        data = np.load(str(sidecar), allow_pickle=True)
        scores_arr = data["scores"]   # (N, K)
        valid_arr = data["valid"]     # (N, K)
        source_arr = data["source"]   # (N, ...)

        poses = []
        for i in range(len(points_arr)):
            src = list(source_arr[i]) if i < len(source_arr) else []
            poses.append(
                Pose3D(
                    points=points_arr[i],
                    scores=scores_arr[i],
                    valid=valid_arr[i].astype(bool),
                    source=src,
                )
            )
        return poses

    else:
        raise ValueError(f"Unsupported load format: {fmt!r}. Use 'json' or 'npy'.")
