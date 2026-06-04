"""End-to-end orchestration: 2D detection -> triangulation -> depth fusion ->
temporal smoothing -> 3D skeleton output.

Confidence flows through the whole chain: the 2D ``score`` of each keypoint is
used both as a triangulation weight (per view) and as a depth-fusion weight, so
occluded/low-score joints are automatically down-weighted (SPEC §5.3).

Distortion handling (documented design decision, SPEC §8.3): detected pixels are
undistorted with ``cv2.undistortPoints`` before triangulation, because ``P``
uses the pinhole model. Depth is assumed aligned to a rectified colour grid, so
the same undistorted pixels are used to sample the depth map and back-project.
For synthetic data (dist = 0) undistortion is the identity. A concrete RGB-D SDK
backend may refine this if its aligned depth lives in the distorted grid.
"""

from __future__ import annotations

import cv2
import numpy as np
import yaml

from src.calibration.camera_io import load_cameras_yaml
from src.core.types import NUM_KEYPOINTS, CameraParams, DepthCameraParams, Pose3D
from src.fusion.depth_fusion import back_project_depth_keypoints, fuse
from src.smoothing.one_euro import PoseSmoother
from src.triangulation.robust import triangulate_robust


def _empty_pose() -> Pose3D:
    """An all-invalid Pose3D placeholder (no joint reconstructed yet)."""
    k = NUM_KEYPOINTS
    return Pose3D(points=np.zeros((k, 3)), scores=np.zeros(k),
                  valid=np.zeros(k, dtype=bool), source=["none"] * k)


class Pipeline:
    """Configurable hybrid 3D-pose pipeline for a single person."""

    def __init__(self, config: dict, cameras: list[CameraParams]) -> None:
        self.config = config
        self.cameras = cameras
        self.proj_matrices = [cam.P for cam in cameras]
        self.depth_idx = next(
            (i for i, c in enumerate(cameras) if isinstance(c, DepthCameraParams)), None
        )

        tri = config.get("triangulation", {})
        self.score_threshold = float(tri.get("score_threshold", 0.3))
        self.min_views = int(tri.get("min_views", 2))
        ransac = tri.get("ransac", {}) or {}
        self.ransac = bool(ransac.get("enabled", False))
        self.reproj_threshold_px = float(ransac.get("reproj_threshold_px", 8.0))

        df = config.get("depth_fusion", {})
        self.fusion_enabled = bool(df.get("enabled", True))
        self.depth_min = float(df.get("depth_min", 0.2))
        self.depth_max = float(df.get("depth_max", 6.0))
        self.fill_missing = bool(df.get("fill_missing", True))
        self.patch_radius = int(df.get("patch_radius_px", 2))
        self.depth_weight = float(df.get("depth_weight", 1.0))

        sm = config.get("smoothing", {})
        if sm.get("enabled", True):
            self.smoother: PoseSmoother | None = PoseSmoother(
                num_keypoints=NUM_KEYPOINTS,
                freq=float(sm.get("freq", 30.0)),
                min_cutoff=float(sm.get("min_cutoff", 1.0)),
                beta=float(sm.get("beta", 0.007)),
                d_cutoff=float(sm.get("d_cutoff", 1.0)),
            )
        else:
            self.smoother = None

        self._detector = None  # built lazily (needs rtmlib)

    @classmethod
    def from_config(cls, config_path: str) -> "Pipeline":
        """Build a pipeline from a ``config/cameras.yaml``-schema file."""
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        cameras = load_cameras_yaml(config_path)
        return cls(config, cameras)

    # -- 2D detection (real mode; requires rtmlib) -------------------------
    def _detector_lazy(self):
        if self._detector is None:
            from src.pose2d.rtmpose_detector import RTMPoseDetector  # lazy: needs rtmlib

            det = self.config.get("detection", {})
            device = "cuda" if det.get("backend", "cuda") == "cuda" else "cpu"
            self._detector = RTMPoseDetector(
                device=device,
                mode=det.get("mode", "balanced"),
                score_threshold=float(det.get("det_score_threshold", 0.3)),
            )
        return self._detector

    def detect_2d(self, frameset) -> tuple[np.ndarray, np.ndarray]:
        """Run 2D pose on every view of a FrameSet -> (V,K,2) px, (V,K) scores."""
        detector = self._detector_lazy()
        n_views = len(self.cameras)
        keypoints = np.zeros((n_views, NUM_KEYPOINTS, 2), dtype=float)
        scores = np.zeros((n_views, NUM_KEYPOINTS), dtype=float)
        for i, cam in enumerate(self.cameras):
            pose = detector.detect_best(frameset.frames[cam.name])
            keypoints[i] = pose.keypoints
            scores[i] = pose.scores
        return keypoints, scores

    # -- numeric core (no rtmlib needed) -----------------------------------
    def _undistort(self, uv: np.ndarray, cam: CameraParams) -> np.ndarray:
        """Undistort (K,2) pixels into the same camera's pinhole pixel grid."""
        if np.allclose(cam.dist, 0.0):
            return np.asarray(uv, dtype=float)
        pts = np.asarray(uv, dtype=float).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, cam.K, cam.dist, P=cam.K)
        return out.reshape(-1, 2)

    def process(
        self,
        keypoints_per_view: np.ndarray,
        scores_per_view: np.ndarray,
        depth_map: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> Pose3D:
        """Reconstruct one frame's 3D pose from per-view 2D + optional depth.

        The reconstruction strategy is chosen by the available inputs:
          - **multi-view triangulation** when ``>= min_views`` camera views are
            configured (confidence-weighted DLT + robust view rejection);
          - **depth-only back-projection** when fewer views exist (e.g. a single
            RGB-D camera, world == camera) and a depth map is supplied;
          - **hybrid**: the triangulation result is fused with the depth view's
            back-projected points when both are available.

        Args:
            keypoints_per_view: (V,K,2) detected pixels (u,v), view order ==
                ``self.cameras``.
            scores_per_view: (V,K) confidence in [0,1].
            depth_map: (H,W) metric depth (meters) for the depth camera, or None.
            timestamp: optional capture time (seconds) for One-Euro smoothing.
        """
        undist = np.stack(
            [self._undistort(keypoints_per_view[i], self.cameras[i])
             for i in range(len(self.cameras))]
        )

        if len(self.cameras) >= self.min_views:
            result = triangulate_robust(
                undist, scores_per_view, self.proj_matrices,
                score_threshold=self.score_threshold, min_views=self.min_views,
                ransac=self.ransac, reproj_threshold_px=self.reproj_threshold_px,
            )
        else:
            # Too few views to triangulate; the depth path below reconstructs.
            result = _empty_pose()

        if self.fusion_enabled and depth_map is not None and self.depth_idx is not None:
            dcam = self.cameras[self.depth_idx]
            depth_points, depth_valid = back_project_depth_keypoints(
                undist[self.depth_idx], depth_map, dcam.depth_K, dcam.R, dcam.t,
                patch_radius=self.patch_radius,
                depth_min=self.depth_min, depth_max=self.depth_max,
            )
            # Gate depth by the depth view's 2D confidence: an occluded joint's
            # garbage pixel can land on valid background depth, so a low 2D score
            # must invalidate that depth sample too (SPEC §5.3/§5.4). Without this
            # gate, fill_missing would accept the spurious back-projected point.
            depth_scores = scores_per_view[self.depth_idx]
            depth_valid = depth_valid & (depth_scores >= self.score_threshold)
            result = fuse(
                result, depth_points, depth_valid, depth_scores,
                fill_missing=self.fill_missing, depth_weight=self.depth_weight,
            )

        if self.smoother is not None:
            result = self.smoother.update(result, timestamp)
        return result
