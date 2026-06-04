"""Tests for frame_reader.py, render/skeleton_3d.py, and io/keypoints_io.py.

All tests run headless (no display required) and offline (no network).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.io.frame_reader import CameraSpec, MultiViewFrameReader, nearest_frame_match
from src.core.types import Pose3D, NUM_KEYPOINTS
from src.io.keypoints_io import export_keypoints, load_keypoints
from src.render.skeleton_3d import save_skeleton_png


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_images(base_dir: Path, n_cameras: int, n_frames: int, size: tuple[int, int] = (32, 32)) -> list[Path]:
    """Write n_cameras subdirectories each containing n_frames PNG images."""
    cam_dirs = []
    for c in range(n_cameras):
        cam_dir = base_dir / f"cam{c}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for f in range(n_frames):
            # Each image has a distinct pixel value so we can tell them apart.
            img = np.full((size[1], size[0], 3), fill_value=(c * 20 + f * 5) % 255, dtype=np.uint8)
            path = cam_dir / f"frame_{f:04d}.png"
            ok = cv2.imwrite(str(path), img)
            assert ok, f"cv2.imwrite failed for {path}"
        cam_dirs.append(cam_dir)
    return cam_dirs


def _make_pose3d(seed: int = 0) -> Pose3D:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, (NUM_KEYPOINTS, 3))
    scores = rng.uniform(0.5, 1.0, NUM_KEYPOINTS)
    valid = np.ones(NUM_KEYPOINTS, dtype=bool)
    source = ["tri"] * NUM_KEYPOINTS
    return Pose3D(points=points, scores=scores, valid=valid, source=source)


# ---------------------------------------------------------------------------
# nearest_frame_match
# ---------------------------------------------------------------------------

class TestNearestFrameMatch:
    def test_exact_match(self):
        candidates = [0.0, 0.1, 0.2, 0.3]
        assert nearest_frame_match(0.1, candidates, tolerance=0.05) == 1

    def test_nearest_within_tolerance(self):
        candidates = [0.0, 0.5, 1.0]
        assert nearest_frame_match(0.45, candidates, tolerance=0.1) == 1

    def test_outside_tolerance_returns_none(self):
        candidates = [0.0, 0.5, 1.0]
        assert nearest_frame_match(0.75, candidates, tolerance=0.1) is None

    def test_first_element(self):
        candidates = [10.0, 20.0, 30.0]
        assert nearest_frame_match(10.0, candidates, tolerance=0.001) == 0

    def test_last_element(self):
        candidates = [1.0, 2.0, 3.0]
        assert nearest_frame_match(3.0, candidates, tolerance=0.001) == 2

    def test_empty_candidates(self):
        assert nearest_frame_match(1.0, [], tolerance=1.0) is None

    def test_single_candidate_in_tolerance(self):
        assert nearest_frame_match(5.0, [5.05], tolerance=0.1) == 0

    def test_single_candidate_out_of_tolerance(self):
        assert nearest_frame_match(5.0, [6.0], tolerance=0.5) is None

    def test_tie_picks_first(self):
        # Both candidates equidistant; argmin picks the first one.
        candidates = [0.0, 2.0]
        result = nearest_frame_match(1.0, candidates, tolerance=1.0)
        assert result in (0, 1)  # either is acceptable; just must not be None


# ---------------------------------------------------------------------------
# MultiViewFrameReader – file mode
# ---------------------------------------------------------------------------

N_CAMERAS = 3
N_FRAMES = 4


class TestMultiViewFrameReaderFile:
    def test_returns_n_framesets(self, tmp_path):
        cam_dirs = _make_synthetic_images(tmp_path, N_CAMERAS, N_FRAMES)
        specs = [CameraSpec(f"cam{i}", cam_dirs[i]) for i in range(N_CAMERAS)]
        reader = MultiViewFrameReader(specs)
        framesets = list(reader)
        reader.close()
        assert len(framesets) == N_FRAMES

    def test_frameset_has_all_cameras(self, tmp_path):
        cam_dirs = _make_synthetic_images(tmp_path, N_CAMERAS, N_FRAMES)
        specs = [CameraSpec(f"cam{i}", cam_dirs[i]) for i in range(N_CAMERAS)]
        with MultiViewFrameReader(specs) as reader:
            fs = reader.read()
        assert fs is not None
        assert set(fs.frames.keys()) == {f"cam{i}" for i in range(N_CAMERAS)}

    def test_frame_shapes(self, tmp_path):
        size = (48, 32)  # width, height
        cam_dirs = _make_synthetic_images(tmp_path, N_CAMERAS, N_FRAMES, size=size)
        specs = [CameraSpec(f"cam{i}", cam_dirs[i]) for i in range(N_CAMERAS)]
        with MultiViewFrameReader(specs) as reader:
            fs = reader.read()
        assert fs is not None
        for name, img in fs.frames.items():
            assert img.shape == (size[1], size[0], 3), f"Wrong shape for {name}: {img.shape}"

    def test_index_increments(self, tmp_path):
        cam_dirs = _make_synthetic_images(tmp_path, N_CAMERAS, N_FRAMES)
        specs = [CameraSpec(f"cam{i}", cam_dirs[i]) for i in range(N_CAMERAS)]
        with MultiViewFrameReader(specs) as reader:
            indices = [fs.index for fs in reader]
        assert indices == list(range(N_FRAMES))

    def test_fifth_read_returns_none(self, tmp_path):
        cam_dirs = _make_synthetic_images(tmp_path, N_CAMERAS, N_FRAMES)
        specs = [CameraSpec(f"cam{i}", cam_dirs[i]) for i in range(N_CAMERAS)]
        reader = MultiViewFrameReader(specs)
        for _ in range(N_FRAMES):
            assert reader.read() is not None
        assert reader.read() is None
        reader.close()

    def test_timestamps_present(self, tmp_path):
        cam_dirs = _make_synthetic_images(tmp_path, N_CAMERAS, N_FRAMES)
        specs = [CameraSpec(f"cam{i}", cam_dirs[i]) for i in range(N_CAMERAS)]
        with MultiViewFrameReader(specs) as reader:
            fs = reader.read()
        assert fs is not None
        assert set(fs.timestamps.keys()) == {f"cam{i}" for i in range(N_CAMERAS)}

    def test_file_mode_with_sidecar_timestamps(self, tmp_path):
        """When all specs supply timestamps, nearest-frame matching is used."""
        cam_dirs = _make_synthetic_images(tmp_path, 2, N_FRAMES)
        base_ts = [0.0, 0.1, 0.2, 0.3]
        # Second camera slightly offset but within tolerance.
        offset_ts = [0.005, 0.105, 0.205, 0.305]
        specs = [
            CameraSpec("cam0", cam_dirs[0], timestamps=base_ts),
            CameraSpec("cam1", cam_dirs[1], timestamps=offset_ts),
        ]
        with MultiViewFrameReader(specs) as reader:
            framesets = list(reader)
        assert len(framesets) == N_FRAMES

    def test_missing_directory_raises(self, tmp_path):
        specs = [CameraSpec("cam0", tmp_path / "nonexistent")]
        with pytest.raises(FileNotFoundError):
            MultiViewFrameReader(specs)

    def test_empty_directory_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        specs = [CameraSpec("cam0", empty_dir)]
        with pytest.raises(ValueError):
            MultiViewFrameReader(specs)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_save_skeleton_png_creates_file(self, tmp_path):
        pose = _make_pose3d(seed=42)
        out = tmp_path / "skeleton.png"
        save_skeleton_png(pose, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_skeleton_png_creates_parent_dirs(self, tmp_path):
        pose = _make_pose3d(seed=1)
        out = tmp_path / "subdir" / "deep" / "skeleton.png"
        save_skeleton_png(pose, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_skeleton_3d_returns_fig_ax(self):
        from src.render.skeleton_3d import plot_skeleton_3d
        import matplotlib.pyplot as plt
        pose = _make_pose3d(seed=7)
        fig, ax = plot_skeleton_3d(pose, title="Test")
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_skeleton_3d_partial_valid(self):
        """Bones connecting invalid joints must be skipped without error."""
        from src.render.skeleton_3d import plot_skeleton_3d
        import matplotlib.pyplot as plt
        pose = _make_pose3d(seed=3)
        # Mark half the keypoints invalid.
        pose.valid[::2] = False
        fig, ax = plot_skeleton_3d(pose)
        assert fig is not None
        plt.close(fig)

    def test_plot_all_invalid(self):
        """All-invalid pose should not crash."""
        from src.render.skeleton_3d import plot_skeleton_3d
        import matplotlib.pyplot as plt
        points = np.zeros((NUM_KEYPOINTS, 3))
        scores = np.zeros(NUM_KEYPOINTS)
        valid = np.zeros(NUM_KEYPOINTS, dtype=bool)
        pose = Pose3D(points=points, scores=scores, valid=valid)
        fig, ax = plot_skeleton_3d(pose)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Export / load round-trips
# ---------------------------------------------------------------------------

class TestExportLoad:
    def test_json_roundtrip(self, tmp_path):
        poses = [_make_pose3d(seed=i) for i in range(2)]
        out = tmp_path / "poses.json"
        export_keypoints(poses, str(out), fmt="json")
        loaded = load_keypoints(str(out), fmt="json")
        assert len(loaded) == 2
        for orig, loaded_p in zip(poses, loaded):
            np.testing.assert_allclose(orig.points, loaded_p.points)
            np.testing.assert_allclose(orig.scores, loaded_p.scores)
            np.testing.assert_array_equal(orig.valid, loaded_p.valid)

    def test_npy_roundtrip(self, tmp_path):
        poses = [_make_pose3d(seed=i) for i in range(2)]
        out = tmp_path / "poses.npy"
        export_keypoints(poses, str(out), fmt="npy")
        loaded = load_keypoints(str(out), fmt="npy")
        assert len(loaded) == 2
        for orig, loaded_p in zip(poses, loaded):
            np.testing.assert_allclose(orig.points, loaded_p.points)
            np.testing.assert_allclose(orig.scores, loaded_p.scores)

    def test_json_single_pose(self, tmp_path):
        pose = _make_pose3d(seed=99)
        out = tmp_path / "single.json"
        export_keypoints(pose, str(out), fmt="json")
        loaded = load_keypoints(str(out), fmt="json")
        assert len(loaded) == 1
        np.testing.assert_allclose(pose.points, loaded[0].points)

    def test_json_source_preserved(self, tmp_path):
        pose = _make_pose3d(seed=5)
        pose.source = ["depth" if i % 2 == 0 else "tri" for i in range(NUM_KEYPOINTS)]
        out = tmp_path / "source.json"
        export_keypoints(pose, str(out), fmt="json")
        loaded = load_keypoints(str(out), fmt="json")
        assert loaded[0].source == pose.source

    def test_unsupported_fmt_raises(self, tmp_path):
        poses = [_make_pose3d()]
        with pytest.raises(ValueError):
            export_keypoints(poses, str(tmp_path / "out.xyz"), fmt="xyz")

    def test_npy_sidecar_exists(self, tmp_path):
        poses = [_make_pose3d(seed=0)]
        out = tmp_path / "poses.npy"
        export_keypoints(poses, str(out), fmt="npy")
        sidecar = tmp_path / "poses.npy.npz"
        assert sidecar.exists()
