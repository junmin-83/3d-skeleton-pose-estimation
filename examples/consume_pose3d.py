"""후속 프로그램이 파이프라인을 인프로세스로 써서 Pose3D를 소비하는 최소 예제.

파일을 거치지 않고 ``Pipeline.process(...)`` 가 돌려주는 ``Pose3D`` 를 바로 쓰는 패턴만
보여준다. 실데이터/rtmlib/GPU 없이 오프라인으로 돌도록, 알려진 3D COCO-17을 2개 RGB 뷰에
투영해 2D 입력을 합성한다(실제로는 이 2D를 검출기나 ``pipe.detect_2d`` 로 얻는다).

Usage::

    uv run python examples/consume_pose3d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry import project_points  # noqa: E402
from src.core.types import COCO_17_KEYPOINTS, CameraParams  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

NUM_KPTS = len(COCO_17_KEYPOINTS)


def _demo_cameras() -> list[CameraParams]:
    """RGB 2뷰(0.6 m 기준선)를 한 world 좌표계에 둔다 (실제로는 캘리브 결과)."""
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    z5 = np.zeros(5)
    return [
        CameraParams("cam0", K, z5, np.eye(3), np.zeros(3), (640, 480)),
        CameraParams("cam1", K, z5, np.eye(3), np.array([-0.6, 0.0, 0.0]), (640, 480)),
    ]


def _synthetic_2d(cameras: list[CameraParams], world: np.ndarray):
    """알려진 3D를 각 뷰에 투영해 (V,17,2) 2D + (V,17) score를 만든다 (검출기 대역)."""
    kpts = np.stack([project_points(c.P, world) for c in cameras])
    scores = np.full((len(cameras), NUM_KPTS), 0.9)
    return kpts, scores


def main() -> None:
    cameras = _demo_cameras()
    config = {
        "triangulation": {"score_threshold": 0.3, "min_views": 2},
        "depth_fusion": {"enabled": False},
        "smoothing": {"enabled": False},
    }
    # 1) 파이프라인 구성. 실제로는 Pipeline.from_config("config/cameras.yaml").
    pipe = Pipeline(config, cameras)

    # 알려진 3D COCO-17 (world, meter). 실제로는 카메라에서 2D를 검출한다.
    rng = np.random.default_rng(0)
    world_gt = rng.uniform(-0.4, 0.4, (NUM_KPTS, 3)) + np.array([0.0, 0.0, 3.0])

    # 2) 프레임마다 2D를 만들어(또는 검출해) 넣고 Pose3D를 받는다.
    keypoints, scores = _synthetic_2d(cameras, world_gt)
    pose = pipe.process(keypoints, scores, depth_map=None, timestamp=0.0)

    # 3) Pose3D를 바로 소비. 이 블록이 후속 프로그램이 하는 일이다.
    print(f"reconstructed {int(pose.valid.sum())}/{NUM_KPTS} joints (world frame, meters):")
    for k, name in enumerate(COCO_17_KEYPOINTS):
        if not pose.valid[k]:          # 반드시 valid 확인 (invalid은 NaN일 수 있음)
            continue
        x, y, z = pose.points[k]
        print(f"  {k:2d} {name:15s} x={x:+.3f} y={y:+.3f} z={z:+.3f}  "
              f"score={pose.scores[k]:.2f} src={pose.source[k]}")

    err = float(np.linalg.norm(pose.points[pose.valid] - world_gt[pose.valid], axis=1).max())
    print(f"max error vs ground-truth 3D: {err * 1000:.3f} mm  (round-trip sanity check)")


if __name__ == "__main__":
    main()
