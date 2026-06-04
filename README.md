# 3D 스켈레톤 포즈 추정 (RGB 2대 + RGB-D 1대)

RGB 카메라 2대와 RGB-D 카메라 1대로 한 사람의 **3D 스켈레톤 포즈**(COCO-17)를 추정합니다.
단안(monocular) 추정의 depth 모호성은 다중 뷰 **confidence 가중 삼각측량(triangulation)**으로
해소하고, depth 센서로 scale을 보정하며 **가려진 관절을 보완**하는 하이브리드 방식입니다.

기술 스택(고정): [rtmlib](https://github.com/Tau-J/rtmlib) (RTMPose ONNX,
`mmcv`/`mmpose`/`mmdet` 미사용) · onnxruntime-gpu · OpenCV · NumPy · SciPy · Python 3.10+.
학습은 하지 않으며 추론만 합니다.

---

## 파이프라인

```
[동기화된 3개 프레임]
   │ (1) 뷰별 RTMPose 2D 키포인트              src/pose2d/rtmpose_detector.py
   │ (2) calibration -> projection matrix       src/calibration/calibrate.py
   │ (3) 삼각측량 (confidence DLT + robust 제외) src/triangulation/{dlt,robust}.py
   │ (4) depth fusion (back-projection + 융합)   src/fusion/depth_fusion.py
   │ (5) 시간적 스무딩 (One-Euro)               src/smoothing/one_euro.py
   ▼
[3D 스켈레톤 + 시각화/저장]                      src/viz/visualize_3d.py
```

오케스트레이션: `src/pipeline.py` · 엔트리포인트: `run.py`.

---

## 설치 (동일 환경 재현)

[uv](https://github.com/astral-sh/uv)만 있으면 됩니다. 클론 후 **`uv.lock`으로 모든 머신에서
정확히 같은 버전**이 설치됩니다(Python은 `.python-version`의 3.11을 자동 사용).

```bash
git clone https://github.com/junmin-83/3d-skeleton-pose-estimation
cd 3d-skeleton-pose-estimation
uv sync          # pyproject.toml + uv.lock 의 고정 버전 그대로 설치
```

- `uv sync`는 lock에 고정된 정확한 버전(numpy/scipy/opencv/onnxruntime/rtmlib + dev: pytest/ruff)을
  설치 → **재현성 보장**. 이후 `uv run ...` 은 실행 시 환경을 자동 동기화합니다.
- **GPU(NVIDIA CUDA):** 기본은 CPU용 `onnxruntime`입니다. GPU 가속은 sync 후
  `uv pip uninstall onnxruntime && uv pip install onnxruntime-gpu` 실행하고
  config의 `detection.backend: cuda`(기본값)를 유지하세요.
- **모델 캐시:** rtmlib RTMPose ONNX 모델은 `TORCH_HOME=./models`로 지정돼
  **`./models/hub/checkpoints/`** 에 받습니다(`src/pose2d/rtmpose_detector.py`에 하드코딩, env로
  override 가능). `models/`는 `.gitignore`. 상대경로이므로 **프로젝트 루트에서 실행**하세요.
- (대안) pip 스타일: `uv venv --python 3.11 && uv pip install -r requirements.txt`
  — `requirements.txt`는 GPU(`onnxruntime-gpu`) 기준이며, **정확 재현은 `uv sync`를 권장**합니다.

---

## 빠른 시작 (다운로드·GPU 불필요)

설치 직후 바로 동작합니다(rtmlib/카메라 불필요 — 합성 데이터로 실제 파이프라인을 구동):

```bash
# 합성 3D 복원 + 시각화 (전체 파이프라인 검증)
uv run python run.py --synthetic --frames 30 --viz
#   -> output/poses_3d.json + output/skeleton_frame0.png

# 단위/통합 테스트
uv run pytest tests/ -q
```

---

## 데모 한눈에

> 모든 명령은 **프로젝트 루트**에서 실행하며, 산출물은 `output/`에 저장됩니다.

| 데모 | 명령 (`examples/` 또는 `run.py`) | 보여주는 것 | 입력 | 다운로드 | GPU |
|---|---|---|---|---|---|
| **A. 실시간 2D 추출** | `realtime_demo.py` | 17 키포인트 + FPS, 오버레이 MP4 | **실제** RTMPose | 샘플 이미지(소)/웹캠 | 권장 |
| **B-1. 합성 3D 검증** | `run.py --synthetic` | 전체 파이프라인 정확도 | 합성 | 없음 | 불필요 |
| **B-2. 3D 영상** | `pose3d_video.py` | 3D 스켈레톤 MP4 | 합성 | 없음 | 불필요 |
| **B-3. 하이브리드 4분할** | `hybrid_3cam_demo.py` | 2RGB+1Depth→3D MP4 | 합성 | 없음 | 불필요 |
| **C-1. 실제 calib+GT** | `panoptic_demo.py` | 실제 calibration 복원 정확도 | **실제** calib + 투영 2D | 작음 | 불필요 |
| **C-2. 실제 HD 영상** | `panoptic_video_demo.py` | 실영상 RTMPose→3D MP4 | **실제** RTMPose + calib | 큼(~8.6GB) | 권장 |
| **D. 실제 라이브** | `run.py` | 자체 리그 실시간 3D | **실제** | 자체 calib | 권장 |

> **합성 vs 실제 (공통):** 3D 복원은 원리상 **calibration된 2대 이상의 동기화 뷰**가 필요합니다
> (단일 뷰로는 depth 모호성을 못 풂). `B-*`는 검증을 위해 **실제 파이프라인 코드**를 합성 2D/depth로
> 구동합니다. 실제 입력으로 바꾸려면 각 뷰의 2D를 `RTMPoseDetector(...).detect_best(frame).keypoints`로,
> depth를 실제 깊이맵(미터)으로 넣으면 동일 코드가 그대로 동작합니다(다인 장면은 뷰 간 인물 매칭 추가 필요 — 미구현).

---

## A. 실시간 2D 키포인트 추출 (`examples/realtime_demo.py`)

단일 뷰 이미지/웹캠에서 **COCO-17 17개 키포인트를 프레임마다 추출**(실제 RTMPose)하고, FPS와
좌표를 출력하며, 스켈레톤 오버레이 **MP4(전 프레임)** + 마지막 프레임 PNG를 저장합니다.

```bash
# (a) 정적 이미지를 스트림처럼 반복 — 가장 간단
uv run python examples/realtime_demo.py --frames 30
#   -> output/realtime_keypoints.mp4 + output/realtime_keypoints.png

# (b) 실제 웹캠 라이브 (장치 0번, 200프레임 전체가 MP4로 저장)
uv run python examples/realtime_demo.py --camera 0 --frames 200

# (c) GPU 고속 (NVIDIA + onnxruntime-gpu, 30+ FPS)
uv run python examples/realtime_demo.py --camera 0 --device cuda --mode balanced
```

옵션: `--image <경로>` · `--mode lightweight|balanced|performance` · `--score-thr <0~1>`
(올리면 저신뢰/오검출 관절 제외) · `--video/--out/--fps`.

```bash
# 데모용 샘플 사진이 없을 때(사람이 있는 임의 이미지면 가능)
curl -sSL -o data/demo/person.jpg "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg"
```

참고 성능: CPU(RTMPose-s, lightweight) 약 **11~12 FPS**, GPU 30+ FPS.

---

## B. 3D 복원 — 합성 데이터 (다운로드·카메라 불필요)

### B-1. 전체 파이프라인 검증 (`run.py --synthetic`)

알려진 3D 스켈레톤을 config 카메라들로 투영 → 전체 파이프라인(삼각측량→depth fusion→스무딩)으로 복원.

```bash
uv run python run.py --synthetic --frames 30 --viz   # JSON + PNG
uv run python run.py --synthetic --frames 30         # 시각화 없이 빠르게
```

기대 출력: `mean3D_err(frame0) ≈ 8.7e-09 m` (수치 정밀도 수준 → 삼각측량·fusion 정확).
시각화에서 사람이 누워 보이는 건 카메라 좌표계 관례(+Y 아래)일 뿐 기하학적으로 정상.

### B-2. 3D 스켈레톤 영상 (`examples/pose3d_video.py`)

복원된 3D 스켈레톤을 프레임마다 렌더링해 MP4로 저장.

```bash
uv run python examples/pose3d_video.py --frames 90              # output/pose3d_demo.mp4
uv run python examples/pose3d_video.py --frames 120 --jitter 0.01   # 노이즈→스무딩 효과
```

### B-3. 2 RGB + 1 Depth 하이브리드 4분할 (`examples/hybrid_3cam_demo.py`)

리그 구조(RGB 2 + Depth 1)를 모사해 전체 하이브리드 파이프라인을 돌리고
`[cam0 RGB | cam1 RGB / cam2 Depth | 3D 복원]` 4분할 MP4를 생성.

```bash
uv run python examples/hybrid_3cam_demo.py --frames 120         # output/hybrid_pose3d.mp4 (960x720)
uv run python examples/hybrid_3cam_demo.py --frames 120 --jitter 0.01
```

---

## C. 3D 복원 — 실제 CMU Panoptic 데이터

### C-1. 실제 calibration + 3D GT (이미지 불필요) (`examples/panoptic_demo.py`)

CMU Panoptic의 **실제 31-HD-카메라 calibration**과 데이터셋 **실제 3D GT**로 파이프라인을 검증.
(2D는 실제 calibration으로 GT를 투영 + 검출 노이즈 — 실영상 RTMPose는 C-2 참고.)

```bash
# 데이터(작음: calibration + GT 프레임)
BASE=https://raw.githubusercontent.com/open-mmlab/mmpose/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/tests/data/panoptic_body3d/160906_band1
mkdir -p data/panoptic/160906_band1
curl -sSL -o data/panoptic/160906_band1/calibration_160906_band1.json "$BASE/calibration_160906_band1.json"
curl -sSL -o data/panoptic/160906_band1/body3DScene_00000168.json "$BASE/hdPose3d_stage1_coco19/body3DScene_00000168.json"

# 실행
uv run python examples/panoptic_demo.py --num-views 4 --noise-px 3.0
#   -> output/panoptic_pose3d.png (복원 3D vs GT)
#   옵션: --num-views 2|3|4  --noise-px <px>  --body 0|1|2  --seed N
```

결과: 노이즈 0 → **0.0 mm / 0.00 px**(calibration 파싱·삼각측량 정확), 4뷰+3px ≈ **6.3 mm**,
2뷰+5px ≈ 12.8 mm.

### C-2. 실제 HD 영상에 RTMPose (`examples/panoptic_video_demo.py`)

실제 멀티뷰 HD 영상에 RTMPose를 돌려 삼각측량으로 3D를 복원하는 **진짜 실영상 데모**.
**단일 인물 시퀀스**를 쓰세요(다인 장면은 인물 매칭 추가 필요). HD 카메라당 ~2.8GB(3대 ≈ 8.6GB).

**Windows PowerShell**
```powershell
$SEQ = "171204_pose1"
$D = "http://domedb.perception.cs.cmu.edu/webdata/dataset/$SEQ"
New-Item -ItemType Directory -Force "data\panoptic\$SEQ\hdVideos" | Out-Null
curl.exe -L -o "data\panoptic\$SEQ\calibration_${SEQ}.json" "$D/calibration_${SEQ}.json"
foreach ($n in '03','12','23') {
  curl.exe -C - -L -o "data\panoptic\$SEQ\hdVideos\hd_00_${n}.mp4" "$D/videos/hd_shared_crf20/hd_00_${n}.mp4"
}
# 다운로드 확인 (각 ~2700MB면 정상)
Get-ChildItem "data\panoptic\$SEQ\hdVideos" | Select-Object Name, @{n="MB";e={[int]($_.Length/1MB)}}
# 실행
uv run python examples/panoptic_video_demo.py --seq-dir data/panoptic/171204_pose1 --cams 00_03,00_12,00_23 --start 500 --num-frames 60 --device cuda
```
> PowerShell의 `curl`은 `Invoke-WebRequest` 별칭일 수 있어 **`curl.exe`**로 명시했고,
> `$n.mp4` 오해 방지를 위해 `${n}`/`${SEQ}` 중괄호를 씁니다.

**Linux / macOS / Git Bash**
```bash
SEQ=171204_pose1
D=http://domedb.perception.cs.cmu.edu/webdata/dataset/$SEQ
mkdir -p data/panoptic/$SEQ/hdVideos
curl -L -o "data/panoptic/$SEQ/calibration_$SEQ.json" "$D/calibration_$SEQ.json"
for n in 03 12 23; do
  curl -C - -L -o "data/panoptic/$SEQ/hdVideos/hd_00_$n.mp4" "$D/videos/hd_shared_crf20/hd_00_$n.mp4"
done
uv run python examples/panoptic_video_demo.py \
    --seq-dir data/panoptic/171204_pose1 --cams 00_03,00_12,00_23 --start 500 --num-frames 60 --device cuda
```

- 산출물: `output/panoptic_video_pose3d.mp4` (HD 뷰들 2D + 3D 복원). 프레임은 mp4에서 직접 읽음.
- 대안 다운로드: 공식 툴박스 `CMU-Perceptual-Computing-Lab/panoptic-toolbox`
  → `./scripts/getData.sh 171204_pose1 0 3`, 느리면 `--snu-endpoint`(SNU 미러).
- 이 데모는 **HD RGB 삼각측량만**(depth fusion off). depth(Kinect)까지 쓰려면
  `getData_kinoptic.sh`의 `kcalibration_*.json` + `KINECTNODE*/depthdata.dat`를 디코딩해 정렬된
  depth맵(미터)을 `pipeline.process(depth_map=...)`로 넘기면 됩니다.

---

## D. 실제 3D 라이브 추론 (자체 calibration 완료 후)

`config/cameras.yaml`에 3대 카메라의 calibration(K/dist/R/t)을 채운 뒤 `--synthetic` 없이 실행:

```bash
uv run python run.py --config config/cameras.yaml
```

---

## 테스트

```bash
uv run pytest tests/ -q
```

모든 수치 모듈(삼각측량/calibration/fusion/스무딩/파이프라인)은 합성 데이터 단위·통합 테스트를
동반하며 rtmlib 없이 오프라인으로 통과합니다.

---

## Calibration 절차

calibration 정확도가 전체 시스템의 병목이므로 항상 reprojection error 리포트를 확인하세요.
체커보드를 새로 촬영한 뒤 `src/calibration/calibrate.py`를 사용합니다.

1. **Intrinsic** — 카메라별 체커보드 이미지 →
   `find_checkerboard_corners(images, pattern_size=(cols,rows), square_size_m)`
   → `calibrate_intrinsics(...)` → `(K, dist, rms)`.
2. **Extrinsic** — 3대가 동시에 보는 보드로 카메라별 `estimate_board_pose(...)` 후
   `calibrate_extrinsics(board_poses, reference="cam0", world_frame="reference_camera")`
   → world→camera `(R, t)`.
3. **저장** — `build_camera_params(...)` → `save_cameras_yaml(cameras, "config/cameras.yaml")`.
4. **검증** — `reprojection_report(cameras, observations)` RMS 확인(목표 ≲1 px).

---

## 설정 (`config/cameras.yaml`)

| 섹션 | 주요 필드 |
|---|---|
| `units` | `length: meter` (전 모듈 공통) |
| `world` | `frame` (`reference_camera` \| `board_origin`), `reference_camera` |
| `cameras[]` | `name, type (rgb\|rgbd), K, dist, R, t, image_size, source`; rgbd는 `depth_K, depth_scale, depth_to_color_R/t` 추가 |
| `detection` | `backend (cuda\|cpu), model, mode, det_score_threshold` |
| `triangulation` | `min_views, score_threshold, ransac.{enabled,reproj_threshold_px}` |
| `depth_fusion` | `enabled, depth_min, depth_max, fill_missing, patch_radius_px, depth_weight` |
| `smoothing` | `enabled, freq, min_cutoff, beta, d_cutoff` |
| `input` | `mode (live\|file), sync (hardware\|software), sync_tolerance_ms` |
| `output` | `format (json\|npy), path` |
| `seed` | 재현성을 위한 난수 시드 (`null`이면 비활성) |

기본 `K/dist/R/t`는 **placeholder**지만 동시에 유효한 합성 리그라서 `--synthetic` 데모가 바로
동작합니다. 실제 사용 시 calibration 결과로 교체하세요.

---

## 설계 결정 (고정)

- **단위:** 전 모듈 meter, 픽셀은 `(u, v)` 순서.
- **World 좌표계:** 기준 카메라 `cam0`(따라서 `cam0`은 `R = I`, `t = 0`). `world.frame`으로 보드 원점 전환 가능.
- **Extrinsic:** world → camera `X_cam = R·X_world + t`, `P = K[R|t]`.
- **키포인트:** COCO-17, 모든 뷰 동일 인덱스 순서(다중 뷰 대응의 전제).
- **Confidence:** 2D `score`를 삼각측량 가중치와 depth fusion 가중치 양쪽에 사용 →
  가려진/저신뢰 관절 자동 down-weight/제외.
- **Depth SDK:** 미정 → 추상 `DepthFrameSource`(`src/io/depth_reader.py`) + `Dummy`/`File` 백엔드.
  RealSense/Kinect/Orbbec는 수학 로직 변경 없이 나중에 연결.
- **왜곡:** 삼각측량 전 픽셀 undistort, aligned depth는 rectified color grid 가정(합성에서 정확).

---

## 프로젝트 구조

```
config/cameras.yaml              calibration + 파이프라인 설정
src/
  core/{types,geometry}.py       공용 dataclass, COCO-17, 기하 프리미티브
  io/{frame_reader,depth_reader}.py  다중 뷰 동기 리더 + depth 소스 추상화
  pose2d/rtmpose_detector.py     rtmlib RTMPose 래퍼 (TORCH_HOME=./models)
  calibration/calibrate.py       intrinsic/extrinsic, reprojection 리포트, yaml I/O
  triangulation/{dlt,robust}.py  confidence 가중 DLT + robust 뷰 선택
  fusion/depth_fusion.py         depth back-projection + 융합
  smoothing/one_euro.py          One-Euro 시간적 필터
  viz/visualize_3d.py            3D 플롯 + JSON/NPY 저장
  synthetic.py                   오프라인 합성 데이터 생성기
  pipeline.py                    엔드투엔드 오케스트레이션
examples/
  realtime_demo.py               A. 실시간 2D 추출 (실제 RTMPose, MP4)
  pose3d_video.py                B-2. 합성 3D 스켈레톤 영상
  hybrid_3cam_demo.py            B-3. 2RGB+1Depth 4분할 결과영상
  panoptic_demo.py               C-1. 실제 Panoptic calib+GT 3D (이미지 불필요)
  panoptic_video_demo.py         C-2. 실제 Panoptic HD 영상 RTMPose → 3D
run.py                           엔트리포인트 (B-1 합성 / D 실제 라이브)
tests/                           합성 데이터 단위/통합 테스트
data/, models/, output/          (gitignore) 데이터셋 / 모델 캐시 / 산출물
```
