# 3D 스켈레톤 포즈 추정 (RGB 2대 + RGB-D 1대)

RGB 카메라 2대와 RGB-D 카메라 1대로 한 사람의 **3D 스켈레톤 포즈**(COCO-17)를 추정합니다.
단안(monocular)의 depth 모호성은 다중 뷰 **confidence 가중 삼각측량**으로 해소하고, depth 센서로
scale 보정·가려진 관절 보완을 하는 하이브리드 방식입니다.

기술 스택(고정): [rtmlib](https://github.com/Tau-J/rtmlib) (RTMPose ONNX, `mmcv`/`mmpose`/`mmdet`
미사용) · onnxruntime · OpenCV · NumPy · SciPy · Python 3.11. 학습 없이 추론만 합니다.

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

> 이 README는 **(1) 환경 구축**, **(2) 실시간 2D pose 추출 데모**, **(3) 실제 영상에서 depth까지
> 활용한 3D pose → MP4 데모** 3가지 실행법을 다룹니다. 모든 명령은 **프로젝트 루트**에서 실행하고
> 산출물은 `output/`에 저장됩니다.

---

## 1. 환경 구축

[uv](https://github.com/astral-sh/uv)만 있으면 됩니다. 클론 후 `uv.lock`으로 **모든 머신에서 동일한
버전**이 설치됩니다(Python은 `.python-version`의 3.11 자동 사용).

```bash
git clone https://github.com/junmin-83/3d-skeleton-pose-estimation
cd 3d-skeleton-pose-estimation
uv sync          # pyproject.toml + uv.lock 의 고정 버전 그대로 설치
```

- `uv sync`는 lock 고정 버전(numpy/scipy/opencv/onnxruntime/rtmlib + dev: pytest/ruff)을 설치 →
  **재현성 보장**. 이후 `uv run ...` 은 실행 시 환경을 자동 동기화합니다.
- **GPU(NVIDIA CUDA):** 기본은 CPU용 `onnxruntime`이라 추가 설치 없이 모든 데모가 CPU로 동작합니다.
  NVIDIA GPU 가속(검출 약 8배)은 1회 설치가 필요하고 lock에 없으므로 **`uv run --no-sync`** 로 실행해야
  합니다 → 아래 **(참고) GPU 가속 (NVIDIA CUDA)** 섹션 참고.
- **모델 캐시:** rtmlib RTMPose ONNX 모델은 `TORCH_HOME=./models`로 지정돼 **`./models/hub/checkpoints/`**
  에 받습니다(`src/pose2d/rtmpose_detector.py`에 하드코딩, env로 override). `models/`는 `.gitignore`.
- **테스트:** `uv run pytest tests/ -q` (수치 모듈 단위·통합 테스트, rtmlib 없이 오프라인 통과).

---

## 데모 실행 — 공통 안내

`examples/` 폴더의 데모 3개는 모두 **프로젝트 루트에서** `uv run python examples/<파일>.py ...` 로 실행합니다.

- `uv run` 이 환경을 자동 동기화하므로 `uv sync` 를 깜빡해도 됩니다.
- **첫 실행 시** RTMPose ONNX 모델이 `./models/hub/checkpoints/` 에 자동 다운로드됩니다(수십 초, 1회).
- 결과물은 모두 `output/` 에 저장됩니다(`.gitignore` 대상 — 지워도 데모 재실행 시 다시 생성).
- 공통 옵션: `--device cpu`(기본) 또는 `--device cuda`(GPU), `--num-frames N`(길이·속도 조절).

| # | 데모 | 스크립트 | 입력 준비물 | 한 줄 실행 예 | 산출물 |
|---|---|---|---|---|---|
| 2 | 실시간 2D 키포인트 | `examples/realtime_demo.py` | 사람 사진 1장(소) 또는 웹캠 | `uv run python examples/realtime_demo.py --frames 30` | `output/realtime_keypoints.mp4` |
| 3-A | RGB-D depth 3D | `examples/rgbd_video_demo.py` | RGB-D 영상(TUM ~422MB) 또는 RealSense | `uv run python examples/rgbd_video_demo.py --tum <dir> --num-frames 60` | `output/rgbd_pose3d.mp4` |
| 3-B | 멀티뷰 HD 3D | `examples/panoptic_video_demo.py` | Panoptic HD 3대(~8.6GB) | `uv run python examples/panoptic_video_demo.py --seq-dir <dir> --cams 00_03,00_12,00_23` | `output/panoptic_video_pose3d.mp4` |

> **가장 빠른 시작은 2번**입니다(사람 사진 1장만 받으면 끝). 3-A·3-B는 데이터 다운로드가 필요하며, 각 섹션의 **① 데이터 준비 → ② 실행** 순서를 따르세요.

---

## 2. 실시간 2D pose 추출 데모 (RGB → COCO-17 17개)

`examples/realtime_demo.py` — 단일 RGB(이미지/웹캠)에서 **실제 RTMPose**로 17개 키포인트를 프레임마다
추출하고, FPS·좌표를 출력하며 스켈레톤 오버레이 **MP4** + 마지막 프레임 PNG를 저장합니다.

**① 데이터 준비** — 사람이 있는 사진 1장(웹캠을 쓰면 이 단계 생략):
```bash
curl -sSL -o data/demo/person.jpg "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg"
```

**② 실행**
```bash
# (a) 정적 이미지를 스트림처럼 반복 — 가장 간단
uv run python examples/realtime_demo.py --frames 30
#   -> output/realtime_keypoints.mp4 + output/realtime_keypoints.png

# (b) 실제 웹캠 라이브 (장치 0번, 200프레임)
uv run python examples/realtime_demo.py --camera 0 --frames 200

# (c) GPU 고속 (GPU 1회 설치 후 — (참고) GPU 가속 섹션, --no-sync 필수)
uv run --no-sync python examples/realtime_demo.py --camera 0 --device cuda --mode balanced
```

옵션: `--image <경로>` · `--mode lightweight|balanced|performance` · `--score-thr <0~1>`(올리면
저신뢰/오검출 관절 제외) · `--video/--out/--fps`. 성능: CPU(lightweight) 약 11~12 FPS, GPU 30+ FPS.

---

## 3. 실제 영상에서 3D pose 추출 → MP4

3D 복원에는 **RGB-D의 depth(back-projection)** 또는 **멀티뷰 calibration(삼각측량)** 이 필요합니다.
실영상 데모 두 가지를 제공합니다.

### 3-A. RGB-D 영상 — Depth 정보로 3D (`examples/rgbd_video_demo.py`)

단일 RGB-D(컬러 + 정렬 depth)에서 RTMPose 2D를 검출하고, **각 관절의 depth를 back-projection**해
3D를 복원합니다(삼각측량 없이 depth만으로). 출력: `[RGB+2D | depth 컬러맵 | 3D 스켈레톤]` MP4.

**① 데이터 준비** — 공개 RGB-D 영상(TUM sitting_static, ~422MB) 다운로드 + 압축 해제.

Windows PowerShell:
```powershell
New-Item -ItemType Directory -Force data\tum | Out-Null
curl.exe -L -o data\tum\sitting_static.tgz "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz"
tar -xzf data\tum\sitting_static.tgz -C data\tum
```

Linux / macOS / Git Bash:
```bash
mkdir -p data/tum
curl -L -o data/tum/sitting_static.tgz "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz"
tar -xzf data/tum/sitting_static.tgz -C data/tum
```

**② 실행** (OS 공통):
```bash
# CPU — 추가 설치 없이 바로 동작 (짧게 보려면 --num-frames 20)
uv run python examples/rgbd_video_demo.py --tum data/tum/rgbd_dataset_freiburg3_sitting_static --start 30 --num-frames 60
# GPU — 1회 설치 후 ((참고) GPU 가속 섹션), --no-sync + --device cuda
uv run --no-sync python examples/rgbd_video_demo.py --tum data/tum/rgbd_dataset_freiburg3_sitting_static --start 30 --num-frames 60 --device cuda
```

- 산출물: `output/rgbd_pose3d.mp4`. 콘솔에 10프레임마다 `N/17 joints from depth` 출력.
- **단일 인물 시퀀스 권장**: `detect_best`가 최고신뢰 1명만 추적하므로 다인 장면에선 프레임마다 대상이 바뀌어 튈 수 있음.
- **Intel RealSense 라이브**(pyrealsense2 필요): `uv run python examples/rgbd_video_demo.py --realsense --num-frames 300`
  (컬러에 정렬된 depth + intrinsics를 자동 사용; GPU는 `uv run --no-sync ... --device cuda`).
- 옵션: `--depth-min/--depth-max`(유효 depth 범위, m) · `--mode` · `--fps`.
- TUM intrinsics(freiburg3: fx=535.4, fy=539.2, cx=320.1, cy=247.6, depth/5000=m)는 코드에 내장.

### 3-B. 멀티뷰 HD 영상 — 삼각측량 3D (`examples/panoptic_video_demo.py`, CMU Panoptic)

실제 멀티뷰 HD 영상에 RTMPose를 돌려 **삼각측량으로 3D**를 복원합니다. 출력: `[HD 뷰들 2D | 3D]` MP4.
**단일 인물 시퀀스**를 쓰세요(다인 장면은 뷰 간 인물 매칭 추가 필요 — 미구현). HD 카메라당 ~2.8GB.

> 이 경로는 멀티뷰 RGB **삼각측량**이며 depth는 쓰지 않습니다. Depth를 활용한 3D는 3-A를 보세요.
> (Panoptic의 Kinect depth는 `.dat` 원시 포맷 디코딩·동기·정렬이 별도로 필요합니다.)

**① 데이터 준비** — Panoptic 단일 인물 시퀀스의 HD 카메라 3대 + calibration(카메라당 ~2.8GB).

Windows PowerShell:
```powershell
$SEQ = "171204_pose1"
$D = "http://domedb.perception.cs.cmu.edu/webdata/dataset/$SEQ"
New-Item -ItemType Directory -Force "data\panoptic\$SEQ\hdVideos" | Out-Null
curl.exe -L -o "data\panoptic\$SEQ\calibration_${SEQ}.json" "$D/calibration_${SEQ}.json"
foreach ($n in '03','12','23') {
  curl.exe -C - -L -o "data\panoptic\$SEQ\hdVideos\hd_00_${n}.mp4" "$D/videos/hd_shared_crf20/hd_00_${n}.mp4"
}
```

Linux / macOS / Git Bash:
```bash
SEQ=171204_pose1
D=http://domedb.perception.cs.cmu.edu/webdata/dataset/$SEQ
mkdir -p data/panoptic/$SEQ/hdVideos
curl -L -o "data/panoptic/$SEQ/calibration_$SEQ.json" "$D/calibration_$SEQ.json"
for n in 03 12 23; do
  curl -C - -L -o "data/panoptic/$SEQ/hdVideos/hd_00_$n.mp4" "$D/videos/hd_shared_crf20/hd_00_$n.mp4"
done
```

**② 실행** (OS 공통):
```bash
# CPU — 바로 동작
uv run python examples/panoptic_video_demo.py --seq-dir data/panoptic/171204_pose1 --cams 00_03,00_12,00_23 --start 500 --num-frames 60
# GPU — 1회 설치 후 ((참고) GPU 가속 섹션), --no-sync + --device cuda
uv run --no-sync python examples/panoptic_video_demo.py --seq-dir data/panoptic/171204_pose1 --cams 00_03,00_12,00_23 --start 500 --num-frames 60 --device cuda
```

산출물: `output/panoptic_video_pose3d.mp4`. 대안 다운로드: 툴박스
`CMU-Perceptual-Computing-Lab/panoptic-toolbox` → `./scripts/getData.sh 171204_pose1 0 3`
(느리면 `--snu-endpoint`).

---

## (참고) GPU 가속 (NVIDIA CUDA)

기본 환경은 CPU `onnxruntime`이라 모든 데모가 추가 설치 없이 동작합니다. NVIDIA GPU(예: RTX 4050)에서
RTMPose 검출을 가속하려면(측정 약 **8배** — CPU ~9.6 FPS → GPU ~78 FPS) 아래 1회 설치 후 반드시
**`uv run --no-sync`** 로 실행하세요.

**왜 `--no-sync`?** GPU 패키지(onnxruntime-gpu + nvidia CUDA 휠)는 크로스플랫폼 재현용 `uv.lock`의
기본(default) 설치에 포함되지 않습니다(CPU 기준). 그래서 그냥 `uv run` 하면 uv가 환경을 lock으로
**재동기화하며 CPU `onnxruntime`으로 되돌립니다**. GPU 실행은 반드시 `uv run --no-sync` 를 쓰세요.

**1회 설치** (Windows x86_64 · CUDA 12 · cuDNN 9.11):
```bash
# rtmlib이 끌어온 CPU onnxruntime을 먼저 제거 (둘 다 같은 import 경로라 공존 불가)
uv pip uninstall onnxruntime
# GPU 런타임 + CUDA 12 휠 설치 (cuDNN은 9.12+ 회피 — 아래 주의)
uv pip install onnxruntime-gpu "nvidia-cudnn-cu12<9.12" nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12
```
> 동일 버전이 `pyproject.toml`의 `gpu` 의존성 그룹과 `uv.lock`(Windows x86_64 마커)에 핀으로 기록돼
> 있습니다. `uv sync --group gpu` 로 한 번에 설치할 수도 있으나, 그 경우에도 위처럼 CPU `onnxruntime`
> 제거가 필요합니다.

**실행** — 데모에 `--no-sync` + `--device cuda`:
```bash
uv run --no-sync python examples/rgbd_video_demo.py --tum data/tum/rgbd_dataset_freiburg3_sitting_static --start 30 --num-frames 60 --device cuda
uv run --no-sync python examples/panoptic_video_demo.py --seq-dir data/panoptic/171204_pose1 --cams 00_03,00_12,00_23 --start 500 --num-frames 60 --device cuda
```

**주의 · 동작 원리**
- **cuDNN 9.12+ 비호환:** cuDNN 9.23은 onnxruntime 1.26의 CUDA provider에서 `CUDNN_BACKEND_API_FAILED`로
  실패해 조용히 CPU로 폴백합니다. 반드시 **`nvidia-cudnn-cu12<9.12`**(검증판 9.11.1.4)을 쓰세요.
- **DLL 자동 로드:** `src/pose2d/rtmpose_detector.py`가 `device="cuda"`일 때 `onnxruntime.preload_dlls()`로
  nvidia 휠의 CUDA/cuDNN DLL을 로드합니다(CPU에선 무해한 no-op, 실패 시 CPU 폴백).
- **확인:** 데모 로그에서 검출 FPS가 크게 오르면 GPU 사용 중. `cublasLt64_12.dll ... missing` 류 에러가
  나면 위 nvidia 휠 설치가 빠진 것이니 다시 설치하세요.

---

## (참고) Calibration 절차

calibration 정확도가 전체 정확도의 병목이므로 reprojection error 리포트를 확인하세요.
체커보드를 새로 촬영한 뒤 `src/calibration/calibrate.py`를 사용합니다.

1. **Intrinsic** — `find_checkerboard_corners(images, pattern_size=(cols,rows), square_size_m)`
   → `calibrate_intrinsics(...)` → `(K, dist, rms)`.
2. **Extrinsic** — 3대가 동시에 보는 보드로 `estimate_board_pose(...)` 후
   `calibrate_extrinsics(board_poses, reference="cam0", world_frame="reference_camera")`.
3. **저장** — `build_camera_params(...)` → `save_cameras_yaml(cameras, "config/cameras.yaml")`.
4. **검증** — `reprojection_report(cameras, observations)` RMS 확인(목표 ≲1 px).

calibration을 채운 뒤 실제 라이브 추론: `uv run python run.py --config config/cameras.yaml`.

---

## (참고) 설정 (`config/cameras.yaml`)

| 섹션 | 주요 필드 |
|---|---|
| `units` / `world` | `length: meter` · `frame`(`reference_camera`\|`board_origin`), `reference_camera` |
| `cameras[]` | `name, type(rgb\|rgbd), K, dist, R, t, image_size, source`; rgbd는 `depth_K, depth_scale, depth_to_color_R/t` 추가 |
| `detection` | `backend(cuda\|cpu), model, mode, det_score_threshold` |
| `triangulation` | `min_views, score_threshold, ransac.{enabled,reproj_threshold_px}` |
| `depth_fusion` | `enabled, depth_min, depth_max, fill_missing, patch_radius_px, depth_weight` |
| `smoothing` | `enabled, freq, min_cutoff, beta, d_cutoff` |
| `input` / `output` | `mode, sync, sync_tolerance_ms` · `format(json\|npy), path` |

---

## (참고) 설계 결정

- **단위:** 전 모듈 meter, 픽셀 `(u, v)`.
- **World 좌표계:** 기준 카메라 `cam0`(R=I, t=0); `world.frame`으로 보드 원점 전환 가능.
- **Extrinsic:** `X_cam = R·X_world + t`, `P = K[R|t]`.
- **키포인트:** COCO-17, 모든 뷰 동일 인덱스 순서(다중 뷰 대응 전제).
- **Confidence:** 2D `score`를 삼각측량·depth fusion 가중치 양쪽에 사용 → 가려진/저신뢰 관절 자동 제외.
- **Depth SDK:** 추상 `DepthFrameSource`(`src/io/depth_reader.py`) + File/Dummy 백엔드.
  RealSense는 `examples/rgbd_video_demo.py`의 `--realsense`에서 직접 연결.
- **왜곡:** 삼각측량 전 픽셀 undistort, aligned depth는 rectified color grid 가정.

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
  pipeline.py                    엔드투엔드 오케스트레이션
examples/
  realtime_demo.py               (2) 실시간 2D 17키포인트 추출 (실제 RTMPose, MP4)
  rgbd_video_demo.py             (3-A) RGB-D depth 활용 3D → MP4
  panoptic_video_demo.py         (3-B) 멀티뷰 HD 삼각측량 3D → MP4
run.py                           실제 라이브/녹화 멀티뷰 엔트리포인트
tests/                           수치 모듈 단위/통합 테스트
data/, models/, output/          (gitignore) 데이터셋 / 모델 캐시 / 산출물
```
