# 3D 스켈레톤 포즈 추정 (RGB 2대 + RGB-D 1대 가정)

RGB 카메라 2대와 RGB-D 카메라 1대로 한 사람의 **3D 스켈레톤 포즈**(COCO-17)를 추정합니다.
단안(monocular)의 depth 모호성은 다중 뷰 **confidence 가중 삼각측량**으로 해소하고, depth 센서로
scale 보정·가려진 관절 보완을 하는 하이브리드 방식입니다.

기술 스택(고정): [rtmlib](https://github.com/Tau-J/rtmlib) (RTMPose ONNX, `mmcv`/`mmpose`/`mmdet`
미사용) · onnxruntime · OpenCV · NumPy · SciPy · Python 3.11. 학습 없이 추론만 합니다.

```
[동기화된 3개 프레임]
   │ (1) 뷰별 RTMPose 2D 키포인트              src/pose2d/rtmpose_detector.py
   │ (2) calibration -> projection matrix       src/calibration/{intrinsics,extrinsics,camera_io}.py
   │ (3) 삼각측량 (confidence DLT + robust 제외) src/triangulation/{dlt,robust}.py
   │ (4) depth fusion (back-projection + 융합)   src/fusion/depth_fusion.py
   │ (5) 시간적 스무딩 (One-Euro)               src/smoothing/one_euro.py
   ▼
[3D 스켈레톤 + 시각화/저장]                      src/render/skeleton_3d.py · src/io/keypoints_io.py
```

> 이 README는 **(1) 환경 구축**, **(2) 실시간 2D pose 추출 데모**, **(3) 실제 영상에서 depth까지
> 활용한 3D pose → MP4 데모**, 그리고 **(3-C) RGB+Depth 융합(hybrid) 합성 데모** 실행법을 다룹니다.
> 모든 명령은 **프로젝트 루트**에서 실행하고 산출물은 `output/`에 저장됩니다.

---

## 빠른 실행 (복사용 치트시트)

환경 구축(§1, `uv sync`)과 데이터 준비(3-A·3-B는 각 섹션의 **① 데이터 준비**)를 마쳤다면, 아래를
**프로젝트 루트에서** 그대로 복사해 실행하세요. 산출물은 모두 `output/`에 저장됩니다.

```bash
# 2) 실시간 2D 17키포인트 — 사람 사진 1장이면 끝 (가장 빠른 시작)
uv run python examples/realtime_demo.py --frames 30
#    -> output/realtime_keypoints.mp4  (+ 마지막 프레임 .png)
#    웹캠 라이브: uv run python examples/realtime_demo.py --camera 0 --frames 60

# 3-A) RGB-D depth로 3D pose -> MP4  (TUM 영상; §3-A ①에서 다운로드)
uv run python examples/rgbd_video_demo.py --tum data/tum/rgbd_dataset_freiburg3_sitting_static --start 30 --num-frames 600
#    -> output/rgbd_pose3d.mp4   ([RGB+2D | Depth | 3D] 3분할)

# 3-B) 멀티뷰 HD 삼각측량으로 3D pose -> MP4  (CMU Panoptic; §3-B ①에서 다운로드)
uv run python examples/panoptic_video_demo.py --seq-dir data/panoptic/171204_pose1 --cams 00_03,00_12,00_23 --start 500 --num-frames 60
#    -> output/panoptic_video_pose3d.mp4   ([HD 뷰 3개 2D | 3D])

# 3-C) (합성) RGB+Depth 융합 = hybrid fused 경로  (다운로드·rtmlib·GPU 불필요)
uv run python examples/fusion_demo.py --num-frames 80
#    -> output/fusion_pose3d.mp4   ([cam0 2D | cam1 2D | Depth | 3D 출처색: fused=초록/depth=파랑/tri=빨강])
```

> - **디바이스:** 기본 `cuda`. GPU 셋업(아래 **(참고) GPU 가속**)을 했고 `UV_NO_SYNC=1`이면 위 명령이
>   그대로 GPU로 실행됩니다. GPU가 없거나 셋업 전이면 자동으로 CPU로 폴백합니다(추가 설정 불필요).
>   CPU로 강제하려면 명령 끝에 `--device cpu`.
> - **길이·구간 조절:** `--num-frames N`으로 길이를, 3-A/3-B의 `--start`로 시작 프레임을 바꿉니다.
> - 데이터 다운로드, 옵션 상세는 아래 §2 / §3-A / §3-B를 참고하세요.

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
- **GPU(NVIDIA CUDA)가 기본:** 모든 데모의 기본 디바이스는 `cuda`입니다. GPU 셋업이 안 된 환경(또는
  GPU 없는 머신)에서는 자동으로 CPU로 폴백하므로 추가 설정 없이도 동작합니다. 실제 GPU 가속(검출 약
  8배)은 **1회 셋업**이 필요합니다(`./scripts/setup-gpu.ps1` + `UV_NO_SYNC=1`) → 아래
  **(참고) GPU 가속 (NVIDIA CUDA)**. CPU로 강제하려면 데모에 `--device cpu`.
- **모델 캐시:** rtmlib RTMPose ONNX 모델은 `TORCH_HOME=./models`로 지정돼 **`./models/hub/checkpoints/`**
  에 받습니다(`src/pose2d/rtmpose_detector.py`에 하드코딩, env로 override). `models/`는 `.gitignore`.
- **테스트:** `uv run pytest tests/ -q` (수치 모듈 단위·통합 테스트, rtmlib 없이 오프라인 통과).

---

## 데모 실행 — 공통 안내

`examples/` 폴더의 데모 3개는 모두 **프로젝트 루트에서** `uv run python examples/<파일>.py ...` 로 실행합니다.

- `uv run` 은 기본적으로 실행 시 환경을 자동 동기화합니다(`uv sync` 를 깜빡해도 됨). 단, GPU 셋업을 한
  머신은 `UV_NO_SYNC=1` 로 이 자동 동기화를 꺼서 GPU 환경이 유지됩니다 → (참고) GPU 가속.
- **첫 실행 시** RTMPose ONNX 모델이 `./models/hub/checkpoints/` 에 자동 다운로드됩니다(수십 초, 1회).
- 결과물은 모두 `output/` 에 저장됩니다(`.gitignore` 대상 — 지워도 데모 재실행 시 다시 생성).
- 공통 옵션: `--device cuda`(**기본**; GPU 미감지 시 자동 CPU 폴백) / `--device cpu`(CPU 강제) ·
  `--num-frames N`(길이·속도 조절). 실제 GPU 가속은 1회 셋업 필요 → (참고) GPU 가속.

| # | 데모 | 스크립트 | 입력 준비물 | 한 줄 실행 예 | 산출물 |
|---|---|---|---|---|---|
| 2 | 실시간 2D 키포인트 | `examples/realtime_demo.py` | 사람 사진 1장(소) 또는 웹캠 | `uv run python examples/realtime_demo.py --frames 30` | `output/realtime_keypoints.mp4` |
| 3-A | RGB-D depth 3D | `examples/rgbd_video_demo.py` | RGB-D 영상(TUM ~422MB) 또는 RealSense | `uv run python examples/rgbd_video_demo.py --tum <dir> --num-frames 600` | `output/rgbd_pose3d.mp4` |
| 3-B | 멀티뷰 HD 3D | `examples/panoptic_video_demo.py` | Panoptic HD 3대(~8.6GB) | `uv run python examples/panoptic_video_demo.py --seq-dir <dir> --cams 00_03,00_12,00_23` | `output/panoptic_video_pose3d.mp4` |
| 3-C | (합성) RGB+Depth 융합 | `examples/fusion_demo.py` | 없음(합성) | `uv run python examples/fusion_demo.py --num-frames 80` | `output/fusion_pose3d.mp4` |

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

# (c) CPU로 강제 (기본은 GPU; CPU 비교용)
uv run python examples/realtime_demo.py --camera 0 --device cpu --mode balanced
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
# 기본 = GPU (셋업 시) · GPU 없으면 자동 CPU 폴백 · 짧게 보려면 --num-frames 20
uv run python examples/rgbd_video_demo.py --tum data/tum/rgbd_dataset_freiburg3_sitting_static --start 30 --num-frames 60
# CPU로 강제: 끝에 --device cpu 추가
```

- 산출물: `output/rgbd_pose3d.mp4`. 콘솔에 10프레임마다 `N/17 joints from depth` 출력.
- **단일 인물 시퀀스 권장**: `detect_best`가 최고신뢰 1명만 추적하므로 다인 장면에선 프레임마다 대상이 바뀌어 튈 수 있음.
- **Intel RealSense 라이브**(pyrealsense2 필요): `uv run python examples/rgbd_video_demo.py --realsense --num-frames 300`
  (컬러에 정렬된 depth + intrinsics 자동 사용; 기본 GPU, CPU는 `--device cpu`).
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
# 기본 = GPU (셋업 시) · GPU 없으면 자동 CPU 폴백 · CPU 강제는 --device cpu
uv run python examples/panoptic_video_demo.py --seq-dir data/panoptic/171204_pose1 --cams 00_03,00_12,00_23 --start 500 --num-frames 60
```

산출물: `output/panoptic_video_pose3d.mp4`. 대안 다운로드: 툴박스
`CMU-Perceptual-Computing-Lab/panoptic-toolbox` → `./scripts/getData.sh 171204_pose1 0 3`
(느리면 `--snu-endpoint`).

### 3-C. (합성) RGB+Depth 융합 — hybrid `fused` 경로 (`examples/fusion_demo.py`)

3-A(depth-only)·3-B(triangulation-only)는 3D 복원의 한쪽 경로만 시연합니다. 이 프로젝트의
간판인 **다중 뷰 삼각측량 + depth back-projection을 관절별로 융합**하는 hybrid 경로
(`Pipeline`의 `fuse`, `source='fused'`)는 "캘리브레이션된 다중 RGB 뷰 + 정렬 depth"를 동시에
주는 공개 단일 데이터셋이 없어 실영상 데모에는 빠져 있습니다(TUM=단일 RGB-D, Panoptic=Kinect
depth 미디코딩). 이 데모는 그 셋업을 **합성**으로 구성해 fused 경로를 눈으로 보여줍니다.
출력: `[cam0 RGB+2D | cam1 RGB+2D | RGB-D depth+2D | 3D(출처색)]` 4분할 MP4.

> 이 경로는 **합성 기하 데모**입니다(알려진 3D를 투영해 입력을 만들므로 RTMPose 미사용).
> 실픽셀 + 실depth 융합은 멀티뷰 RGB와 정렬 depth를 한 좌표계로 묶어야 하며(예: Panoptic
> Kinect depth 디코딩·동기·정렬), 그건 별도 작업입니다 — 미구현.

**① 데이터 준비** — **없음**. 합성이라 데이터셋 다운로드·모델 캐시·GPU가 모두 불필요하고
오프라인에서 결정론적으로 실행됩니다(rtmlib도 import 안 함). 바로 ②로 가세요.

**② 실행** (OS 공통):
```bash
# 다운로드·rtmlib·GPU 불필요 · 짧게 보려면 --num-frames 20 · CPU/GPU 무관
uv run python examples/fusion_demo.py --num-frames 80
```

- 산출물: `output/fusion_pose3d.mp4`. 콘솔에 10프레임마다 `fused=… depth=… tri=… missing=…`
  (관절 출처별 개수) 출력. 기본 시나리오는 매 프레임 `fused=15 depth=1 tri=1 missing=0`.
- **합성 셋업:** RGB 2대(`cam0` 원점, `cam1` 기준선 0.6 m) + RGB-D 1대(`cam2`, 0.3 m)를 한
  world 좌표계에 배치(공통 intrinsics fx=fy=600, 640×480). 알려진 3D COCO-17(오른팔 흔들기 +
  좌우 sway)을 각 뷰에 투영해 per-view 2D+score를, RGB-D용으로 정렬 depth 맵을 만든 뒤
  **진짜 `Pipeline`**(삼각측량+RANSAC → depth fusion → One-Euro 스무딩)에 그대로 넣습니다.
- **출처 색 구분(3D 패널)** — 융합의 핵심을 한 장면에서 보여주려 관절별 출처를 색으로 구분:
  - **fused(초록):** 두 RGB 뷰 + depth 모두 유효 → confidence 가중 평균(대부분 관절).
  - **depth(파랑):** 두 RGB 뷰에서 가려진(score↓) **오른손목**을 depth가 보완 → README 제목의
    "가려진 관절 보완". RGB 패널엔 손목이 안 그려지고 depth 패널엔 그려지는 걸로 확인됩니다.
  - **triangulation(빨강):** depth 맵에 구멍 난 **왼발목**을 RGB 삼각측량만으로 복원.
- 옵션: `--num-frames N`(길이) · `--fps`(One-Euro 주파수 겸 재생 속도) · `--depth-min/--depth-max`
  (유효 depth 범위, m) · `--video <경로>`(출력 MP4).

---

## (참고) GPU 가속 (NVIDIA CUDA)

데모의 기본 디바이스는 `cuda`입니다. **GPU 셋업이 된 머신**에서는 그냥 `uv run python examples/...` 가
GPU로 돌고, **셋업이 안 된 환경**(또는 GPU 없는 머신)에서는 자동으로 CPU로 폴백합니다(검출기의
`resolve_device`가 결정). 측정 약 **8배** — CPU ~9.6 FPS → GPU ~78 FPS(RTX 4050).

### 1회 셋업 (Windows x86_64 · CUDA 12 · cuDNN 9.11)
```powershell
# (a) GPU 패키지 설치/복구 — 한 방에 (onnxruntime-gpu + CUDA 12 휠, cuDNN<9.12 핀, CUDA provider 검증)
./scripts/setup-gpu.ps1
# (b) uv run 이 환경을 CPU로 되돌리지 않도록 자동 동기화 끄기 (1회, 새 터미널부터 적용)
[Environment]::SetEnvironmentVariable("UV_NO_SYNC", "1", "User")
```
이후 데모는 플래그 없이 GPU로 실행되고, CPU로 비교하려면 `--device cpu` 만 붙이면 됩니다
(onnxruntime-gpu가 CPU provider도 포함하므로 패키지 교체 불필요).

### ⚠️ `uv sync` 로는 GPU가 안 됩니다 (검증됨)
`rtmlib` 이 CPU `onnxruntime` 을 **강제 의존**하는데 uv 로는 이 전이 의존성을 뺄 방법이 없고, CPU
`onnxruntime` 과 `onnxruntime-gpu` 는 **같은 `onnxruntime/` 폴더**에 설치돼 공존이 불가합니다 — 둘을
같이 깔면 CUDA provider가 사라집니다(실측: `[Tensorrt, CUDA, CPU]` → `[Azure, CPU]`). 따라서 `uv sync`
도, `uv sync --group gpu` 도 GPU를 깨뜨립니다.
- 그래서 `UV_NO_SYNC=1` 로 `uv run` 의 **자동** 동기화를 꺼 GPU 환경을 유지합니다.
- `uv sync` 를 직접 돌려 CPU로 되돌아갔다면 **`./scripts/setup-gpu.ps1` 한 번**으로 복구됩니다.
- `pyproject.toml`/`uv.lock` 의 `gpu` 그룹은 **버전 핀 참고용**입니다(설치는 위 스크립트 사용).

### 주의 · 동작 원리
- **cuDNN 9.12+ 비호환:** cuDNN 9.23은 onnxruntime 1.26의 CUDA provider에서 `CUDNN_BACKEND_API_FAILED`로
  실패해 조용히 CPU로 폴백합니다. 반드시 **`nvidia-cudnn-cu12<9.12`**(검증판 9.11.1.4) — 스크립트가 핀.
- **DLL 자동 로드 + 폴백:** `src/pose2d/rtmpose_detector.py`가 `device="cuda"`일 때
  `onnxruntime.preload_dlls()`로 nvidia 휠의 CUDA/cuDNN DLL을 로드하고, CUDA provider가 없으면 자동으로
  CPU로 폴백합니다(그래서 기본 `cuda`가 GPU 없는 환경에서도 안전).
- **확인:** `./scripts/setup-gpu.ps1` 가 끝에 provider 목록과 `GPU ready` 를 출력합니다.
  `cublasLt64_12.dll ... missing` 류 에러가 나면 nvidia 휠 설치가 빠진 것이니 스크립트를 다시 실행하세요.

---

## (참고) Calibration 절차

calibration 정확도가 전체 정확도의 병목이므로 reprojection error 리포트를 확인하세요.
체커보드를 새로 촬영한 뒤 `src/calibration/`의 `intrinsics`·`extrinsics`·`reprojection`·`camera_io` 모듈을 사용합니다.

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
- **Depth SDK:** 추상 `DepthFrameSource`(`src/io/depth_reader.py`, 합성 `DummyDepthSource`) +
  RGB-D 소스 어댑터 `src/io/sources/`(TUM·RealSense를 `RGBDSource`로 구현). 데모는 이 백엔드로 입력.
- **왜곡:** 삼각측량 전 픽셀 undistort, aligned depth는 rectified color grid 가정.

---

## 프로젝트 구조

```
config/cameras.yaml              calibration + 파이프라인 설정
src/
  core/{types,geometry}.py       공용 dataclass, COCO-17, 기하 프리미티브
  io/{frame_reader,depth_reader}.py  다중 뷰 동기 리더 + depth 추상화(ABC+Dummy)
  io/sources/{rgbd_source,tum,realsense,panoptic}.py  RGB-D/멀티뷰 데이터셋 어댑터
  io/keypoints_io.py             3D 포즈 JSON/NPY 직렬화
  pose2d/rtmpose_detector.py     rtmlib RTMPose 래퍼 (TORCH_HOME=./models)
  calibration/{intrinsics,extrinsics,reprojection,camera_io}.py  보정 + 리포트 + yaml I/O
  triangulation/{dlt,robust}.py  confidence 가중 DLT + robust 뷰 선택
  fusion/depth_fusion.py         depth back-projection + 융합
  smoothing/one_euro.py          One-Euro 시간적 필터
  render/{skeleton_2d,skeleton_3d,video_writer}.py  2D/3D 스켈레톤 오버레이·플롯·MP4
  pipeline.py                    엔드투엔드 오케스트레이션 (3D 전략 선택)
examples/
  realtime_demo.py               (2) 실시간 2D 17키포인트 추출 (실제 RTMPose, MP4)
  rgbd_video_demo.py             (3-A) RGB-D depth 활용 3D → MP4
  panoptic_video_demo.py         (3-B) 멀티뷰 HD 삼각측량 3D → MP4
  fusion_demo.py                 (3-C) (합성) RGB+Depth 융합 hybrid fused 경로 → MP4
run.py                           실제 라이브/녹화 멀티뷰 엔트리포인트
tests/                           수치 모듈 단위/통합 테스트
data/, models/, output/          (gitignore) 데이터셋 / 모델 캐시 / 산출물
```
