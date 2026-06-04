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
   │  (1) 뷰별 RTMPose 2D 키포인트          src/pose2d/rtmpose_detector.py
   ▼  (2) calibration -> projection matrix   src/calibration/calibrate.py
 삼각측량 (confidence 가중 DLT + robust 뷰 제외)
   │                                          src/triangulation/{dlt,robust}.py
   ▼  (4) depth fusion (back-projection + 융합)  src/fusion/depth_fusion.py
 시간적 스무딩 (One-Euro 필터)                 src/smoothing/one_euro.py
   ▼
[3D 스켈레톤 + 시각화/저장]                     src/viz/visualize_3d.py
```

오케스트레이션: `src/pipeline.py`, 엔트리포인트: `run.py`.

---

## 설치

[uv](https://github.com/astral-sh/uv)가 필요합니다.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

`requirements.txt`는 `onnxruntime-gpu`(CUDA NVIDIA GPU 필요)로 고정되어 있습니다.
**GPU가 없는 환경**에서는 `onnxruntime-gpu`를 주석 처리하고 `onnxruntime`을 활성화한 뒤,
config의 `detection.backend`를 `cpu`로 바꾸면 됩니다(또는 데모용으로
`uv pip install onnxruntime`만 추가 설치).
`rtmlib`은 첫 실행 시 RTMPose ONNX 모델을 인터넷에서 자동 다운로드합니다.

---

## 데모 실행 가이드

> 모든 명령은 프로젝트 루트(`D:\Programming\MMPose`)에서 실행합니다.
> 산출물은 `output/` 폴더에 저장됩니다.

### 1) 실시간 2D 키포인트 추출 데모 (실제 RTMPose 추론)

단일 뷰 이미지/웹캠에서 **COCO-17 키포인트 17개를 프레임마다 추출**하고, FPS와 키포인트 좌표를
출력하며, 스켈레톤을 그린 **MP4 영상(전 프레임)**과 마지막 프레임 PNG를 저장합니다.
(`rtmlib` + `onnxruntime` 필요)

```bash
# (a) 정적 이미지를 스트림처럼 30프레임 반복 — 가장 간단한 시작
uv run python examples/realtime_demo.py --frames 30
#   -> 콘솔에 프레임별 FPS + 17개 키포인트 표
#   -> output/realtime_keypoints.mp4 (전 프레임) + output/realtime_keypoints.png (마지막 프레임)

# (b) 실제 웹캠 라이브 (장치 0번, 200프레임) — 200프레임 전체가 MP4로 저장됨
uv run python examples/realtime_demo.py --camera 0 --frames 200

# (c) GPU로 고속 추론 (NVIDIA + onnxruntime-gpu 설치 시, 30+ FPS)
uv run python examples/realtime_demo.py --camera 0 --device cuda --mode balanced

# 옵션:
#   --image <경로>     다른 이미지 사용
#   --mode             lightweight|balanced|performance (속도/정확도 조절)
#   --video <경로>     MP4 저장 경로 (기본 output/realtime_keypoints.mp4)
#   --fps <값>         저장 MP4의 재생 프레임레이트 (기본 30)
#   --out <경로>       마지막 프레임 PNG 경로
```

> 데모용 샘플 사진이 없다면 아래로 받을 수 있습니다(사람이 포함된 임의 이미지면 무엇이든 가능):
> ```bash
> curl -sSL -o data/demo/person.jpg "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg"
> ```
> 참고 성능: CPU(RTMPose-s, lightweight) 정상상태 약 **11~12 FPS**, GPU에서는 30+ FPS로 실시간 동작.

### 2) 합성 데이터 3D 복원 데모 (calibration·카메라 불필요)

알려진 3D 스켈레톤을 config의 카메라들로 투영 → **전체 파이프라인(삼각측량→depth fusion→스무딩)**으로
다시 3D 복원합니다. rtmlib/GPU/카메라/calibration 없이 **실제 파이프라인 코드**가 그대로 돕니다.

```bash
# 30프레임 + 3D 스켈레톤 시각화 저장
uv run python run.py --config config/cameras.yaml --synthetic --frames 30 --viz
#   -> output/poses_3d.json (프레임별 3D 키포인트) + output/skeleton_frame0.png

# 시각화 없이 빠르게 (JSON만)
uv run python run.py --synthetic --frames 30

# 단일 프레임만
uv run python run.py --synthetic --frames 1 --viz
```

기대 출력 예시:

```
[synthetic] frames=30 keypoints=17 valid(frame0)=17/17 mean3D_err(frame0)=8.7e-09 m
```

복원 오차 ≈ 1e-8 m(수치 정밀도 수준) → 삼각측량·depth fusion 구현이 정확함을 의미합니다.
시각화에서 사람이 누워 보이는 것은 카메라 좌표계 관례(+Y가 아래쪽) 때문이며 기하학적으로 정상입니다.

### 2-b) 3D 포즈 추정 결과를 영상(MP4)으로 렌더링

전체 3D 파이프라인을 프레임 시퀀스에 돌려, 복원된 **3D 스켈레톤을 프레임마다 렌더링한 MP4**로 저장합니다.

```bash
uv run python examples/pose3d_video.py --frames 90
#   -> output/pose3d_demo.mp4 (3D 스켈레톤 애니메이션) + output/pose3d_demo_frame.png

# 노이즈를 주입해 One-Euro 스무딩 효과를 확인
uv run python examples/pose3d_video.py --frames 120 --jitter 0.01
```

> ⚠️ **실제 영상으로 3D를 추정하려면 calibration된 2대 이상의 동기화 카메라가 필요합니다**(삼각측량의 전제).
> 단일 뷰 영상으로는 기하학적으로 불가능하므로 이 데모는 합성 멀티뷰 2D로 실제 파이프라인을 구동합니다.
> 실제 멀티뷰 녹화가 준비되면 각 뷰를 `RTMPoseDetector` → `pipeline.process`에 넣으면 동일 코드로 동작합니다.
> (참고: 스무딩 ON + 빠른 모션에서는 One-Euro 지연으로 순간 오차가 보일 수 있으며, 복원 자체는 정확합니다.)

### 2-c) 실제 공개 데이터셋(CMU Panoptic)으로 3D 복원

**실제 멀티뷰 데이터셋의 calibration + 실제 캡처된 3D 포즈**로 파이프라인을 검증합니다.
CMU Panoptic Studio의 실제 31-HD-카메라 calibration과 데이터셋이 제공하는 실제 사람의 3D
GT를 사용합니다. (Apache-2.0, GitHub raw에서 작게 직접 다운로드)

먼저 데이터(작음: calibration + GT 프레임)를 받습니다:

```bash
BASE=https://raw.githubusercontent.com/open-mmlab/mmpose/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/tests/data/panoptic_body3d/160906_band1
mkdir -p data/panoptic/160906_band1
curl -sSL -o data/panoptic/160906_band1/calibration_160906_band1.json "$BASE/calibration_160906_band1.json"
curl -sSL -o data/panoptic/160906_band1/body3DScene_00000168.json "$BASE/hdPose3d_stage1_coco19/body3DScene_00000168.json"
```

실행:

```bash
uv run python examples/panoptic_demo.py --num-views 4 --noise-px 3.0
#   -> output/panoptic_pose3d.png (복원 3D vs GT 오버레이)

# 옵션: --num-views 2|3|4  --noise-px <px>  --body 0|1|2(장면 내 인물)  --seed N
```

기대 출력(예):

```
[panoptic] using 4 views: ['00_00', '00_03', '00_10', '00_05']
[panoptic] reconstructed 16/17 joints | mean 3D error vs GT = 6.3 mm (median 6.2 mm)
[panoptic]   view 00_00: reprojection RMS = 2.65 px   ...
```

- 노이즈 0으로 주면 **0.0 mm / 0.00 px** → calibration 파싱·삼각측량이 정확함을 의미.
- 뷰 수를 늘리고 노이즈를 줄일수록 정확해짐(2뷰+5px ≈ 12.8 mm, 4뷰+3px ≈ 6.3 mm).

> ⚠️ **정직한 한계:** Panoptic의 동기화 영상은 무료로 작게 받을 수 없어(영상 수 GB, test
> 샘플엔 이미지 미포함), 이 데모는 RTMPose를 '실제 픽셀'에 돌리는 대신 **실제 calibration으로
> 실제 GT 3D를 각 카메라에 투영한 2D(+검출 노이즈)**를 입력으로 씁니다. 실제 프레임이 준비되면
> `examples/panoptic_demo.py`에서 투영 2D를 `RTMPoseDetector(...).detect_best(frame_v).keypoints`로
> 바꾸기만 하면 동일하게 동작합니다(단, 다인 장면은 뷰 간 인물 매칭이 추가로 필요).

### 2-d) 2 RGB + 1 Depth 하이브리드 결과영상 (당신의 리그 구조)

당신의 실제 리그(RGB 2대 + Depth 1대)를 그대로 모사해, 전체 하이브리드 파이프라인
(삼각측량 + depth fusion + One-Euro)을 돌리고 **3개 입력 뷰 + 3D 복원을 4분할로 합성한
결과 MP4**를 만듭니다. (입력 2D/depth는 합성 — 공개로 받을 수 있는 2RGB+1Depth 동기화
영상이 없어서)

```bash
uv run python examples/hybrid_3cam_demo.py --frames 120
#   -> output/hybrid_pose3d.mp4 (960x720): [cam0 RGB|cam1 RGB / cam2 Depth|3D 복원]

# 노이즈 주입 시 스무딩 효과 확인
uv run python examples/hybrid_3cam_demo.py --frames 120 --jitter 0.01
```

> 실제 영상이 준비되면 각 RGB 뷰는 `RTMPoseDetector(...).detect_best(frame).keypoints`,
> Depth 뷰는 실제 depth 맵(미터)으로 바꾸면 동일 파이프라인이 그대로 동작합니다
> (`config/cameras.yaml`의 calibration 값 필요).

### 2-e) 실제 CMU Panoptic HD 영상으로 3D 복원 (RTMPose-on-pixels)

실제 멀티뷰 영상에 RTMPose를 돌려 삼각측량으로 3D를 복원하는 **진짜 실영상 데모**.
**단일 인물 시퀀스**를 쓰세요(다인 장면은 뷰 간 인물 매칭이 추가로 필요 — 미구현).

HD 카메라당 ~2.8GB(3대 ≈ 8.6GB), `-C -`로 이어받기 가능. **반드시 프로젝트 루트에서 실행.**

#### Windows PowerShell

```powershell
# 1) 다운로드
$SEQ = "171204_pose1"
$D = "http://domedb.perception.cs.cmu.edu/webdata/dataset/$SEQ"
New-Item -ItemType Directory -Force "data\panoptic\$SEQ\hdVideos" | Out-Null
curl.exe -L -o "data\panoptic\$SEQ\calibration_${SEQ}.json" "$D/calibration_${SEQ}.json"
foreach ($n in '03','12','23') {
  curl.exe -C - -L -o "data\panoptic\$SEQ\hdVideos\hd_00_${n}.mp4" "$D/videos/hd_shared_crf20/hd_00_${n}.mp4"
}
# (선택) 3D GT — 정확도 비교용
curl.exe -L -o "data\panoptic\$SEQ\hdPose3d_stage1_coco19.tar" "$D/hdPose3d_stage1_coco19.tar"
tar -xf "data\panoptic\$SEQ\hdPose3d_stage1_coco19.tar" -C "data\panoptic\$SEQ"

# 2) 다운로드 확인 (세 파일이 각각 ~2700MB면 정상)
Get-ChildItem "data\panoptic\$SEQ\hdVideos" | Select-Object Name, @{n="MB";e={[int]($_.Length/1MB)}}

# 3) 실행 (한 줄)
uv run python examples/panoptic_video_demo.py --seq-dir data/panoptic/171204_pose1 --cams 00_03,00_12,00_23 --start 500 --num-frames 60 --device cuda
```

> PowerShell의 `curl`은 `Invoke-WebRequest` 별칭일 수 있어 **`curl.exe`**로 명시했습니다.
> `$n.mp4`가 "변수 n의 .mp4 속성"으로 오해되지 않도록 `${n}`/`${SEQ}` 중괄호를 씁니다.

#### Linux / macOS / Git Bash

```bash
# 1) 다운로드
SEQ=171204_pose1
D=http://domedb.perception.cs.cmu.edu/webdata/dataset/$SEQ
mkdir -p data/panoptic/$SEQ/hdVideos
curl -L -o "data/panoptic/$SEQ/calibration_$SEQ.json" "$D/calibration_$SEQ.json"
for n in 03 12 23; do
  curl -C - -L -o "data/panoptic/$SEQ/hdVideos/hd_00_$n.mp4" "$D/videos/hd_shared_crf20/hd_00_$n.mp4"
done
curl -L -o "data/panoptic/$SEQ/hdPose3d_stage1_coco19.tar" "$D/hdPose3d_stage1_coco19.tar"
tar -xf "data/panoptic/$SEQ/hdPose3d_stage1_coco19.tar" -C "data/panoptic/$SEQ/"

# 2) 실행
uv run python examples/panoptic_video_demo.py \
    --seq-dir data/panoptic/171204_pose1 \
    --cams 00_03,00_12,00_23 --start 500 --num-frames 60 --device cuda
```

산출물: `output/panoptic_video_pose3d.mp4` (HD 뷰들 2D + 3D 복원). 프레임은 mp4에서 직접
읽으므로 별도 추출 불필요. 대안: 공식 툴박스 `CMU-Perceptual-Computing-Lab/panoptic-toolbox`
→ `./scripts/getData.sh 171204_pose1 0 3` (HD 3대), 느리면 `--snu-endpoint`(SNU 미러).

- **GPU 권장**(`--device cuda` + `onnxruntime-gpu`); CPU도 가능하나 느립니다.
- depth(Kinect)는 `.dat` 원시 포맷 디코딩/동기/정렬이 필요해 이 데모는 **HD RGB 삼각측량만**
  수행합니다(depth fusion off). depth까지 쓰려면 `getData_kinoptic.sh`로 받은
  `kcalibration_*.json` + `KINECTNODE*/depthdata.dat`를 디코딩해 aligned depth맵(미터)을
  `pipeline.process(depth_map=...)`로 넘기면 동일 파이프라인에 연결됩니다.

### 3) 실제 3D 라이브 추론 (calibration 완료 후)

`config/cameras.yaml`에 3대 카메라의 calibration(K/dist/R/t)을 채운 뒤, `--synthetic` 없이 실행하면
동일 파이프라인이 실제 영상에 적용됩니다.

```bash
uv run python run.py --config config/cameras.yaml
```

### 4) 테스트 (모든 수치 모듈은 합성 데이터 단위 테스트 포함)

```bash
uv run pytest tests/ -q
#   삼각측량/calibration/fusion/스무딩/파이프라인 검증 (rtmlib 없이 오프라인 동작)
```

---

## Calibration 절차

calibration 정확도가 전체 시스템의 병목이므로 항상 reprojection error 리포트를 확인하세요.
체커보드를 촬영(이 리그는 새 촬영 필요)한 뒤 `src/calibration/calibrate.py`를 사용합니다.

1. **Intrinsic** — 카메라별 체커보드 이미지를 모아
   `find_checkerboard_corners(images, pattern_size=(cols,rows), square_size_m)`
   → `calibrate_intrinsics(...)` → `(K, dist, rms)`.
2. **Extrinsic** — 3대가 동시에 보는 보드를 촬영해 카메라별 `estimate_board_pose(...)` 후
   `calibrate_extrinsics(board_poses, reference="cam0", world_frame="reference_camera")`
   → world→camera `(R, t)`.
3. **저장** — 카메라별 `build_camera_params(...)` 후 `save_cameras_yaml(cameras, "config/cameras.yaml")`.
4. **검증** — `reprojection_report(cameras, observations)`로 RMS 확인
   (목표 ≲1 px). 큰 오차는 3D 오차로 직결됩니다.

---

## 설정 (`config/cameras.yaml`)

| 섹션 | 주요 필드 |
|---|---|
| `units` | `length: meter` (전 모듈 공통) |
| `world` | `frame` (`reference_camera` \| `board_origin`), `reference_camera` |
| `cameras[]` | `name, type (rgb\|rgbd), K, dist, R, t, image_size, source`; rgbd는 추가로 `depth_K, depth_scale, depth_to_color_R/t` |
| `detection` | `backend (cuda\|cpu), model, mode, det_score_threshold` |
| `triangulation` | `min_views, score_threshold, ransac.{enabled,reproj_threshold_px}` |
| `depth_fusion` | `enabled, depth_min, depth_max, fill_missing, patch_radius_px, depth_weight` |
| `smoothing` | `enabled, freq, min_cutoff, beta, d_cutoff` |
| `input` | `mode (live\|file), sync (hardware\|software), sync_tolerance_ms` |
| `output` | `format (json\|npy), path` |
| `seed` | 재현성을 위한 난수 시드 (`null`이면 비활성) |

기본 제공되는 `K/dist/R/t` 값은 **placeholder**이지만 동시에 유효한 합성 리그를 구성하므로
`--synthetic` 데모가 바로 동작합니다. 실제 사용 시 calibration 결과로 교체하세요.

---

## 설계 결정 (고정)

- **단위:** 전 모듈 meter, 픽셀은 `(u, v)` 순서.
- **World 좌표계:** 기준 카메라 `cam0` (따라서 `cam0`은 `R = I`, `t = 0`).
  `world.frame`으로 calibration 보드 원점으로 전환 가능.
- **Extrinsic:** world → camera 변환 `X_cam = R·X_world + t`, `P = K[R|t]`.
- **키포인트:** COCO-17, 모든 뷰에서 동일한 인덱스 순서(다중 뷰 삼각측량 대응의 전제).
- **Confidence:** 2D `score`를 삼각측량 가중치(뷰별)와 depth fusion 가중치 양쪽에 사용 →
  가려진/저신뢰 관절은 자동으로 down-weight 또는 제외.
- **Depth 카메라 SDK:** 미정 → 취득은 추상 `DepthFrameSource`(`src/io/depth_reader.py`) 뒤에 두고
  `Dummy`/`File` 백엔드 제공. RealSense/Kinect/Orbbec 백엔드는 수학 로직 변경 없이 나중에 연결.
- **왜곡(distortion):** 삼각측량 전 픽셀을 undistort, aligned depth는 rectified color grid에 있다고
  가정(합성 데이터에서는 정확).

---

## 프로젝트 구조

```
config/cameras.yaml              calibration + 파이프라인 설정
src/
  core/{types,geometry}.py       공용 dataclass, COCO-17, 기하 프리미티브
  io/{frame_reader,depth_reader}.py  다중 뷰 동기 리더 + depth 소스 추상화
  pose2d/rtmpose_detector.py     rtmlib RTMPose 래퍼
  calibration/calibrate.py       intrinsic/extrinsic, reprojection 리포트, yaml I/O
  triangulation/{dlt,robust}.py  confidence 가중 DLT + robust 뷰 선택
  fusion/depth_fusion.py         depth back-projection + 융합
  smoothing/one_euro.py          One-Euro 시간적 필터
  viz/visualize_3d.py            3D 플롯 + JSON/NPY 저장
  synthetic.py                   오프라인 합성 데이터 생성기
  pipeline.py                    엔드투엔드 오케스트레이션
examples/
  realtime_demo.py               실시간 2D 키포인트 추출 데모 (실제 RTMPose, MP4 저장)
  pose3d_video.py                합성 멀티뷰 3D 포즈를 MP4로 렌더링
  hybrid_3cam_demo.py            2 RGB + 1 Depth 리그 4분할 3D 결과영상
  panoptic_demo.py               실제 CMU Panoptic calibration+GT로 3D 복원 (이미지 불필요)
  panoptic_video_demo.py         실제 Panoptic HD 영상에 RTMPose → 3D 결과영상
data/demo/person.jpg             2D 데모용 샘플 이미지(직접 받음)
data/panoptic/                   CMU Panoptic calibration + 3D GT(직접 받음)
tests/                           합성 데이터 단위/통합 테스트
run.py                           엔트리포인트
```
