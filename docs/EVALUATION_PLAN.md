# 3D 스켈레톤 포즈 추정 시스템 — 성능평가 계획 (용역보고서용)

> 작성: 2026-06-16 · 교차검증: Claude · Codex(방법론·통계) · Gemini(벤치마크·보고서 구성) · 인터넷 자료 검증
> 대상 코드베이스: RGB 2대 + RGB-D 1대 기반 단일 인물 COCO-17 3D 포즈 추정(추론 전용)

---

## 0. 평가 대상 정의 (코드베이스 이해)

| 단계 | 모듈 | 성격 |
|---|---|---|
| 2D 검출 | `src/pose2d/rtmpose_detector.py` (rtmlib RTMPose) | **차용 모델** (사전학습, 본 시스템 기여 아님) |
| 왜곡보정 | `src/pipeline.py::_undistort` (cv2.undistortPoints) | 시스템 고유 |
| 삼각측량 | `src/triangulation/dlt.py` (confidence 가중 DLT) + `robust.py` (2-view RANSAC) | **시스템 고유 핵심** |
| depth 융합 | `src/fusion/depth_fusion.py` (back-projection + confidence 가중 평균) | **시스템 고유 핵심** |
| 스무딩 | `src/smoothing/one_euro.py` (온라인/causal) | 시스템 고유 |
| 출력 | COCO-17, world 좌표 meter, `valid`/`source` 태그 | — |

**핵심 통찰:** 평가의 본질은 *"차용한 RTMPose의 성능"이 아니라 "본 시스템이 추가한 기하 복원 파이프라인(삼각측량 + depth fusion + 스무딩)의 성능"* 을 분리 입증하는 것. 이것이 용역보고서의 방어선이다.

### 코드 현황에서 비롯된 실현 가능성 제약 (반드시 인지)
- Panoptic 경로는 현재 **삼각측량만**(depth_fusion off); depth fusion은 **TUM(사람 3D GT 없음)** 에서만 작동 → **depth fusion의 *정확도 향상*을 공개 GT로 직접 입증할 경로가 코드에 없음**. (평가 도구 추가 개발 필요 + 정직한 한계 명시)
- Panoptic GT(`hdPose3d_stage1_coco19`)를 **로드/비교하는 코드가 아직 없음** → GT 로더 + coco19↔coco17 매핑 + MPJPE 하니스를 **신규 작성** 필요.
- `src/calibration/reprojection.py::reprojection_report`는 이미 구현 → reprojection self-consistency는 저비용.
- One-Euro 스무딩은 `PoseSmoother.update(timestamp)` 온라인/causal → 지연(phase-lag) 정량화 대상.
- `min_views=2`, 단일 인물(`detect_best`) → 다인 미지원·2-view 퇴화는 한계로 명시.

---

## 1. ccg 교차검증 결과 요약

### 3자 합의 (그대로 채택)
- CMU Panoptic = 3D 정확도 **주 평가셋**. TUM RGB-D = 사람 pose GT 없음 → **보조(depth 안정성/런타임)로 강등**.
- **Ablation study가 보고서의 핵심**: 일반 DLT → +confidence 가중 → +depth fusion → +스무딩, 각 단계 MPJPE 개선 표.
- COCO-17엔 골반/목 없음 → `pelvis = (LHip + RHip)/2`, `neck = (LShoulder + RShoulder)/2` 보간 공식을 **부록에 명시**.
- 스무딩 jitter 감소는 **지연(phase-lag)·정확도 trade-off와 반드시 함께** 보고.

### 이견 · 중재 결정
| 쟁점 | Codex | Gemini | 채택 |
|---|---|---|---|
| 기여 분리 방법 | **Oracle 2D 실험**(GT 3D→GT 2D 투영 입력)으로 기하 파이프라인을 RTMPose와 분리 | 미언급 | **Codex 채택** — End-to-end + Oracle 2단 체계 |
| depth fusion 정확도 입증 | Panoptic에서만 정확도 주장, TUM은 안정성만 | **PROX**(RGB-D+SMPL GT)/Panoptic Kinect 활용 | **둘 다 반영** — 주 경로는 Panoptic 삼각측량, depth-fusion 정확도는 PROX/Panoptic-Kinect를 **선택(stretch)** 으로. 미수행 시 한계 명시 |
| 통계 집계 | **시퀀스 단위 + paired + 95% CI + valid-rate 분리** | 보고서 구조 중심 | **Codex 통계 + Gemini 구조** 결합 |

### 인터넷 검증된 비교군 수치
- RTMPose-m **75.8 AP** / RTMPose-s 72.2 AP (COCO val) — 2D는 SOTA급. (MMPose RTMPose README)
- 멀티뷰 삼각측량 MPJPE: algebraic 계열 H36M **수십 mm대**, volumetric SOTA 17.7mm (Iskakov 2019). 본 시스템(학습 없는 DLT+스무딩)은 **30~50mm 현실적 baseline 목표**.
- Panoptic GT = **coco19**(OpenPose18 + 골반중심), 순서가 COCO17과 **다름** → 매핑표 필수. (CMU panoptic-toolbox)

---

## 2. 추출 가능한 성능 지표 매트릭스

| # | 지표 | 정의 | 데이터셋 | 측정 대상 | 비고 |
|---|---|---|---|---|---|
| A1 | **Absolute MPJPE** | per-frame 정렬 **없이** world 좌표 직접 비교 (mm) | Panoptic | 시스템 **절대정확도** | 캘리브/스케일 오차 포함 |
| A2 | **Root-relative MPJPE** | pelvis 평행이동 제거 후 | Panoptic | 관절 배치 정확도 | pelvis 평가서 제외 |
| A3 | **PA-MPJPE** | Procrustes(R,t,s) 정렬 후 | Panoptic | **구조정확도** | 캘리브 오차 은닉 |
| A4 | **PCK3D / AUC** | 임계값(25/50/100mm) 이내 비율 + AUC | Panoptic | 강건성 | 단일 임계값 금지 |
| A5 | **Valid rate / coverage** | 관절별 복원 성공률 | Panoptic | 공정성 | A1~A4와 **분리 보고** |
| B1 | **Reprojection self-consistency** | 3D→각 뷰 재투영 px RMSE | Panoptic/실측 | 자기합치성 | 3D 정확도 아님(주의) |
| B2 | **Depth fusion gain** | error(tri) − error(tri+depth), subset별 | Panoptic-Kinect/PROX(선택) | depth 기여 | GT 없으면 정성+안정성으로 |
| B3 | **Smoothing jitter↓ vs lag** | 정지구간 가속도 RMS↓ + 동작구간 지연(ms) + MPJPE 변화 | Panoptic | 스무딩 trade-off | 3개 동시 |
| C1 | **Runtime 분해** | 단계별 ms/frame + FPS(throughput), CPU vs GPU | 전 데이터 | 실시간성 | warm-up 제외, 조건 명시 |
| C2 | **캘리브 민감도** | R(deg)/t(mm)/depth-scale 노이즈 sweep vs MPJPE | Panoptic | 강건성 | rot/trans 독립 sweep |
| D1 | **2D 참고성능** | RTMPose AP (공개 수치 인용) | COCO | 차용모델 | 직접 측정 불필요, 인용 |

---

## 3. 측정 방법론 (방어 가능한 설계)

### 3-1. 기여 분리 — 2단 실험 체계
- **End-to-end:** 실제 RTMPose 2D 입력 → 실사용 성능
- **Oracle 2D:** GT 3D를 각 카메라로 투영한 GT 2D 입력 → **기하 파이프라인 순수 성능 상한**(RTMPose 영향 제거)
- **Noise-oracle:** GT 2D + 가우시안 픽셀 노이즈(1/2/4 px) sweep → 2D 오차 민감도

### 3-2. Ablation (paired, 동일 프레임 on/off)
삼각측량 only → +confidence 가중 → +RANSAC → +depth fusion → +One-Euro / 2-view vs 3-view / undistort on-off

### 3-3. 정렬 프로토콜 분리
- A1 = 정렬 無 = 절대정확도, A3 = Procrustes = 구조정확도
- 절대정확도엔 **per-frame 정렬 금지**, 필요 시 시퀀스당 단일 rigid transform만 허용

### 3-4. 좌표계·단위 검증
Panoptic cm→m, `X_cam = R·X_world + t` 규약 일치 확인, 시간 동기/프레임 드롭 규칙 명시.

---

## 4. 통계 신뢰성 & 한계

### 통계 처리
- **집계 단위 = 시퀀스**(인접 프레임 상관 → frame N 과신 금지). per-sequence mean → 시퀀스 간 평균 ± 95% CI(bootstrap). Ablation은 **paired**.
- 결측·outlier: valid-joint accuracy와 coverage **분리**, median + 90/95 percentile 병행, 임의 프레임 제거 금지(규칙 사전 정의).

### 명시할 한계
single-person 전제(다인 미지원·`detect_best`) · Panoptic 도메인 편향(실내 돔·고품질 캘리브) · GT 동기/좌표 오차 · Panoptic GT 자체 오차 · 2D 차용 의존성 · One-Euro causal latency · 2-view RANSAC 퇴화(작은 baseline/교차각) · depth 센서 특성(홀·경계 bleed).

---

## 5. 용역보고서 목차 (권고)

1. Executive Summary (핵심 수치: "Absolute MPJPE __mm, PA-MPJPE __mm, GPU __FPS")
2. 시스템 아키텍처 (파이프라인 공학적 타당성)
3. 평가 방법론 (지표 정의·데이터셋·정렬 프로토콜·매핑표)
4. **정량 결과** (Ablation 표 = 핵심, CI 포함) + 비교군(SOTA/baseline 수치)
5. 정성 평가 (occlusion 보정 시각화, jitter 전후 그래프)
6. 런타임·강건성 (단계별 latency, 캘리브 민감도)
7. 한계 및 향후 과제
8. 부록 (coco19↔coco17 매핑표, 보간 공식, 실험 환경)

---

## 6. 실행 로드맵

1. **평가 하니스 구축**: Panoptic `hdPose3d_stage1_coco19` GT 로더 + coco19↔coco17 매핑표 + MPJPE/PA-MPJPE/PCK 계산 (신규)
2. Panoptic 단일 인물 시퀀스 5~10개 선정, 가시 프레임 추출
3. End-to-end / Oracle / Noise-oracle × Ablation 실행 → 시퀀스 단위 집계
4. 부가지표(reprojection, runtime 분해, 캘리브 민감도, jitter-lag) 측정
5. (선택) PROX/Panoptic-Kinect로 depth-fusion gain — 미수행 시 한계 명시
6. 보고서 작성 (목차 5)

---

## 부록 A. coco19(Panoptic) → COCO-17 매핑 (초안, 구현 시 확정)

Panoptic coco19 순서(OpenPose 기반): `0 Neck, 1 Nose, 2 BodyCenter(=pelvis), 3 LShoulder, 4 LElbow, 5 LWrist, 6 LHip, 7 LKnee, 8 LAnkle, 9 RShoulder, 10 RElbow, 11 RWrist, 12 RHip, 13 RKnee, 14 RAnkle, 15 LEye, 16 LEar, 17 REye, 18 REar`

COCO-17 순서: `0 nose, 1 LEye, 2 REye, 3 LEar, 4 REar, 5 LShoulder, 6 RShoulder, 7 LElbow, 8 RElbow, 9 LWrist, 10 RWrist, 11 LHip, 12 RHip, 13 LKnee, 14 RKnee, 15 LAnkle, 16 RAnkle`

→ 공통 17개를 coco19에서 직접 인덱싱(Neck/BodyCenter는 평가 제외, pelvis는 BodyCenter 또는 (LHip+RHip)/2로 root 정렬에만 사용). **구현 단계에서 실제 JSON 필드로 검증 후 확정.**
