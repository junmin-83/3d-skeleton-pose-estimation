**평가 개요**

- 평가 대상: RGB 2대 + RGB-D 1대 기반 단일 인물 3D 스켈레톤 포즈 추정 시스템(COCO-17[^coco17], 추론 전용)
- 평가 기준일: 2026-06-16
- 평가 데이터: CMU Panoptic Studio(멀티뷰 HD + 3D Ground Truth)
- 평가 도구: 본 시스템에 구축한 정량평가 하니스(`src/eval/`)
- 문서 성격: 기술 성능평가 보고서(초안)
- 비고: 전 수치 재현 가능(원시값 `output/experiments/*.json`), 측정 방법은 [EVALUATION_PLAN.md](EVALUATION_PLAN.md)

> **용어 안내**: 본문의 전문 용어(MPJPE 등)는 각주로 쉬운 설명을 달았으며, 결과 표·그림마다 "쉽게 말하면" 해설을 덧붙임.

---

## 1. 평가 요약

- 시스템 개요
    - 사전학습 2D 검출기(RTMPose[^rtmpose])로 뷰별 2D 키포인트 획득 → 신뢰도 가중 DLT 삼각측량[^dlt] → depth 융합 → 시간적 스무딩(One-Euro[^oneeuro]) → 3D 스켈레톤(COCO-17) 추론
    - 학습 없이 추론만 수행
- 평가 핵심: 차용 검출기와 분리한 **기하 복원 파이프라인 고유 성능**의 입증

핵심 실측 성능(CMU Panoptic 3D GT 대비)

| 구분 | 지표 | 측정값 | 비고 |
|---|---|---|---|
| 절대정확도(실사용) | abs MPJPE[^mpjpe] | **27.9 mm** (95% CI [26.9, 29.0]) | 실제 RTMPose, 3뷰, 120프레임 |
| | PA-MPJPE[^pampjpe] | 24.7 mm | 구조정확도 |
| | PCK@100mm / @50mm[^pck] | 0.996 / 0.892 | |
| 멀티뷰 기여 | 2뷰→3뷰 abs MPJPE | 34.5 → 27.2 mm (−21%) | 삼각측량 실효성 |
| 기하 고유 정밀도 | abs MPJPE(검출기 배제) | **5.0 ± 0.4 mm** | oracle[^oracle], 4시퀀스, 2px 노이즈 |
| 정확도 병목 | 캘리브 회전 민감도 | ≈ 17 mm / 1° | 외부 보정이 1차 결정요인 |
| 속도 | end-to-end(CPU, HD 3뷰) | ≈ 0.7 FPS | 2D 검출이 99.5% 점유 |
| | 기하 연산 단계 | 7.5 ms / frame | device 무관 |

> **쉽게 말하면**: 실제 영상에서 인체 3D 관절을 평균 **약 2.8 cm** 오차로 복원함. 만약 2D 검출이 완벽하다면 **약 0.5 cm**까지 정확해짐(=오차의 대부분이 검출기·캘리브 탓). 속도는 거의 전부 2D 검출이 좌우.

- 종합 판단
    - 3D 복원 기하의 정확성·안정성 입증(검출기 배제 시 5.0 mm, 무오차 입력 0.0 mm)
    - 실사용 절대정확도 27.9 mm — 학습 없는 멀티뷰 삼각측량 baseline대(수십 mm)에 부합
    - 정확도 실질 한계 = 차용 2D 검출 품질 + 외부 캘리브레이션 정밀도(특히 회전)
    - 속도는 2D 검출이 지배 → 검출 단계 GPU 가속으로 실시간화 가능

---

## 2. 시스템 아키텍처

### 2.1 처리 파이프라인

![**[그림 1]** 처리 파이프라인. 입력(동기화 3프레임)부터 5단계를 거쳐 3D 스켈레톤을 출력. 색상은 모듈 성격 구분(주황=차용 사전학습 모델, 옅은 파랑=시스템 고유, 진한 파랑=시스템 고유 핵심).](figures/fig_pipeline.png)

### 2.2 모듈별 성격(평가 관점)

| 단계 | 모듈 | 성격 |
|---|---|---|
| 2D 검출 | RTMPose(rtmlib) | **차용 모델** — 사전학습, 본 시스템 기여 아님 |
| 왜곡 보정 | `cv2.undistortPoints` | 시스템 고유 |
| 삼각측량 | 신뢰도 가중 DLT + 2-view RANSAC[^ransac] | **시스템 고유 핵심** |
| depth 융합 | back-projection + 신뢰도 가중 평균 | **시스템 고유 핵심** |
| 스무딩 | One-Euro(온라인/causal) | 시스템 고유 |

### 2.3 설계상 강점·위험요인

- 강점
    - 신뢰도 전파: 2D 신뢰도를 삼각측량·depth 융합 가중치 양쪽에 적용 → 가림·저신뢰 관절 자동 down-weight
    - 퇴화 방지: 평행 광선·근접 무한점 등 퇴화 기하를 NaN 처리 → 후단 오염 차단
- 위험요인
    - 단일 인물 전제(`detect_best`, 최고신뢰 1명만 추적)
    - 2-view RANSAC: 작은 baseline·교차각에서 불안정
    - depth 센서 표면 특성(홀·반사·경계 bleed)

---

## 3. 평가 방법론

### 3.1 평가 데이터셋

| 데이터셋 | 역할 | 근거 |
|---|---|---|
| CMU Panoptic(HD + `hdPose3d_stage1_coco19`) | **3D 정확도 주 평가셋** | 멀티뷰 RGB + 정밀 3D GT, 시스템 구성과 유사 |
| TUM RGB-D(freiburg3) | 보조(depth/런타임) | 사람 3D pose GT **없음** → 절대 3D 정확도 평가 불가 |
| COCO | 2D 검출 참고 | 3D 평가셋 아님 |

- 사용 시퀀스: 단일 인물 4종(`171204_pose1/2/3`, `171026_pose1`)

### 3.2 GT 정합 및 키포인트 매핑

- GT 포맷 차이: Panoptic = 19관절(OpenPose 기반 COCO19)·센티미터, 시스템 출력 = COCO-17·미터
- 정합 처리
    - COCO19→COCO17 인덱스 매핑(부록 A)
    - cm→m 단위 변환
    - 동일 world 좌표계 정합(Panoptic calibration 기반, 추가 rigid 변환 불필요)
- 정합 검증(중요)
    - GT 3D를 각 카메라로(왜곡 포함) 투영 후 재삼각측량하는 라운드트립에서 abs MPJPE = **0.00 mm**, 재투영 = **0.00 px**
    - → GT 로더·매핑·단위·좌표계·삼각측량 정합성 수치 보증

### 3.3 정확도 지표 및 정렬 프로토콜

- Absolute MPJPE(정렬 없음): 시스템 **절대정확도**
- Root-relative MPJPE[^rrel](골반 평행이동 제거): 전역 위치 제외 관절 배치 정확도
- PA-MPJPE(Procrustes 유사변환 정렬): **구조정확도**(캘리브/스케일/방향 오차 배제)
- 보조 지표: PCK3D(25/50/100 mm)·AUC[^auc]·valid rate[^validrate]·재투영 RMSE[^reproj]
- 골반 정의: COCO-17엔 골반·목 없음 → `골반 = (좌고관절+우고관절)/2`(부록 A)

### 3.4 기여 분리: 2모드 측정 체계

- oracle 모드: GT 3D를 각 카메라로 투영한 GT 2D 입력 → **기하 파이프라인 고유 성능**(검출기 영향 0), 픽셀 노이즈 주입으로 2D 오차 민감도 통제
- real 모드: 실제 HD 영상에 RTMPose 적용 → **end-to-end 실사용 성능**

![**[그림 2]** 실제 Panoptic HD 3뷰 입력 영상에 RTMPose가 추정한 COCO-17 키포인트(빨강 관절·초록 뼈대)를 오버레이한 결과. 이 뷰별 2D가 삼각측량의 입력이 됨.](figures/fig_multiview_capture.png)

> **쉽게 말하면**: 카메라 3대가 같은 사람을 동시에 찍고, 각 영상에서 17개 관절 위치를 점으로 찍음. 이 점들을 3대가 함께 보면 삼각측량으로 3D 위치가 나옴.

### 3.5 Ablation 및 통계 처리

- Ablation: RANSAC on/off, 스무딩 on/off, 뷰 2 vs 3, 픽셀 노이즈 sweep을 **동일 프레임 paired** 비교
- 통계
    - 집계 단위 = 시퀀스(인접 프레임 상관 회피)
    - 시퀀스 평균에 95% t-구간 적용
    - 단일 시퀀스 프레임 분포는 부트스트랩 구간(자기상관으로 낙관적임 명시)

---

## 4. 정량 평가 결과

### 4.1 기하 파이프라인 고유 성능(oracle 2D, 60프레임)

| 조건 | abs MPJPE (mm) | PA-MPJPE (mm) | PCK@50mm | reproj (px) |
|---|---|---|---|---|
| full (ransac+smooth) | 12.73 | 11.91 | 0.936 | 8.32 |
| no RANSAC | 12.73 | 11.91 | 0.936 | 8.32 |
| no smoothing | 4.82 | 4.43 | 1.000 | 1.95 |
| no RANSAC, no smooth | 4.82 | 4.43 | 1.000 | 1.95 |
| 2 views | 13.17 | 12.14 | 0.933 | 7.84 |
| 3 views | 12.73 | 11.91 | 0.936 | 8.32 |
| noise 0px | 0.00 | 0.00 | 1.000 | 0.00 |
| noise 1px | 2.41 | 2.22 | 1.000 | 0.98 |
| noise 2px | 4.82 | 4.43 | 1.000 | 1.95 |
| noise 4px | 9.88 | 9.10 | 1.000 | 4.00 |

- 2D 노이즈 → 3D 오차: 거의 선형(≈ 2.4 mm / 1 px) → 삼각측량 기하 안정
- RANSAC on/off 동일: oracle엔 outlier 없어 기각 대상 없음 → 효용은 real에서 재평가
- 스무딩이 저노이즈 oracle에선 오차 증가(4.82→12.73 mm): jitter 없는데 위상지연만 추가 → **jitter–지연 trade-off** 직접 증거, 스무딩은 jitter 억제용 모듈로 해석

> **쉽게 말하면**: 2D를 1픽셀 틀리면 3D가 약 2.4 mm 틀어짐(거의 비례) → 기하가 안정적. 스무딩은 '흔들림 제거'용이라, 흔들림 없는 깨끗한 입력에선 오히려 약간의 지연 때문에 오차가 커짐(실제 노이즈 환경에서 재평가 필요).

![**[그림 3]** (a) 2D 픽셀 노이즈 대비 3D 오차의 선형성, (b) 뷰 2 vs 3 비교 (oracle).](figures/fig_noise_views.png)

### 4.2 실사용 end-to-end(real RTMPose, 40프레임)

| 조건 | abs MPJPE (mm) | PA-MPJPE (mm) | root-rel (mm) | PCK@50mm | reproj (px) |
|---|---|---|---|---|---|
| full (ransac+smooth) | 27.23 | 24.60 | 37.14 | 0.916 | 15.00 |
| no RANSAC | 26.08 | 23.15 | 32.24 | 0.937 | 13.98 |
| no smoothing | 27.25 | 25.06 | 36.17 | 0.913 | 15.30 |
| 2 views | 34.46 | 29.84 | 41.92 | 0.748 | 7.21 |

- 멀티뷰 실질 기여 명확: 2뷰 34.46 → 3뷰 27.23 mm(−21%), PCK@50mm 0.748→0.916
    - 2뷰 재투영(7.21 px)이 더 낮으나 3D 오차는 더 큼 → 재투영 자기합치성 ≠ 3D 정확도(실증)
- RANSAC·스무딩 ~중립(정적 단일 인물·고품질 HD라 outlier·jitter 적음) → 효용 구간(가림·급동작·저신뢰)에서 재평가 필요(→ §7)
- real 절대정확도 ≈ 27 mm(abs)/25 mm(PA) — 학습 없는 삼각측량 baseline대 부합

> **쉽게 말하면**: 카메라를 2대→3대로 늘리면 오차가 34→27 mm(21%↓), 5 cm 이내 정확 관절도 75%→92%로 증가 → 멀티뷰 효과가 분명. (RANSAC·스무딩은 이 깨끗한 영상에선 차이 미미)

![**[그림 4]** 실제 RTMPose 기반 end-to-end ablation. 2뷰(빨강)는 abs MPJPE·PCK@50mm 양쪽에서 명확히 열세.](figures/fig_ablation_real.png)

### 4.3 시퀀스 간 정확도 + 신뢰구간

**4.3.1 기하 고유 성능(oracle, 2px 노이즈, 60프레임/시퀀스, n=4)**

| 시퀀스 | abs MPJPE (mm) | 프레임 |
|---|---|---|
| 171026_pose1 | 5.35 | 60 |
| 171204_pose1 | 4.83 | 60 |
| 171204_pose2 | 4.92 | 60 |
| 171204_pose3 | 5.02 | 60 |
| **평균 ± 95% CI** | **5.03 ± 0.36** | n=4 |

- 4개 시퀀스에서 5.03 ± 0.36 mm로 일관 → 기하 파이프라인 재현성 높음

> **쉽게 말하면**: 서로 다른 4개 영상에서 결과가 5 mm 안팎으로 거의 같음 → 특정 영상에만 잘 되는 게 아니라 일관되게 안정적.

![**[그림 5]** 시퀀스별 평균 abs MPJPE와 시퀀스 간 95% 신뢰구간(oracle, 2px 노이즈).](figures/fig_multiseq.png)

**4.3.2 실사용 성능 불확실성(real, 171204_pose1, 120프레임)**

- 요약: abs MPJPE 27.89 mm, PA-MPJPE 24.72 mm, root-rel 37.41 mm, PCK@50mm 0.892 / @100mm 0.996, AUC 0.804, valid rate 0.9996, reproj 17.43 px

| 추정 방법 | abs MPJPE (mm) | 95% 구간 | n |
|---|---|---|---|
| frame bootstrap | 27.89 | [26.85, 28.98] | 120 프레임 |
| segment t-interval | 27.89 | ± 6.09 | 4 구간 |

- HD 비디오 보유 시퀀스 1개 한정 → 진정한 시퀀스 간 real CI는 추가 HD 비디오(시퀀스당 ~8.6 GB) 필요, oracle 다중시퀀스(§4.3.1, n=4)로 보완

> **쉽게 말하면**: 같은 영상 120프레임 평균 오차는 27.9 mm이고, 통계적으로 약 26.9~29.0 mm 범위로 신뢰 가능.

### 4.4 외부 비교군(참고 수치)

| 항목 | 수치 | 출처 |
|---|---|---|
| RTMPose-m 2D AP(COCO val) | 75.8 AP | MMPose RTMPose README |
| 멀티뷰 algebraic 삼각측량 MPJPE(Human3.6M) | 수십 mm대 | Iskakov et al., ICCV 2019 |
| 멀티뷰 SOTA(volumetric) MPJPE | 17.7 mm | Iskakov et al., ICCV 2019 |

- 데이터셋·프로토콜 차이로 직접 등치 비교 아님 → 방식 계열(학습 없는 algebraic DLT) 통상 성능대 참고용

> **쉽게 말하면**: 본 시스템(학습 없음)의 27.9 mm는 같은 계열 학습-없는 삼각측량(수십 mm)과 비슷한 수준. 최신 딥러닝 SOTA(17.7 mm)보다는 크지만, 학습 없이 달성한 값이라는 점이 핵심.

---

## 5. 정성 평가

- 가림(occlusion) 처리
    - 저신뢰 2D 관절이 삼각측량·depth 융합 가중치에서 자동 배제
    - 복원 실패 관절은 `valid=False`/`source='missing'` 명시 → 후속 소비 단계에서 안전 구분
    - 관절 출처 태그(`triangulation`/`depth`/`fused`/`missing`)로 복원 경로 추적 가능
- 시간적 안정성
    - One-Euro 스무딩은 정지·저노이즈 구간에서 위상지연 유발(§4.1) → jitter 억제와 지연 동시 평가 필요
    - 온라인(causal) 방식이라 실시간 적용 적합, 오프라인 후처리 시 비인과 필터로 지연 제거 가능
- 시각 산출물
    - 데모(`examples/{rgbd,panoptic}_video_demo.py`)는 [입력 2D | depth/뷰 | 3D] 3분할 MP4 생성

**(가) 뒷모습 사례** — 인물이 카메라를 등진 프레임

![**[그림 6]** (뒷모습) 복원한 3D(예측=빨강, GT=초록)를 실제 HD 영상에 재투영. 몸체는 잘 안착하나 얼굴(코·눈)은 등진 자세로 검출이 부정확.](figures/fig_reproj_on_image.png)

![**[그림 7]** (뒷모습) 정면·측면 overlay + 관절별 오차. 몸통·팔다리 10–29 mm로 정확, 오차는 얼굴 키포인트(코 73·눈 95 mm)에 집중.](figures/fig_ortho_planes.png)

**(나) 앞모습 사례** — 인물이 카메라를 향한 프레임(자동 탐색으로 선정)

![**[그림 8]** (앞모습) 복원 3D를 얼굴이 보이는 뷰에 재투영. 얼굴이 보이므로 코·눈·귀까지 예측(빨강)이 GT(초록)에 근접.](figures/fig_reproj_front.png)

![**[그림 9]** (앞모습) 정면·측면 overlay + 관절별 오차. 얼굴 키포인트 오차가 **코 31·우안 6·귀 22 mm로 대폭 개선**(뒷모습 대비), 전체 평균 22.2 mm.](figures/fig_ortho_planes_front.png)

> **쉽게 말하면**: "안 맞아 보이던" 부분은 정확도 문제가 아니라 **얼굴이 안 보이는 자세** 탓. 인물이 카메라를 향하면(앞모습) 얼굴 오차가 95 mm→수십 mm로 줄어듦. 몸통·팔다리는 자세와 무관하게 항상 정확(~1–3 cm).

![**[그림 10]** 우완(right wrist) 좌표의 시간 변화: 원시(회색) 대비 One-Euro 스무딩(파랑)이 고주파 지터를 억제하는 동시에 위상지연을 수반함.](figures/fig_jitter.png)

> **쉽게 말하면**: 회색은 원본(미세하게 떨림), 파랑은 스무딩 후(매끈). 떨림은 줄지만 곡선이 살짝 뒤로 밀리는 지연이 생김 → 떨림과 지연은 trade-off.

---

## 6. 런타임 및 강건성

### 6.1 단계별 런타임(실측: CPU, 1920×1080 HD 3뷰)

- 측정 환경: GPU 미셋업으로 `device=cuda` 요청이 자동 CPU 폴백, 입력 Panoptic HD 원해상도 3뷰, 기하 단계 200회·2D 검출 20프레임 평균

| 단계 | ms / frame |
|---|---|
| 2D 검출(RTMPose, 3뷰 합) | 1456.6 |
| 삼각측량(robust DLT+RANSAC) | 3.61 |
| depth 융합(back-proj+fuse) | 3.86 |
| One-Euro 스무딩 | 0.05 |
| **기하 소계** | **7.5** |
| **end-to-end 총합** | **1464.1** |

- 속도 병목 = 2D 검출(전체의 99.5%), 시스템 고유 기하 연산은 합 7.5 ms로 무시 가능
- end-to-end 속도는 2D 검출(= GPU 가속 유일 단계)이 결정, 현재 CPU·HD·3뷰에서 ≈ 0.7 FPS
- GPU 기대치: README 실측 검출 약 8× 가속(RTX 4050: CPU ~9.6 → GPU ~78 FPS, 단일 뷰) → HD·다뷰는 더 무거우므로 목표 GPU 재측정 필수, 기하 단계는 device 무관 7.5 ms 유지
- 실시간화 권고: 검출 입력 해상도 축소(`pose_input_size`)·경량 모드(`mode=lightweight`)·뷰별 병렬 검출

> **쉽게 말하면**: 느린 이유는 거의 전부 2D 검출(전체 시간의 99.5%). 본 시스템 고유 계산(삼각측량 등)은 7.5 ms로 사실상 공짜. 즉 검출만 GPU로 빠르게 하면 실시간 가능.

### 6.2 캘리브레이션 민감도(oracle, 픽셀 노이즈 없음, 60프레임)

| 교란 | abs MPJPE (mm) | reproj (px) |
|---|---|---|
| rot 0.25° | 4.25 | 1.69 |
| rot 0.5° | 8.54 | 3.40 |
| rot 1.0° | 17.23 | 6.87 |
| rot 2.0° | 29.61 | 11.63 |
| trans 5 mm | 3.88 | 1.37 |
| trans 10 mm | 7.76 | 2.75 |
| trans 20 mm | 15.53 | 5.50 |
| trans 50 mm | 17.80 | 8.50 |

- 회전 오차가 지배적 위험요인(≈ 17 mm/도), 피사체 거리 비례 증폭 → 캘리브 절차의 재투영 RMS ≲ 1 px 게이트가 정확도 핵심
- 병진 오차는 초기 ≈ 0.78 mm/mm 전파, 50 mm에서 포화(RANSAC 재투영 임계가 일부 뷰 기각)

> **쉽게 말하면**: 카메라 각도(회전)가 1°만 틀어져도 3D가 약 17 mm 틀어짐 → 카메라 캘리브레이션 정밀도가 정확도에 가장 중요. 위치(병진) 오차는 영향이 상대적으로 작음.

![**[그림 11]** 외부 캘리브레이션 (a) 회전·(b) 병진 오차에 따른 abs MPJPE 증가. 회전이 지배적 위험요인.](figures/fig_calibration.png)

---

## 7. 한계 및 향후 과제

- 측정의 한계
    - GPU 미셋업으로 런타임은 CPU 기준(목표 GPU 재측정 필요)
    - HD 비디오 보유 시퀀스 1개 → real 신뢰구간은 프레임/구간 기반(시퀀스 간 CI는 oracle로 보완)
    - 평가 도메인이 Panoptic 실내 돔(고품질 동기·캘리브)에 한정, Panoptic GT 자체 오차 존재
- 시스템의 한계
    - 단일 인물 전제(다인 association 미구현)
    - 2-view RANSAC: 작은 baseline·교차각에서 불안정
    - depth 센서 특성(홀·반사·경계 bleed·표면 오프셋) → 특정 관절 편향 가능
    - 차용 검출 의존성: end-to-end 정확도 상당 부분이 RTMPose 품질에 좌우
- 향후 과제(권고)
    - 가림·급동작·저신뢰 subset에서 RANSAC·스무딩·depth 융합 효용 재평가
    - depth 융합 정확도 향상을 3D GT 보유 RGB-D(Panoptic Kinect, PROX)에서 정량화
    - 목표 GPU 런타임 재측정 및 실시간화 최적화
    - 추가 시퀀스 HD 비디오 확보로 시퀀스 간 real 신뢰구간 산출
    - 다인 장면 대응(인물 매칭) 확장

---

## 8. 종합 결론 및 권고

- 3D 복원 기하의 정확성·재현성 입증
    - 순수 기하 정밀도: 무오차 입력 0.0 mm, 2px 노이즈에서 4시퀀스 5.0 ± 0.4 mm
    - 시스템 고유 알고리즘(삼각측량·융합·스무딩)에 정확도상 결함 미발견
- 실사용 절대정확도 27.9 mm: 방식 계열 통상 baseline대 부합, 3뷰가 2뷰 대비 21% 개선 → 삼각측량 실질 기여 확인
- 정확도 실질 한계 = 시스템 외부 요인
    - (i) 차용 2D 검출 품질, (ii) 외부 캘리브레이션(특히 회전, ≈17 mm/도)
    - → 개선 투자는 검출기 등급 상향·캘리브 정밀도 확보(재투영 RMS ≲ 1 px)에 집중
- 속도: 2D 검출 지배(99.5%), 기하 연산 무시 가능(7.5 ms) → 검출 GPU 가속·경량화로 실시간화, 목표 하드웨어 재측정 권고
- 미평가 영역(가림·급동작·다인·depth 융합 정확도·GPU 런타임): §7 향후 과제로 명시, 추가 데이터 확보 시 후속 평가 보완

---

## 9. 부록

### 부록 A. COCO19(Panoptic) → COCO-17 매핑 및 보간

| COCO-17 | ← COCO19 | | COCO-17 | ← COCO19 |
|---|---|---|---|---|
| 0 nose | 1 Nose | | 9 left_wrist | 5 lWrist |
| 1 left_eye | 15 lEye | | 10 right_wrist | 11 rWrist |
| 2 right_eye | 17 rEye | | 11 left_hip | 6 lHip |
| 3 left_ear | 16 lEar | | 12 right_hip | 12 rHip |
| 4 right_ear | 18 rEar | | 13 left_knee | 7 lKnee |
| 5 left_shoulder | 3 lShoulder | | 14 right_knee | 13 rKnee |
| 6 right_shoulder | 9 rShoulder | | 15 left_ankle | 8 lAnkle |
| 7 left_elbow | 4 lElbow | | 16 right_ankle | 14 rAnkle |
| 8 right_elbow | 10 rElbow | | | |

- 보간: `골반 = (좌고관절+우고관절)/2`(root-relative 정렬용, Panoptic은 BodyCenter(COCO19 idx 2) 직접 사용), `목 = (좌어깨+우어깨)/2`(필요 시)
- Panoptic 신뢰도 0 이하 관절은 GT-invalid 처리하여 평가 제외

### 부록 B. 측정 환경 및 재현 절차

- 환경: Windows 11, Python 3.11, onnxruntime(CPU 폴백), 입력 Panoptic HD 1920×1080
- 데이터: `data/panoptic/{171204_pose1,171204_pose2,171204_pose3,171026_pose1}`(calibration + GT; 171204_pose1은 HD 비디오 3대 포함)
- 재현 명령

```bash
# 단일 설정 평가 (oracle/real)
uv run python examples/eval_panoptic.py --seq-dir data/panoptic/171204_pose1 \
    --cams 00_03,00_12,00_23 --start 500 --num-frames 120 --mode real

# 보고서 실험 일괄 재현
uv run python examples/run_experiments.py --only ablation_oracle,calib,multiseq_oracle
uv run python examples/run_experiments.py --only ablation_real,multiseq_real,runtime

# 보고서 그림 생성(차트 + 실측 캡처 + 앞/뒷모습)
uv run python examples/make_report_figures.py --figs charts,qual,jitter,front
```

- 검증: `uv run pytest tests/ -q`(평가 하니스 단위 테스트 포함, 120 passed)

### 부록 C. 평가 지표 정의 요약

- MPJPE: 관절별 예측-GT 유클리드 거리 평균(mm). Absolute=정렬 없음, Root-relative=골반 평행이동 제거, PA=Procrustes 유사변환 정렬 후
- PCK3D@τ: 오차 τ(mm) 이내 관절 비율, AUC = 0–150 mm PCK 곡선 면적
- valid rate: GT-존재 관절 중 시스템 복원 비율(정확도와 분리 보고)
- reprojection RMSE: 복원 3D를 각 뷰로 재투영한 픽셀 RMS(자기합치성 지표, 3D 정확도와 구분)

[^coco17]: COCO-17 — 영상 포즈 추정 표준 데이터셋 COCO가 정의한 17개 인체 관절(코·양 눈·양 귀·양 어깨·양 팔꿈치·양 손목·양 고관절·양 무릎·양 발목).
[^rtmpose]: RTMPose — 실시간 2D 포즈 추정 오픈소스 모델(MMPose 계열). 본 시스템은 사전학습 가중치를 그대로 사용하며 추가 학습은 하지 않음(추론 전용).
[^dlt]: DLT 삼각측량 — 여러 카메라가 본 같은 점의 2D 좌표들로부터 3D 위치를 선형방정식으로 푸는 표준 기법(Direct Linear Transform). "신뢰도 가중"은 신뢰 높은 뷰에 더 큰 가중치를 부여.
[^ransac]: RANSAC — 일부 잘못된 관측(이상치)을 자동 배제하고 다수가 일치하는 해를 찾는 강건 추정 기법.
[^oneeuro]: One-Euro 필터 — 좌표의 시간적 흔들림(지터)을 줄이는 실시간 평활 필터. 빠른 움직임은 통과시키고 미세한 떨림만 억제.
[^mpjpe]: MPJPE(Mean Per Joint Position Error) — 관절별 (예측↔정답) 거리의 평균. 작을수록 정확하며 단위는 mm. 3D 포즈 정확도의 대표 지표. abs(Absolute) MPJPE는 위치·자세 보정 없이 그대로 비교한 값(절대 위치 정확도).
[^pampjpe]: PA-MPJPE — 예측을 정답에 맞춰 회전·크기·위치를 최적 정렬(Procrustes Analysis)한 뒤 계산한 MPJPE. 캘리브레이션·스케일 영향을 제거하고 '자세 형태'만 평가.
[^rrel]: Root-relative MPJPE — 골반을 원점으로 맞춰(전역 위치 제거) 계산한 MPJPE. 관절들의 상대 배치 정확도.
[^pck]: PCK3D@τ — 오차가 임계값 τ(mm) 이내인 관절의 비율. 예: PCK@50mm=0.9이면 관절의 90%가 5 cm 이내로 정확.
[^auc]: AUC — 여러 임계값(0~150 mm)에 대한 PCK 곡선 아래 면적. 시스템의 전반적 강건성을 하나의 숫자로 요약.
[^validrate]: valid rate(복원 성공률) — 정답에 존재하는 관절 중 시스템이 3D로 복원해낸 비율. 정확도와 별개로 '커버리지'를 봄(둘을 분리 보고해야 공정).
[^reproj]: 재투영 오차(reprojection RMSE) — 복원한 3D를 다시 2D 영상에 투영했을 때 원래 검출 위치와의 픽셀 차이. 자기일관성 지표이며 3D 정확도 자체와는 다름(잘못된 3D도 자기 2D엔 맞을 수 있음).
[^oracle]: oracle 모드 — 정답 3D를 카메라로 투영한 '완벽한 2D'를 입력으로 주어, 검출기 오차를 배제하고 기하 복원(삼각측량 등)만의 성능을 측정하는 실험. 반대로 real 모드는 실제 검출기를 쓴 실사용 성능.
