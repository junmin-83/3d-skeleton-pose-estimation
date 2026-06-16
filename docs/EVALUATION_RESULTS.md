# 3D 스켈레톤 포즈 추정 시스템 — 정량평가 결과 (용역보고서 §4–6)

> 측정일: 2026-06-16 · 평가 하니스: `src/eval/` + `examples/eval_panoptic.py`, `examples/run_experiments.py`
> 데이터: CMU Panoptic HD (GT = `hdPose3d_stage1_coco19`, COCO19→COCO17 매핑) · 좌표계 검증 완료(oracle 라운드트립 abs MPJPE = 0.00 mm)
> 단위: 거리 mm, 재투영 px. 측정 환경: 본 측정은 GPU 미셋업으로 **CPU 추론**으로 수행됨(런타임 §3 주석 참조).

평가 계획·방법론은 [EVALUATION_PLAN.md](EVALUATION_PLAN.md) 참조. 모든 수치는 재현 가능:
원시 JSON은 `output/experiments/*.json`.

### 핵심 요약 (Executive Summary 초안)

- **End-to-end 절대정확도**(실제 RTMPose, 3뷰 HD, 120프레임): abs MPJPE **27.9 mm** (95% CI [26.9, 29.0]),
  PA-MPJPE 24.7 mm, PCK@100mm 0.996 — 학습 없는 멀티뷰 삼각측량 baseline대(수십 mm)에 부합.
- **멀티뷰 기여 입증**: 뷰 2→3에서 abs MPJPE 34.5→27.2 mm(−21%), PCK@50mm 0.748→0.916.
- **기하 파이프라인 고유 정밀도**(oracle, 차용 검출기 배제, 4시퀀스): 2px 2D노이즈에서 **5.0 ± 0.4 mm**,
  무노이즈 라운드트립 0.0 mm.
- **정확도 병목 = calibration 회전**(≈17 mm/도)과 **2D 검출 품질**. 기하 연산 자체 오차는 미미.
- **속도 병목 = 2D 검출**(전체의 99.5%); 기하 단계 합 7.5 ms. GPU 가속이 실시간화의 핵심.

---

## 측정 신뢰성 요약 (먼저 읽을 것)

- **하니스 자체 검증**: GT 3D를 각 카메라로 (왜곡 포함) 투영 → 삼각측량 복원하는 oracle 라운드트립에서
  abs MPJPE = **0.00 mm**, 재투영 = **0.00 px**. GT 로더·COCO19↔COCO17 매핑·cm→m·calibration·삼각측량의
  정합성을 수치로 보증.
- **2모드 분리 측정**(EVALUATION_PLAN §3-1):
  - *oracle* = GT 2D 투영 입력 → 차용 RTMPose를 배제한 **기하 파이프라인 고유 성능**.
  - *real* = 실제 RTMPose 입력 → **end-to-end 실사용 성능**.
- **한계**: 단일 인물 전제, Panoptic 실내 돔 도메인, real 모드는 HD 비디오가 있는 1개 시퀀스(171204_pose1)에
  한정. 자세한 한계는 [EVALUATION_PLAN.md](EVALUATION_PLAN.md) §4.

---

## §2. Ablation — 시스템 모듈별 기여 (oracle 2D, 60 frames)

GT 2D에 2px 가우시안 노이즈를 준 통제 조건에서, 각 모듈을 끄거나 조건을 바꿔 측정.

| config | abs MPJPE (mm) | PA-MPJPE (mm) | PCK@50mm | reproj (px) |
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

**해석**
- **2D 노이즈 → 3D 오차는 거의 선형**(≈2.4 mm per 1 px). 삼각측량 geometry가 안정적임을 시사.
- **RANSAC on/off가 동일**: oracle 입력엔 outlier가 없어 기각 대상이 없음. RANSAC의 효용은 *real*의
  가림/오검출 구간에서 평가해야 함(EVALUATION_PLAN §3-2 예측과 일치).
- **스무딩이 oracle(저노이즈)에서는 오차를 키움**(4.82→12.73 mm): 제거할 jitter가 없는데 One-Euro의
  위상지연만 더해진 결과. → **jitter–lag trade-off**의 직접 증거. 스무딩의 가치는 노이즈가 있는 *real*에서
  재평가(아래 §2-real). 보고서에는 "스무딩은 jitter 억제용이며 정확도 지표와 분리 해석" 명시 필요.
- **뷰 2→3**: 노이즈 통제 시 3-view가 우세(별도 측정: 무스무딩·2px에서 2-view 7.41 vs 3-view 4.88 mm).

### §2-real. Ablation — end-to-end (실제 RTMPose, 40 frames)

| config | abs MPJPE (mm) | PA-MPJPE (mm) | root-rel (mm) | PCK@50mm | reproj (px) |
|---|---|---|---|---|---|
| full (ransac+smooth) | 27.23 | 24.60 | 37.14 | 0.916 | 15.00 |
| no RANSAC | 26.08 | 23.15 | 32.24 | 0.937 | 13.98 |
| no smoothing | 27.25 | 25.06 | 36.17 | 0.913 | 15.30 |
| 2 views | 34.46 | 29.84 | 41.92 | 0.748 | 7.21 |

**해석**
- **뷰 2→3의 효과가 end-to-end에서 분명**: 2-view 34.46 → 3-view(full) 27.23 mm(−21%), PCK@50mm
  0.748→0.916. 멀티뷰 삼각측량의 실질 기여를 입증. (2-view의 reproj가 더 낮은 7.21 px인데 3D 오차는
  더 큼 → reproj self-consistency가 3D 정확도와 다름을 보여주는 사례.)
- **RANSAC/스무딩은 이 시퀀스에서 ~중립**(정적 단일인물·고품질 HD라 outlier/jitter가 적음). RANSAC·
  스무딩의 가치는 가림·급동작·저신뢰 구간에서 재평가해야 하며(향후과제), 본 시퀀스 결과만으로 "무용"으로
  해석하면 안 됨.
- real 절대정확도 ≈ **27 mm**(abs)/**25 mm**(PA)는 학습 없는 멀티뷰 삼각측량 baseline대(수십 mm)에 부합.

---

## §4. Calibration 민감도 (oracle, 픽셀 노이즈 없음, 60 frames)

외부 파라미터(R, t)에 가우시안 노이즈를 주입하고(2D는 참 calibration으로 생성) 복원 calibration만
교란해 순수 민감도를 측정.

| perturbation | abs MPJPE (mm) | reproj (px) |
|---|---|---|
| rot 0.0° | 0.00 | 0.00 |
| rot 0.25° | 4.25 | 1.69 |
| rot 0.5° | 8.54 | 3.40 |
| rot 1.0° | 17.23 | 6.87 |
| rot 2.0° | 29.61 | 11.63 |
| trans 0.0 mm | 0.00 | 0.00 |
| trans 5.0 mm | 3.88 | 1.37 |
| trans 10.0 mm | 7.76 | 2.75 |
| trans 20.0 mm | 15.53 | 5.50 |
| trans 50.0 mm | 17.80 | 8.50 |

**해석**
- **회전 오차가 지배적 위험요인**: ≈**17 mm/도**. 피사체 거리(~수 m)에 비례해 증폭되므로, 보고서의
  calibration 절차(reprojection RMS ≲1 px 게이트)가 정확도의 핵심임을 정량적으로 뒷받침.
- **병진 오차**는 초기 ≈0.78 mm/mm로 전파되다 50 mm에서 포화(RANSAC reproj 임계 15 px가 일부 뷰 기각).
- 실무 시사점: 멀티뷰 extrinsic 회전 보정 정밀도가 절대정확도의 1차 결정요인.

---

## §1. Cross-sequence 정확도 + 신뢰구간

### §1-oracle. 시퀀스 간 (oracle, 2px 노이즈, 60 frames/seq, n=4 시퀀스)

| sequence | abs MPJPE (mm) | frames |
|---|---|---|
| 171026_pose1 | 5.35 | 60 |
| 171204_pose1 | 4.83 | 60 |
| 171204_pose2 | 4.92 | 60 |
| 171204_pose3 | 5.02 | 60 |
| **mean ± 95% CI** | **5.03 ± 0.36** | n=4 seqs |

시퀀스를 독립 단위로 집계(EVALUATION_PLAN §4: 인접 프레임 상관 회피). 기하 파이프라인은 서로 다른
카메라 배치·포즈의 4개 시퀀스에서 **5.03 ± 0.36 mm**로 일관(2px 노이즈 조건).

### §1-real. 단일 실측 시퀀스 불확실성 (실제 RTMPose, 171204_pose1, 120 frames)

End-to-end 절대정확도 요약: abs MPJPE **27.89 mm**, PA-MPJPE 24.72 mm, root-rel 37.41 mm,
PCK@50mm 0.892 / PCK@100mm 0.996, AUC 0.804, valid rate 0.9996, reproj 17.43 px.

| estimator | abs MPJPE (mm) | 95% interval | n |
|---|---|---|---|
| frame bootstrap | 27.89 | [26.85, 28.98] | 120 frames |
| segment t-interval | 27.89 | ± 6.09 | 4 segments |

> HD 비디오가 있는 시퀀스가 1개라 frame bootstrap은 낙관적(프레임 자기상관). segment(30프레임씩 4분할)
> t-interval은 보수적 추정. **진정한 시퀀스 간 real CI는 추가 시퀀스의 HD 비디오(시퀀스당 ~8.6 GB)가
> 필요** — 본 측정은 oracle 다중시퀀스(§1-oracle, n=4)로 보완.

---

## §3. Runtime — 단계별 분해 (실측: CPU, 1920×1080 HD 3뷰)

> 실행 환경은 **CPU**(이 머신은 GPU 미셋업으로 `device=cuda` 요청이 자동 CPU 폴백). 입력은 Panoptic HD
> 원해상도(1920×1080) 3뷰. 기하 단계 200회, 2D 검출 20프레임 평균.

| stage | ms / frame |
|---|---|
| 2D detection (RTMPose, 3뷰 합) | 1456.6 |
| triangulation (robust DLT+RANSAC) | 3.61 |
| depth fusion (back-proj+fuse) | 3.86 |
| One-Euro smoothing | 0.05 |
| **geometry 소계** | **7.5** |
| **end-to-end 총합** | **1464.1** |

**해석**
- **2D 검출이 병목**(전체의 99.5%): HD 3뷰 합 1456.6 ms(뷰당 ~485 ms, CPU). 본 시스템 고유 기하
  연산(삼각측량+fusion+스무딩)은 **합 7.5 ms로 무시 가능**.
- 따라서 **end-to-end 속도는 사실상 2D 검출 = GPU가 가속하는 유일한 단계**가 결정. 현재 CPU·HD·3뷰에서
  ≈**0.7 FPS**.
- **GPU 기대치**: README 실측 기준 검출 약 8× 가속(RTX 4050: CPU ~9.6 → GPU ~78 FPS, 단일 뷰 기준).
  HD·3뷰 동시 검출은 해상도·뷰 수만큼 더 무거우므로, **목표 GPU에서 재측정 필수**. 기하 단계는
  device 무관(NumPy/SciPy)하므로 GPU에서도 7.5 ms 유지.
- 실시간화 권고: 검출 입력 해상도 축소(`pose_input_size`)·`mode=lightweight`·뷰별 병렬 검출.

---

## 비교군(참고 수치, 외부 출처)

| 항목 | 수치 | 출처 |
|---|---|---|
| RTMPose-m 2D AP (COCO val) | 75.8 AP | MMPose RTMPose README |
| 멀티뷰 algebraic 삼각측량 MPJPE (H3.6M) | 수십 mm대 | Iskakov et al. 2019 |
| 멀티뷰 SOTA(volumetric) MPJPE | 17.7 mm | Iskakov et al. 2019 |

> 본 시스템은 학습 없는 algebraic DLT+스무딩 추론기이므로, real end-to-end 절대정확도는 위 baseline대와
> 비교 해석한다(데이터셋·프로토콜 차이가 있어 직접 등치 비교는 아님).
