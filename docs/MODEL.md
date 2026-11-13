# HiDE-Seg 구조와 최종 설정

## 기저 분할 모델

| 항목 | 최종 설정 |
|---|---|
| HRNet | `timm`의 `hrnet_w18.ms_in1k`, ImageNet 초기화 |
| 고해상도 경로 | 3×3 Conv·BN·ReLU 두 층, 16채널 |
| 특징 교환 | HRNet stage 2·3 이후 1/4 분기 ↔ 입력 해상도 detail |
| 교환 방식 | detail→context adaptive average pooling, context→detail bilinear 보간, 학습 scalar 잔차 |
| DINO | `vit_small_patch14_dinov2.lvd142m`, 마지막 블록, 동결/eval |
| stage 3 연결 | 384→18 투영, 0 초기화 후 덧셈 |
| direct 연결 | 384→64 Conv·GELU, 좌표 정렬 후 최종 head에 concat |
| direct 초기화 | 기존 head 입력 가중치는 복사, 새 64채널 입력 가중치만 0 |
| 최종 head | 112→32, 3×3 Conv·BN·ReLU; 32→4, 1×1 Conv |
| 학습 가능 파라미터 | 9,643,750; 동결 DINO 파라미터 제외 |

구현: [HRNet/detail](../src/hrnet_phase.py), [DINO 결합](../src/hrnet_dino_phase.py).
교환은 두 방향에서 갱신 전 특징을 사용합니다. DINO 입력은 우·하단 반사 패딩으로 14배수에 맞추며, 특징 정렬은 원래 픽셀 중심 좌표를 사용합니다. 768 입력은 770으로 패딩됩니다.

## 기저 모델 학습

- train70 내부 A번호 그룹 5-fold, 각 fold 학습 56장 / 검증 14장.
- seed42, 60 epochs × 100 updates × batch4 = 24,000 크롭 / fold.
- 먼저 원본 이미지를 균등 선택하고 원본 384 크롭 위치를 무작위 선택합니다.
- 768 입력으로 확대: 영상 bilinear, GT nearest.
- D4 공간 증강, gain 0.9~1.1 및 offset −8~8의 밝기 증강.
- ImageNet mean/std 정규화, AdamW lr=1e-4, weight decay=1e-4, 고정 학습률.
- loss = CE + (1 − batch·공간 합산 soft Dice 평균). 클래스·경계 가중치 없음.
- AMP 사용. 매 5 epoch 단일 방향·균등 겹침 검증으로 최고 checkpoint 선택.
- 최종 D4/Gaussian 평가와 체크포인트 선택 시의 추론 조건은 구분합니다.

## D4 및 원본 좌표 복원

원본 타일 384, stride192. 768 정사각형 입력을 8개 회전·반전으로 예측하고, softmax 확률을 역변환해 평균합니다. 2×2 평균 풀링으로 원본 해상도로 돌아온 후 Gaussian 겹침 평균을 수행합니다. 이미지 전체 회전 TTA가 아닙니다.

## IS-P 공간 보정

입력 채널은 RGB3 + 확률4이지만 최종 IS-P는 RGB를 0으로 둡니다. 채널16/32/64의 작은 encoder-decoder가 확률의 log값에 잔차를 더합니다. GroupNorm과 GELU를 사용하며, 마지막 잔차 Conv는 0 초기화합니다.

seed731, 10 epochs, epoch당100 updates, batch4, AdamW lr=.001, weight decay=.0001, gradient clipping5, GradScaler initial scale128. 공간 D4 증강만 사용하고 마지막 epoch10을 선택합니다. 해당 fold 기저 모델이 자기 학습56장을 예측한 in-sample 확률로 학습합니다. 추론은 원본384/stride192/Gaussian이며 보정망 자체의 추가 TTA는 없습니다.

## B2 + PCA16 Si 성분 재판별

IS-P 결과의 argmax에서 PSi·ESi 합집합을 8연결 성분으로 구성합니다.

- B2: 성분 평균 4상 확률, Si 내부 PSi 비율의 평균·표준편차 → 6개.
- DINO: 원본384→768, stride192, 마지막 블록 특징을 Gaussian 중첩·성분 평균으로 모음 → 384개. 특징 추출에는 별도 D4 없음.
- 학습 성분으로만 PCA16을 적합한 후 B2와 결합 → 22개 특징.
- HistGradientBoostingClassifier: learning_rate=.05, max_iter100, max_leaf_nodes7, max_depth3, min_samples_leaf20, L2=1, early_stopping=False, seed42.
- GT Si 픽셀의 다수결로 PSi/ESi 목표를 정하며, 동수·정답 Si 없음 성분은 적합에서 제외합니다.
- 성분별 PSi 확률 q로 PSi/ESi 확률을 재분배합니다. Si 확률 합과 다른 두 상 확률은 보존합니다.

형태·주변 조성 특징을 명시적으로 추가한 B3는 최종 채택 조건이 아닙니다. 코드의 `extract`는 연구용 전체 특징을 계산하지만 최종 분류기는 앞 6개와 PCA16만 사용합니다.
