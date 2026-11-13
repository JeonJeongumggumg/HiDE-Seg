# 최종 레시피 학습

이 문서는 **공식 train.zip을 사용할 권한이 있는 사용자**를 위한 순서입니다. 데이터 다운로드 기능은 제공하지 않습니다. 개발 중 사용한 여러 실험 래퍼를 최종 조건에 맞춰 독립 진입점으로 정리했습니다. 공개 준비 과정에서 5-fold 전체를 다시 학습한 것은 아니므로, 아래 재학습이 제출 점수나 체크포인트와 완전히 같아진다는 보장은 없습니다.

## 환경과 분할 확인

명령은 저장소 루트의 Anaconda Prompt에서 실행합니다. `environment.yaml`은 검증했던 **Windows/CUDA** 환경입니다. GPU 메모리 사용량과 속도는 장치에 따라 달라집니다.

```bat
conda env create -f environment.yaml
conda activate phase-seg-submit
python -m pip check
python train.py base --help
python train.py base --train "D:\data\train.zip" --fold 0 --output "runs\base\fold0" --dry-run
python src\prepare_dino_weights.py
```

`--dry-run`은 데이터 ZIP 해시, 70개 ID, A번호 그룹의 fold 분리를 확인합니다. [fold manifest](../resources/folds.json)는 제출 개발 분할 그대로입니다. 파일명의 A번호를 물리적 합금·공정으로 해석하지 않습니다. 다른 데이터셋에 적용하려면 데이터 로더와 분할 검증을 별도로 변경해야 합니다.

DINO 준비 명령은 공개 사전학습 파일 1개를 고정 revision에서 다운로드하고 크기·SHA256을 확인합니다. HRNet ImageNet 초기화는 첫 학습 시 timm에서 다운로드합니다. 이 단계에는 네트워크가 필요합니다. 최종 제출 추론은 모든 가중치를 로컬에서 읽습니다.

## 1. 기저 모델 5개

먼저 fold 0을 실행하고 로그와 GPU 메모리를 확인합니다.

```bat
python train.py base --train "D:\data\train.zip" --fold 0 --output "runs\base\fold0"
```

나머지 fold도 같은 명령의 `--fold`와 출력 폴더만 바꿉니다. Anaconda Prompt에서 순차 실행하려면 다음처럼 씁니다. 배치 파일 안에서는 `%f`를 `%%f`로 바꿉니다.

```bat
for %f in (1 2 3 4) do python train.py base --train "D:\data\train.zip" --fold %f --output "runs\base\fold%f"
```

고정 조건은 seed 42, 60 epochs × 100 steps, batch 4, 원본 384→입력 768, AdamW `1e-4`, CE+Dice입니다. 5 epoch마다 단일 방향·균등 겹침 평균 검증으로 `best.pt`를 선택합니다. 최종 D4/Gaussian 점수로 체크포인트를 다시 고르지 않습니다. 자세한 구조와 증강은 [MODEL.md](MODEL.md)에 있습니다.

## 2. fold-local 기저 확률 캐시

각 fold의 모델로 train 56장과 held-out 14장에 D4 추론을 수행합니다. 이 중 **자기 학습 56장의 in-sample 예측**이 IS-P 학습 입력입니다.

```bat
for %f in (0 1 2 3 4) do python train.py cache --train "D:\data\train.zip" --fold %f --base "runs\base\fold%f\best.pt" --output "cache\base\fold%f"
```

확률 배열은 원본 좌표의 float32 `(4,H,W)`입니다. 캐시 manifest에 데이터·분할·기저 모델·확률 파일의 SHA256을 기록합니다. 디스크 여유 공간을 확보해 주세요. 데이터 캐시를 GitHub에 올리지 마세요.

## 3. IS-P 공간 보정

```bat
for %f in (0 1 2 3 4) do python train.py refiner --train "D:\data\train.zip" --fold %f --cache "cache\base\fold%f" --output "runs\refiner\fold%f"
```

seed 731, 10 epochs × 100 steps, batch 4, 확률 입력만 사용합니다. `last.pt`가 고정된 마지막 epoch의 모델입니다. RGB 입력은 0이며 기저 모델은 갱신하지 않습니다. 검증 fold의 정답은 이 학습에 들어가지 않습니다. in-sample 확률과 미관측 영상 확률의 차이는 [평가 한계](EVALUATION.md)에 설명했습니다.

## 4. IS-P 보정 OOF 생성

```bat
for %f in (0 1 2 3 4) do python train.py cache --train "D:\data\train.zip" --fold %f --base "runs\base\fold%f\best.pt" --refiner "runs\refiner\fold%f\last.pt" --output "cache\is_p_oof\fold%f"
```

`--refiner`가 있으면 해당 fold의 held-out 14장만 저장합니다. 이 단계는 기존 확률을 다시 사용하지 않고 기저 D4부터 계산합니다.

## 5. Si 성분 재판별기

```bat
python train.py reclassifier --train "D:\data\train.zip" --cache-root "cache\is_p_oof" --base-root "runs\base" --output "runs\reclassifier"
```

새 IS-P OOF에서 Si 성분을 다시 만들고, DINO 특징을 성분별로 평균합니다. 각 outer fold에 대해 다른 56장의 유효 성분으로 PCA16과 HistGradientBoosting을 학습합니다. 조건은 B2 확률 요약 6개 + PCA16, 재분배 강도 1, 성분 표본 가중치 1입니다. 형태·주변 특징 B3는 사용하지 않습니다.

이 단계의 OOF 재분할에는 다른 fold의 기저 모델을 통한 간접 의존성이 있습니다. 엄밀한 nested CV로 부르지 않습니다. [평가 프로토콜](EVALUATION.md)

## 6. 재학습 가중치로 새 추론 패키지 만들기

```bat
python scripts\package_trained_models.py --base-root "runs\base" --refiner-root "runs\refiner" --reclassifier-root "runs\reclassifier" --output "outputs\retrained_package"
python outputs\retrained_package\predict.py --input "D:\data\valid.zip" --output "outputs\retrained_valid" --resume
python evaluate.py --labels "D:\data\valid.zip" --predictions "outputs\retrained_valid\masks" --output "outputs\retrained_valid_scores.json"
```

새 패키지에는 **새 가중치의 해시와 현재 패키지 버전**을 기록합니다. 제출 가중치용 `src/manifest.json`을 새 가중치에 억지로 맞추지 않습니다. 저장소 공개 범위는 여전히 코드·문서이며, 생성한 모델과 데이터는 `.gitignore` 대상입니다.

## 중단·재시작

공개 학습·캐시 진입점은 비어 있지 않은 출력 폴더를 보호하며 자동 resume를 지원하지 않습니다. 중단된 학습은 새 출력 폴더에서 시작합니다. 전체 실행을 백그라운드로 넘기기 전에 fold 0과 저장 공간을 확인하세요. 최종 `predict.py`의 `--resume`만 동일 입력·manifest의 완료 파일을 검증하여 이어갑니다.
