# 데이터·환경·가중치

## 공개 범위

대회 제공 train70 / valid20 / test10의 원본·GT·크롭·예측 마스크·확률맵은 포함하지 않습니다. 공개한 그림은 코드 구조를 설명하는 SVG이며, 현미경 영상이 들어 있지 않습니다. 데이터 접근 권한과 이용 조건은 주최 측 안내를 따릅니다.

학습·평가에 사용하는 ZIP 구조:

```text
train.zip
  train/images/*_image.png
  train/masks/*_mask.png
valid.zip
  valid/images/*_image.png
  valid/masks/*_mask.png
test.zip
  test/images/*_image.png
```

팔레트 순서는 Primary Si `(76,178,76)`, Al₃Ni `(25,76,153)`, ESi `(204,204,204)`, Al `(153,127,76)`입니다. 원본은 RGB 파일로 읽고, 클래스 라벨은 정확한 RGB 매핑으로 복원합니다.

## 환경

`environment.yaml`은 실제 설치·실행을 확인한 Windows Conda 환경입니다. `requirements.txt`는 주요 직접 의존성의 고정 버전입니다. 학습된 scikit-learn 모델 호환성을 위해 버전을 유지합니다. Linux/다른 GPU로의 이식 결과는 별도로 검증해야 합니다.

## 가중치

현재 공개판은 코드·문서만 제공합니다. `resources/checkpoints.json`은 제출 가중치의 해시 메타데이터이며 가중치 자체가 아닙니다. 접근 가능한 공개 다운로드 링크는 현재 제공하지 않습니다.

제출 가중치를 별도로 보유한 경우 `src/models/foldN/{base.pt,refiner.pt,reclassifier.joblib}`에 배치합니다. DINO 가중치는 base.pt 내부에 포함되어 있어 제출 추론은 모델을 다운로드하지 않습니다.

새 학습에는 DINO 사전학습 파일을 아래 명령으로 준비합니다. 고정 revision과 SHA256을 검사합니다.

```bat
python src/prepare_dino_weights.py
```

HRNet ImageNet 가중치는 기저 학습의 `pretrained=True` 초기화 때 timm으로 내려받을 수 있습니다. 이 동작은 학습 준비 단계이며 오프라인 제출 추론과 구분합니다.

## 파일 변경과 해시

`src/manifest.json`은 제출 시점의 코드와 모델을 고정합니다. 해당 추론 파일의 줄바꿈까지 보존하도록 `.gitattributes`를 설정했습니다. 학습을 다시 한 checkpoint는 제출 가중치 해시와 달라지는 것이 정상입니다. 새 가중치는 `scripts/package_trained_models.py`로 새 패키지를 만들어 사용합니다.

출력 디렉터리가 이미 있으면 덮어쓰지 않습니다. 제출 추론의 `--resume`는 입력·버전·패키지 contract가 일치할 때만 재개합니다. 공개용 학습 스크립트는 새 출력 경로를 요구하며 중간 학습 재개 기능을 제공하지 않습니다.
