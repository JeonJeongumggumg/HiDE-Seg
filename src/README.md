# 미세조직 4상 최종 추론 패키지

검증 환경: Python 3.11, Windows, CUDA GPU. 코드와 모델만 포함하며 데이터와 GT는 포함하지 않는다.
선택 모델: seed42 HRNet+detail+DINO direct64 5fold. 각fold D4 → IS-P → B2/PCA16 → 확률5개 균등 평균 → argmax.
DINO 가중치는 각 base.pt에 포함된다. pretrained=False로 생성하고 로컬 state_dict만 읽는다. 실행 중 네트워크 연결을 차단한다.

## 실행

패키지를 임의의 폴더에 풀고 Anaconda Prompt에서 해당 폴더로 이동한다.

```bat
conda activate phase-seg
python predict.py --input "D:\data\test.zip" --output "D:\result\phase_seg" --resume
```

입력은 `*/images/*_image.png` 형태의 ZIP. 출력 `masks/*_mask.png`는 입력 크기의 RGB 무손실 PNG다. 확률과 fold별 출력은 재현 점검 및 재개를 위해 같이 저장한다. 정답 마스크를 읽지 않는다. ZIP의 파일명은 입력에서 유도하며 공식 제출 파일명 규칙은 별도 대조해야 한다.
정상 완료시 DONE.json과 status.json의 complete를 확인한다. 중단 시 동일 명령 --resume로 이어간다. 입력·모델·환경이 달라지면 기존 출력을 보존하고 중단한다.

## 환경

requirements-lock.txt는 검증 당시 직접 의존성의 정확한 버전이다. 학습된 sklearn 모델 호환성 때문에 임의 버전 변경은 피한다.
새 환경 설치 예시(설치 시 네트워크 필요):

```bat
conda create -n phase-seg-submit python=3.11 pip -y
conda activate phase-seg-submit
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-lock.txt
```

위 설치 명령으로 새 환경을 실제 생성한 검증과, 기존 환경에서 새 폴더로 옮긴 실행 검증은 구분한다. wheel 파일 자체는 포함하지 않는다. 타GPU/드라이버에서 수치가 달라질 가능성이 있으므로 동일환경 마스크 일치 결과가 모든 환경의 비트일치를 보장하지 않는다.

## 클래스 RGB

- Primary Si: (76,178,76)
- Al3Ni: (25,76,153)
- Eutectic Si: (204,204,204)
- Al: (153,127,76)

## 무결성과 결과 범위

manifest.json은 코드·가중치 SHA256과 버전을 기록한다. base/refiner 체크포인트에서 학습용 optimizer 등은 제거했고 모든 모델 텐서의 완전 일치를 확인했다. PCA/HGB 파일은 원본 그대로 복사했다. 신뢰할 수 있는 패키지 파일만 사용한다.
개발70장 OOF79.9828%, valid20 제출절차78.5597%. test 정답 점수가 아니며 두 평가의 데이터·모델 수가 다르다. 독립적인 공식 성능 추정치로 과장하지 않는다.
