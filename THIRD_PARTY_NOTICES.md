# 외부 연구·코드·모델 출처

HiDE-Seg 자체 결합 코드와 문서는 [MIT](LICENSE)입니다. 아래 라이브러리는 설치 의존성이며, 이 저장소에 그 구현 전체나 가중치를 재배포하지 않습니다. 각 프로젝트의 원래 라이선스와 고지는 그대로 적용됩니다.

| 구성 | 사용 방식 | 원 출처 / 라이선스 |
|---|---|---|
| HRNet | timm HRNet-W18의 stage와 분기 활용 | [공식 연구 코드](https://github.com/HRNet/HRNet-Semantic-Segmentation), [공식 고지](https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/LICENSE) |
| DINOv2 | timm ViT-S/14 동결 자기지도 특징 | [공식 연구 코드](https://github.com/facebookresearch/dinov2), [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| timm | HRNet·DINO 모델 생성과 ImageNet 사전학습 | [코드 / Apache-2.0](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE) |
| segmentation-models-pytorch | 보존한 모델 빌더의 기반 의존성 | [코드 / MIT](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE) |
| PyTorch / torchvision | 신경망·자동미분·GPU 연산 | [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE) |
| scikit-learn / joblib | PCA, HistGradientBoosting, 모델 직렬화 | [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING), [joblib](https://github.com/joblib/joblib/blob/main/LICENSE.txt) |
| NumPy / SciPy / scikit-image | 배열, 연결 성분, 거리·형태 특징 | [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt), [scikit-image](https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt) |
| Pillow / safetensors / threadpoolctl | 이미지 입출력, 안전한 텐서 로딩, 스레드 관리 | [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE), [safetensors](https://github.com/huggingface/safetensors/blob/main/LICENSE), [threadpoolctl](https://github.com/joblib/threadpoolctl/blob/master/LICENSE) |

HRNet 모델 구현은 timm에 포함된 구현을 호출합니다. HiDE-Seg의 detail 교환, DINO 투영·정렬, 확률 보정과 성분 재판별은 이를 조합한 프로젝트 코드입니다. 기존 모델 자체를 새로 제안한 것으로 표기하지 않습니다.

사전학습 DINO 파일의 출처는 [timm/vit_small_patch14_dinov2.lvd142m](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)입니다. 준비 스크립트는 revision `4610ca143709d58a633b6397a74412c2c3842454`와 SHA256을 고정합니다. 모델 카드의 사용 조건을 확인하고 사용하세요. 본 저장소의 MIT 라이선스가 원 데이터셋·사전학습 가중치의 이용 권한을 대신하지 않습니다.

대회 데이터셋·정답·예측 마스크·학습 가중치·제출 문서는 공개하지 않습니다. `resources`에는 분할 재현용 ID와 집계 수치, 해시만 제공합니다.
