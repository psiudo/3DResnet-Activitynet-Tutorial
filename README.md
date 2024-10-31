# 3D ResNet을 통해 ActivityNet 데이터셋 훈련하기

## 프로젝트 설명
이 프로젝트는 Kenshohara의 3D ResNet을 이용하여 ActivityNet 데이터셋을 학습시키기 위한 과정을 정리한 저널이다. 
각 단계에서 필요한 설치, 설정, 수정 사항들을 포함하고 있으며, 데이터셋을 활용해 비디오를 프레임 이미지로 변환하고 모델 훈련을 진행하는 방법을 제공한다.
모델 훈련 과정에서 일어난 여러 시행착오와 오해의 과정들을 모두 기록하고 있다.

## 목차
1. 프로젝트 개요
2. 준비 사항
3. 실행 및 설정 가이드

## 프로젝트 구조
- [**docs/First_Training_Attempt.md**](docs/First_Training_Attempt.md): 첫번째 3D ResNet 및 ActivityNet 학습 과정의 전체 단계가 포함된 파일
- **scripts/**: 다운로드 및 변환에 필요한 스크립트 파일
- **requirements.txt**: 프로젝트에 필요한 라이브러리 목록

## 사용 방법
1. 레포지토리를 클론한 후 `requirements.txt` 파일을 통해 필요한 라이브러리를 설치한다:
   ```bash
   pip install -r requirements.txt
