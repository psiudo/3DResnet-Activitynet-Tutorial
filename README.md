# 3D ResNet을 통해 ActivityNet 데이터셋 훈련하기

## 프로젝트 설명
이 프로젝트는 Kenshohara의 3D ResNet을 이용하여 ActivityNet 데이터셋을 학습시키기 위한 과정을 정리한 저널이다. 각 단계별로 필요한 설치, 설정, 수정 사항들을 정리해 두었으며, 데이터셋을 활용하여 비디오를 프레임 이미지로 변환하고 모델 훈련을 진행하는 과정이 포함되어 있다.

## 목차
1. [3D ResNet 및 ActivityNet 레포지토리 클론](docs/step1_3DResnet_Clone.md)
2. [유튜브 비디오 다운로드 준비](docs/step2_Video_Download_Preparation.md)
1. 레포지토리를 클론한 후 `requirements.txt` 파일을 사용해 필요한 라이브러리를 설치한다:
   ```bash
   pip install -r requirements.txt
   ```
3. [유튜브 비디오 다운로드 실행](docs/step3_Youtube_Download_Execution.md)
4. [비디오 다운로드 과정 정리](docs/step4_Video_Download_Summary.md)
5. [비디오를 프레임으로 변환하기](docs/step5_Video_to_Frames.md)

## 프로젝트 구조
- **docs/**: 각 단계별 설명 파일
- **scripts/**: 다운로드 및 변환 스크립트 파일
- **requirements.txt**: 프로젝트에 필요한 라이브러리 목록

## 사용 방법
2. 각 단계별 문서를 참고하여 프로젝트를 설정하고 실행한다.