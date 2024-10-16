torchvision.transforms에서 더 이상 Scale이라는 함수가 지원되지 않는다.
따라서, spatial_transforms.py 파일에서 transforms.Scale을 transforms.Resize로 변경했다.
