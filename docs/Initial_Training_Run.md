# Step 1: 3D ResNet 및 ActivityNet 레포지토리 클론

- Kenshohara의 GitHub를 통해 3D ResNet을 로컬로 클론함.
- ActivityNet 데이터셋이 필요하여 ActivityNet GitHub 레포지토리도 클론하려 함.
- PyCharm 터미널에서 `git clone` 명령어를 통해 클론할 수 있다는 사실을 알게 되었음.
- 그러나 PyCharm 터미널이 PowerShell 기반이라 명령어가 제대로 작동하지 않아 Git Bash로 터미널을 변경함.

# Step 2: 유튜브 비디오 다운로드 준비

- ActivityNet을 이용해 유튜브 비디오를 다운로드하려고 함.
- 필요한 파일은 `ActivityNet/Crawler` 폴더 안의 `fetch_activitynet_videos.sh` 스크립트였음.
- 해당 스크립트에 실행 권한을 부여하기 위해 다음 명령어를 사용함.

  ```bash
  chmod +x fetch_activitynet_videos.sh
  ```

# Step 3: 유튜브 비디오 다운로드 실행

- PyCharm 터미널에서 `fetch_activitynet_videos.sh`를 실행하려 했으나 JSON 파일을 찾을 수 없다는 문제가 발생함.
- ReadMe 파일에 나왔던 `activity_net.v1-X.json` 파일이 실제로는 `v1-3` 버전으로 이름이 변경된 상태였음.
- 인자 이름을 변경하여 다시 시도했으나, `youtube-dl`이 오래된 버전이라 유튜브의 최신 구조를 지원하지 않음.
- 이 문제를 해결하기 위해 `yt-dlp`로 대체하고 `run_crosscheck.py` 코드도 `yt-dlp`를 사용할 수 있도록 수정함.
- 약 16,000개의 비디오 다운로드를 완료함.

# Step 4: 비디오 다운로드 과정 정리

- 비디오를 다운로드하기 위해 `fetch_activitynet_videos.sh` 스크립트를 실행하였음.
- `run_crosscheck.py`를 사용해 `activity_net.v1-X.json` 파일을 읽고, 이미 다운로드된 비디오와 다운로드되지 않은 비디오를 교차 검토함.
- 필요한 비디오만 `command_list.txt` 파일에 작성하여 다운로드하도록 수정함.

# Step 5: 비디오를 프레임으로 변환하기

- `generate_video_jpgs.py` 스크립트를 사용하여 비디오를 프레임 이미지로 변환해야 함.
- 이를 위해 **FFmpeg**가 필요했으며, 공식 웹사이트에서 FFmpeg 압축 파일을 다운로드하여 C 드라이브에 설치함.
- 환경 변수를 추가하기 위해 **Path** 설정을 수정하였으며, 초기에는 터미널을 재시작하지 않아 인식되지 않았지만, 터미널을 재시작한 후 제대로 인식됨.
- `generate_video_jpgs.py` 스크립트에 인자를 넣어 실행하였으며, 정상적으로 작동 중임.
- 이 과정에서 `generate_video_jpgs.py`의 `def video_process` 함수를 절대 경로로 수정하였으나, 다른 컴퓨터에서 경로가 변경될 경우 문제가 발생할 수 있으므로 다시 상대 경로로 원상 복구함.

# Step 6: `main.py` 실행 후 `Scale` 함수 오류 발생

- `torchvision.transforms`에서 더 이상 `Scale` 함수가 지원되지 않음.
- 따라서, `spatial_transforms.py` 파일에서 `transforms.Scale`을 `transforms.Resize`로 변경함.

# Step 7: `fps` 관련 오류 발생

- 다음 명령어를 실행했더니 `fps` 관련 에러가 발생함:

  ```bash
  python D:/Tutorial/main.py --root_path D:/Tutorial --video_path D:/ActivityNetFrames \
  --annotation_path D:/Tutorial/ActivityNet/Evaluation/data/activity_net.v1-3.min.json \
  --result_path D:/Tutorial/results --dataset activitynet --n_classes 400 \
  --n_epochs 50 --batch_size 8
  ```

- 이를 해결하기 위해 `activitynet.py` 파일의 `get_video_ids_annotations_and_fps` 함수에서 `fps` 값을 기본값으로 설정하는 코드를 추가함.

  ```python
  def get_video_ids_annotations_and_fps(data, subset):
      video_ids = []
      annotations = []
      fps_values = []

      for key, value in data['database'].items():
          this_subset = value['subset']
          if this_subset == subset:
              video_ids.append(key)
              annotations.append(value['annotations'])
              fps_values.append(value.get('fps', 30))  # 기본값 30 설정

      return video_ids, annotations, fps_values
  ```

# Step 8: `TypeError` 오류 해결

- `video_path_formatter` 람다 함수에서 `label` 변수가 정의되지 않아 `TypeError`가 발생함.
- 이를 해결하기 위해 아래 코드를 `activitynet.py`의 `__make_dataset` 함수에 추가함.

  ```python
  if annotations[i]:
      label = annotations[i][0]['label']  # 첫 번째 레이블 사용
  else:
      label = 'unknown'  # 레이블이 없을 경우 기본값 사용
  ```

# Step 9: 비디오 로딩 오류 해결

- `videodataset.py`의 `__loading` 함수에서 클립이 `None`이거나 비어 있을 경우 명시적으로 오류를 발생시키고 메시지를 출력하도록 수정함.

  ```python
  def __loading(self, path, frame_indices):
      # 클립 로딩 시도
      clip = self.loader(path, frame_indices)

      # 디버깅: 클립이 None이거나 비었는지 확인
      if clip is None:
          print(f"[DEBUG] Clip is None for path: {path}, frame indices: {frame_indices}")
          return None
      if len(clip) == 0:
          print(f"[DEBUG] Clip is empty for path: {path}, frame indices: {frame_indices}")
          return None

      # 디버깅: 로드된 클립의 길이를 확인
      print(f"[DEBUG] Loaded clip length: {len(clip)} for path: {path}")

      # 공간 변환 적용
      if self.spatial_transform is not None:
          print(f"[DEBUG] Applying spatial transformation for path: {path}")
          self.spatial_transform.randomize_parameters()
          transformed_clip = []
          for img in clip:
              transformed_img = self.spatial_transform(img)
              print(f"[DEBUG] Transformed image shape: {transformed_img.shape}")
              transformed_clip.append(transformed_img)
          clip = transformed_clip

      # 텐서로 변환
      if len(clip) > 0:
          clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
          print(f"[DEBUG] Final tensor shape: {clip.shape}")
      else:
          print(f"[ERROR] Clip has no valid images after transformation for path: {path}")
          return None

      return clip
  ```

- `__getitem__` 함수에서 `None`인 클립을 건너뛰도록 변경함.

  ```python
  if clip is None:
      print(f"Skipping index {index} due to loading error.")
      return self.__getitem__((index + 1) % len(self))
  ```

- `main.py`에서 `val_loader`, `val_logger` 부분을 수정하고 `get_validation_data()` 함수 정의 부분에 `video_path_formatter=None`를 추가함.

- `training.py`의 `train_epoch()` 메서드에서 데이터 로딩 중 오류가 발생하면 건너뛰거나 다시 시도하는 로직을 추가하여 학습 과정이 중단되지 않도록 함.

- `main.py`의 `get_opt()` 함수와 학습 과정 내 경로 설정 부분에 디버그 출력을 추가함.

  ```python
  print(f"Video path set to: {opt.video_path}")
  print(f"Annotation path set to: {opt.annotation_path}")
  ```

- 아래 명령어로 실행함.

  ```bash
  python D:/Tutorial/main.py --root_path D:/Tutorial --video_path D:/ActivityNetFrames \
  --annotation_path D:/Tutorial/ActivityNet/Evaluation/data/activity_net.v1-3.min.json \
  --result_path D:/Tutorial/results --dataset activitynet --n_classes 200 \
  --n_epochs 50 --batch_size 8
  ```

# Step 10: 데이터 전처리 함수 정의

- `main.py`에 `get_spatial_transform` 함수를 추가함.

  ```python
  def get_spatial_transform(opt):
      spatial_transform = []
      if opt.train_crop == 'random':
          spatial_transform.append(
              RandomResizedCrop(opt.sample_size, (opt.train_crop_min_scale, 1.0),
                                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
      elif opt.train_crop == 'corner':
          scales = [1.0]
          scale_step = 1 / (2**(1 / 4))
          for _ in range(1, 5):
              scales.append(scales[-1] * scale_step)
          spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
      elif opt.train_crop == 'center':
          spatial_transform.append(Resize(opt.sample_size))
          spatial_transform.append(CenterCrop(opt.sample_size))
      normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm, opt.no_std_norm)
      if not opt.no_hflip:
          spatial_transform.append(RandomHorizontalFlip())
      if opt.colorjitter:
          spatial_transform.append(ColorJitter())
      spatial_transform.append(ToTensor())
      if opt.input_type == 'flow':
          spatial_transform.append(PickFirstChannels(n=2))
      spatial_transform.append(ScaleValue(opt.value_scale))
      spatial_transform.append(normalize)
      return Compose(spatial_transform)
  ```

- `get_temporal_transform` 함수도 추가함.

  ```python
  def get_temporal_transform(opt):
      temporal_transform = []
      if opt.sample_t_stride > 1:
          temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
      if opt.train_t_crop == 'random':
          temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
      elif opt.train_t_crop == 'center':
          temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
      return TemporalCompose(temporal_transform)
  ```

- `main_worker` 함수 내에 전처리 설정을 추가함.

  ```python
  criterion = CrossEntropyLoss().to(opt.device)

  # Define spatial_transform and temporal_transform
  spatial_transform = get_spatial_transform(opt)
  temporal_transform = get_temporal_transform(opt)

  if not opt.no_train:
      (train_loader, train_sampler, train_logger, train_batch_logger,
       optimizer, scheduler) = get_train_utils(opt, parameters, subset=opt.subset)
  ```

- 이후 `main_worker` 함수를 수정하여 전처리가 적용되도록 함.

# Step 11: 기타 코드 수정 및 디버깅

- `loader.py`의 `__call__` 메서드를 수정하여 `frame_indices`가 리스트의 리스트인 경우를 처리함.

  ```python
  def __call__(self, video_path, frame_indices):
      # 디버깅: 원본 프레임 인덱스 확인
      print(f"[디버그] 원본 프레임 인덱스 길이: {len(frame_indices)}")

      video = []
      if isinstance(frame_indices[0], list):
          frame_indices = frame_indices[0]
          print(f"[디버그] 첫 번째 리스트 사용, 프레임 인덱스 길이: {len(frame_indices)}")

      missing_images = 0  # 로드 실패한 이미지 카운터

      for i in frame_indices:
          if isinstance(i, (list, tuple)):
              i = i[0]

          # 이미지 경로 설정
          image_path = video_path / self.image_name_formatter(i)

          # 이미지 로드 확인 및 디버그 출력 최소화
          if image_path.exists():
              video.append(self.image_loader(image_path))
          else:
              missing_images += 1

      # 디버깅: 로드된 이미지 개수와 실패한 이미지 개수
      print(f"[디버그] 로드된 이미지 수: {len(video)}, 실패한 이미지 수: {missing_images}")

      return video
  ```

- `videoset_multiclips.py`의 `collate_fn` 함수를 수정하여 배치 처리가 올바르게 되도록 함.

  ```python
  def collate_fn(batch):
      batch_clips, batch_targets = zip(*batch)

      # 클립을 배치 차원으로 스택 (batch_size, 3, 16, 112, 112)
      batch_clips = torch.stack(batch_clips, 0)  # 올바른 배치 스택

      # 타겟도 스택
      batch_targets = torch.tensor(batch_targets)

      # 디버그 메시지 추가
      print(f"[DEBUG] batch_clips shape: {batch_clips.shape}")
      print(f"[DEBUG] batch_targets length: {len(batch_targets)}")
      print(f"[DEBUG] batch_targets (first element): {batch_targets[0].item()}")

      return batch_clips, batch_targets
  ```

- `dataset.py`에서 불필요한 출력문을 제거하고, `activitynet.py`의 `__make_dataset` 함수에서도 개별 경로 확인에 대한 `print` 문을 제거함.

# Step 12: 학습 및 추론 명령어 정리

- 초기 학습 명령어:

  ```bash
  CUDA_VISIBLE_DEVICES=0 python main.py --root_path D:/Tutorial --video_path D:/ActivityNetFrames \
  --annotation_path D:/Tutorial/ActivityNet/Evaluation/data/activity_net.v1-3.min.json \
  --result_path D:/Tutorial/results --dataset activitynet --model resnet \
  --model_depth 50 --n_classes 200 --batch_size 8 --n_epochs 50 --learning_rate 0.01 \
  --n_threads 12 --checkpoint 5
  ```

- 새로운 학습 명령어:

  ```bash
  CUDA_VISIBLE_DEVICES=0 python main.py --root_path D:/Tutorial --video_path D:/ActivityNetFrames \
  --annotation_path D:/Tutorial/ActivityNet/Evaluation/data/activity_net.v1-3.min.json \
  --result_path D:/Tutorial/results --dataset activitynet --model resnet --model_depth 50 \
  --n_classes 200 --batch_size 16 --n_epochs 100 --learning_rate 0.001 --n_threads 4 \
  --checkpoint 5 --sample_size 112 --sample_duration 16 --train_crop random \
  --learning_rate_schedule step --step_size 30 --weight_decay 1e-4 \
  --pretrain_path pretrained_models/resnet-50.pth
  ```

- 사전 학습된 모델로 추론하는 명령어:

  ```bash
  python main.py --root_path D:/Tutorial --video_path D:/ActivityNetFrames \
  --annotation_path D:/Tutorial/ActivityNet/Evaluation/data/activity_net.v1-3.min.json \
  --result_path D:/Tutorial/results --dataset activitynet --model resnet --model_depth 50 \
  --n_classes 200 --batch_size 8 --n_threads 8 --checkpoint 5 \
  --resume_path D:/Tutorial/results/save_50.pth --no_train --no_val --inference \
  --output_topk 5 --inference_batch_size 1
  ```

