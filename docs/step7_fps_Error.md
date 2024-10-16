# fps Error
-python D:/Tutorial/main.py --root_path D:/Tutorial --video_path D:/ActivityNetFrames --annotation_path D:/Tutorial/ActivityNet/Evaluation/data/activity_net.v1-3.min.json --result_path D:/Tutorial/results --dataset activitynet --n_classes 400 --n_epochs 50 --batch_size 8
를 실행했더니 fps관련 에러 발생
- 따라서, activitynet.py 파일의  get_video_ids_annotations_and_fps 함수에서 get을 추가하는 부분에 넣어준다.
- 코드는 다음과 같다.

  
'
def get_video_ids_annotations_and_fps(data, subset):
    video_ids = []
    annotations = []
    fps_values = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            fps_values.append(value.get('fps', 30))

    return video_ids, annotations, fps_values
    ...
  '
