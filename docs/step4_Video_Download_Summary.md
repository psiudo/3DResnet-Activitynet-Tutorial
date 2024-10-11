# Step 4: 비디오 다운로드 과정 정리

- 비디오를 다운로드하기 위해 `fetch_activitynet_videos.sh` 스크립트를 실행하였음.
- `run_crosscheck.py`를 사용해 `activity_net.v1-3.min` 파일을 읽고, 이미 다운로드된 비디오와 다운로드되지 않은 비디오를 교차 검토함.
- 필요한 비디오만 `command_list.txt` 파일에 작성하여 다운로드하도록 수정함.
