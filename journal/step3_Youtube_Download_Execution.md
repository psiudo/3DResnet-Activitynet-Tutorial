# Step 3: 유튜브 비디오 다운로드 실행

- PyCharm 터미널에서 `fetch_activitynet_videos.sh`를 실행하려 했으나 JSON 파일을 찾을 수 없다는 문제 발생.
- ReadMe 파일에 나왔던 `activity_net.v1-X.json` 파일이 실제로는 `v3` 버전으로 이름이 변경된 상태였음.
- 인자 이름을 변경하여 다시 시도했으나, youtube-dl이 오래된 버전이라 유튜브의 최신 구조를 지원하지 않음. 이 문제를 해결하기 위해 `yt-dlp`로 대체하고 `run_crosscheck.py` 코드도 `yt-dlp`를 사용할 수 있도록 수정함.
- 약 16,000개의 비디오 다운로드 완료.
