# TypeError: unsupported operand type(s) for /: 'WindowsPath' and 'int' 오류 발생
- `video_path_formatter` 람다 함수에서 `label`이라는 변수가 정의되지 않았기 때문에 발생한다.  
- `label` 변수가 정의되지 않아, 이를 사용하려고 할 때 `TypeError`가 발생한다.
- 이를 해결하기 위해 아래 코드를 `activitynet.py`의 `__make_dataset` 함수에 추가한다.

```python
if annotations[i]:
    label = annotations[i][0]['label']  # 첫 번째 레이블 사용
else:
    label = 'unknown'  # 레이블이 없을 경우 기본값 사용
