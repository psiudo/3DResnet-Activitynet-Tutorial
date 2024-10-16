# TypeError: unsupported operand type(s) for /: 'WindowsPath' and 'int' 오류발생
- video_path_formatter 람다 함수에서 label이라는 변수가 정의되지 않았기 때문에
label 변수는 정의되지 않았고, 이로 인해 TypeError가 발생한다. label이라는 변수를 사용하고 있는데, 
실제 코드에는 이 변수가 정의되지 않았다.
- 이를 위해 다음과 같은 코드를 activitynet.py의 __make_dataset 함수에서
'''python
# label을 정의하기 위해 annotations에서 가져옴
            if annotations[i]:
                label = annotations[i][0]['label']  # 첫 번째 레이블 사용
            else:
                label = 'unknown'  # 레이블이 없을 경우 기본값 사용'
를 추가
