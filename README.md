# Prediction Models

#### 설치 방법

1. Python은 버전 3.6.x를 설치

2. 아래와 같이 virtual env를 생성

   ```
   conda create --name <envname> --file requirements.txt
   activate <envname>
   ```

3. Jupyter notebook을 사용하고 싶은 경우

   ```
   activate <envname> # 앞서 만든 virtual env를 활성화
   jupyter notebook 
   ```

4. 로그 데이터는 레포지토리에 올리지 않을 것이므로 아래와 같은 구조로 직접 넣어주어야 합니다.

   ```
   prediction_models
   	┕ log_data
   	┕ logs
   		┕ convertlog.2017100x
   ```