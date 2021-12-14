# CCPP Streamlit Demo App

Temporary Address: http://13.209.234.231:8501/

## 기술 요소
docker
- docker-compose (container management)

python
- streamlit (front-end)
- sklearn (machine learning model)
- xgboost (machine learning model)
- tensorflow (deep learning model)

AWS
- AWS Lightsail (computing & deployment)

## 사용 방법

### 준비사항

1. `.env` 파일 생성
```shell
# Custom env
PREPROCESSING_OBJECTS_PATH=/api/model/preprocessing_objects_211129_0103.pkl
MODEL_PATH=/api/model/XGB_211129_0103.pkl
MODEL_TYPE=tree
BACKGROUND_DATA_PATH=/api/model/background_data_211129_0103.pkl
EXPLAINER_PATH=/api/model/XGB_explainer_211129_0103.pkl
```
- Git 에 포함되어 있지 않은 환경변수 파일로 추가 생성 필요
- model 새로 생성하면 바라보는 파일명 환경변수 변경 (추후 ml-streamlit-app 처럼 db로 처리하거나 mlflow 기능 활용 가능)

2. logs 디렉토리 생성
- 로컬로 돌릴 때 .log 파일 .gitignore 에 등록되어 있으므로 logs 디렉토리 생성 필요
- Docker 로 띄울 때는 `RUN mkdir -p /api/logs` 처럼 선언되어있어 문제 없음


### 배포

1. AWS Lightsail Instance 생성
- OS Ubuntu 20.04 선택

2. SSH 접속

3. Instance 환경 설정
- root 패스워드 재설정: 
  - `sudo passwd root`
- 패키지 매니저 업데이트: 
  - `apt-get update`
- 필요 패키지 설치: 
  - `sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release`
  - `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg`
  - `echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`
  - `sudo systemctl enable docker && service docker start`
- Docker 설치
  - `sudo apt-get update`
  - `sudo apt-get install docker-ce docker-ce-cli containerd.io`
- Docker 권한 설정
  - `sudo usermod -a -G docker $USER`
  - `sudo service docker restart`
- Docker-compose 설치
  - ```sudo curl -L https://github.com/docker/compose/releases/download/v2.1.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose```
  - `sudo chmod +x /usr/local/bin/docker-compose`
- pip install 에러 처리: 
  - `apt-get install libssl-dev`

4. Git clone
- `git clone https://github.com/jinwoo1990/ccpp-demo`

5. `sh init.sh` 로 컨테이너 초기화 및 기본 데이터 불러오기
- `docker-compose up --build -d` 실행됨


### 사용
어플리케이션 접속: `https://<domain-ip>:8501/`

어플리케이션 멈추기: `docker-compose stop`

어플리케이션 다시 띄우기: `docker-compose start`

어플리케이션 삭제: `docker-compose down`

어플리케이션 다시 만들기: `sh init.sh`


## Version

### Version 1.0.0
Jupyter notebook 에서 만든 모델을 바탕으로 작동하는 데모 어플리케이션 개발

### Version 1.0.1 (개발중)
기본 시스템 로깅 기능 구현 (python 내장 logging 모듈 사용)

mlflow 를 활용한 모델 학습 파이프라인 개발 및 정확도 트랙킹 기능 구현

추가 사용 방법
- api 디렉토리로 이동
- `mlflow ui` 로 mlflow ui 띄우기
- `python train.py <sys.argv[1]> <sys.argv[2]> <...>` 로 정의된 파라미터를 넘겨 모델 학습
- 127.0.0.1:5000 (로컬에서 띄웠을 때 기본 주소, 설정에 따라 다를 수 있음) 에서 새로 고침 후 모델 학습 결과 확인
