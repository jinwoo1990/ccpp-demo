# CCPP Streamlit Demo App

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

5. `.env` 파일 생성
```shell
# Custom env
PREPROCESSING_OBJECTS_PATH=/api/model/preprocessing_objects_211129_0103.pkl
MODEL_PATH=/api/model/XGB_211129_0103.pkl
MODEL_TYPE=tree
BACKGROUND_DATA_PATH=/api/model/background_data_211129_0103.pkl
EXPLAINER_PATH=/api/model/XGB_explainer_211129_0103.pkl
```
- Git 에 포함되어 있지 않은 환경변수 파일로 추가 생성 필요
- model 새로 생성하면 바라보는 파일명 환경변수 변경 (추후 ml-streamlit-app 처럼 db로 처리하거나 mlflow 기능 활용 가능 )

6. `sh init.sh` 로 컨테이너 초기화 및 기본 데이터 불러오기
- `docker-compose up --build -d` 실행됨


### 사용
어플리케이션 접속: `https://<domain-ip>:8501/`

어플리케이션 멈추기: `docker-compose stop`
어플리케이션 다시 띄우기: `docker-compose start`

어플리케이션 삭제: `docker-compose down`
어플리케이션 다시 만들기: `sh init.sh`


## Version

### Version 1.0.0
