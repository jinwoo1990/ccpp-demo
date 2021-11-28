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
- AWS EC2 (computing & deployment)

## 사용 방법

### 배포

1. AWS EC2 Instance 생성
- OS Ubuntu 20.04 선택

2. SSH 접속

3. Instance 환경 설정
- root 패스워드 재설정: 
  - `sudo passwd root`
- 사용자 전환: 
  - `su root`
- 패키지 매니저 업데이트: 
  - `apt update`
- 패키지 매니저 업그레이드 (필수 x): 
  - `apt upgrade`
- 필요 패키지 설치: 
  - `sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common`
  - `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -`
  - `sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"`
  - `sudo systemctl enable docker && service docker start`
- pip install 에러 처리: 
  - `apt-get install libssl-dev`

4. `sh init.sh` 로 컨테이너 초기화 및 기본 데이터 불러오기
- `docker-compose up --build -d` 실행됨


### 사용
어플리케이션 접속: https://<domain ip>:80/

어플리케이션 멈추기: `docker-compose stop`
어플리케이션 다시 띄우기: `docker-compose start`

어플리케이션 삭제: `docker-compose down`
어플리케이션 다시 만들기: `sh init.sh`


## Version

### Version 1.0.0

