import os
import sys
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import mlflow
from loggers import logger


DATA_PATH = os.environ.get('DATA_PATH', './data/Folds5x2_pp.xlsx')


# Evaluate metrics
def eval_metrics(actual, pred):
    """
    regression 평가 결과를 산출하는 함수

    :param actual: 실제값
    :param pred: 모델 예측값
    :return: rmse, mae, r2
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


def main():
    """
    mlflow 를 이용해 모델 학습과 학습 결과를 저장하는 함수

    터미널에서 api 폴더로 이동 후 `python train.py 0.1 0.1` 같은 식으로 모델 파라미터를 넘겨 학습 가능
    api 폴더 내에서 `mlflow ui` 를 입력해 실행되는 127.0.0.1:5000 에서 학습된 모델과 모델 결과를 확인 가능

    :return: None
    """
    # 데이터 로드
    logger.info('## Load data')
    data = pd.read_excel(DATA_PATH)
    logger.info('data: \n %s' % data.head())

    # null 값 채우기
    logger.info('## Impute null features')
    null_counts = data.isnull().sum()
    null_cols = list(null_counts[null_counts > 0].index)
    logger.info('null_cols: \n %s' % null_cols)
    # null 인 컬럼이 0개가 아닐 때 최빈값으로 null 값 채우기
    if len(null_cols) != 0:
        null_converter = defaultdict(str)
        for col in null_cols:
            null_converter[col] = data[col].value_counts().index[0]
        for col in null_converter.keys():
            temp = data[col].fillna(null_converter[col])
            data[col] = temp
    logger.info('data: \n %s' % data.head())

    # categorical 변수 변환
    # TODO: 모두 numeric data 라 코드 생략. But, generalized 한 코드를 위해 추후 추가

    # feature selection
    # TODO: 컬럼이 4개라 일단 전부 사용해서 생략. 마찬가지로, generalized 한 코드를 위해 추후 추가
    features_selected = ['AT', 'V', 'AP', 'RH']
    target_list = ['PE']

    # scale features
    logger.info('## Scale features')
    # X, y 분리
    X = data[features_selected].values
    y = np.ravel(data[target_list].values, order='C')  # TODO: multi-target 일 때 ravel 안 쓰도록 변경
    logger.info('X: \n %s' % X)
    logger.info('y: \n %s' % y)
    # scaler 적용
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    logger.info('scaled X: \n %s' % X)
    logger.info('scaled y: \n %s' % y)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=32)
    logger.info('X_train.shape: %s, X_test.shape: %s' % (X_train.shape, X_test.shape))
    logger.info('y_train.shape: %s, y_test.shape: %s' % (y_train.shape, y_test.shape))

    # 모델 parameter 입력
    # TODO: 뒤에 모델 fitting 부분과 연결지어 ElasticNet 외에도 다른 알고리즘들 인자로 받는 식으로 해서 테스트 가능하게 구성
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    logger.info('alpha: %s, l1_ration: %s' % (alpha, l1_ratio))
    # 모델 fitting
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)
    logger.info('Model %s created successfully.' % model)

    # evaluation metrics 계산
    y_hat = model.predict(X_test)
    logger.info('y_hat: \n %s' % y_hat)
    rmse, mae, r2 = eval_metrics(y_test, y_hat)
    logger.info('rmse: %s, mae: %s, r2: %s' % (rmse, mae, r2))

    # mlflow 학습 모델 저장
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
