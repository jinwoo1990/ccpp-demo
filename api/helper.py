import os
import pickle
import shap


PREPROCESSING_OBJECTS_PATH = os.environ.get('PREPROCESSING_OBJECTS_PATH', './model/PREPROCESSING_OBJECTS_211129_0103.pkl')
MODEL_PATH = os.environ.get('MODEL_PATH', './model/XGB_211129_0103.pkl')
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'tree')
BACKGROUND_DATA_PATH = os.environ.get('BACKGROUND_DATA_PATH', './model/background_data_211129_0103.pkl')
EXPLAINER_PATH = os.environ.get('EXPLAINER_PATH', './model/XGB_explainer_211129_0103.pkl')


def load_preprocessor_and_model():
    """
    preprocessor 와 모델을 로드하기 위한 함수

    :return: preprocessing objects 와 model
    """
    with open(PREPROCESSING_OBJECTS_PATH, 'rb') as handle:
        preprocessing_objects = pickle.load(handle)

    with open(MODEL_PATH, 'rb') as handle:
        model = pickle.load(handle)

    return preprocessing_objects, model


def preprocess_record(data, scaler):
    """
    preprocessing 결과를 반환하는 함수

    :param data: 전처리되지 않은 input 데이터
    :param scaler: scaling 객체
    :return: preprocessing 이 완료된 데이터
    """
    data = scaler.transform(data)

    return data


def predict_record(data, model):
    """
    예측값과 예측확률을 반환하는 함수

    :param data: preprocess_record() 을 통해 전치리가 된 input 데이터
    :param model: 예측 모델
    :return: 모델 예측값
    """
    prediction = model.predict(data)[0]

    return prediction


def generate_shap_explainer(model):
    with open(BACKGROUND_DATA_PATH, 'rb') as handle:
        background_data = pickle.load(handle)

    if MODEL_TYPE == 'linear':
        explainer = shap.LinearExplainer(model, background_data)
    elif MODEL_TYPE == 'tree':
        explainer = shap.TreeExplainer(model, background_data)
    elif MODEL_TYPE == 'neural-network':
        explainer = shap.DeepExplainer(model, background_data)
    else:
        explainer = shap.KernelExplainer(model.predict, background_data)

    return explainer


def get_base_and_shap_values(data, model):
    if MODEL_TYPE == 'neural-network':
        shap_explainer = generate_shap_explainer(model)
        base_value = shap_explainer.expected_value[0]
        shap_values = shap_explainer.shap_values(data)[0][0].tolist()
    else:
        with open(EXPLAINER_PATH, 'rb') as handle:
            shap_explainer = pickle.load(handle)
        base_value = shap_explainer.expected_value
        shap_values = shap_explainer.shap_values(data)[0].tolist()

    return base_value, shap_values
