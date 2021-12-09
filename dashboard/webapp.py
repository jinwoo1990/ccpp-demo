import streamlit as st
import streamlit.components.v1 as components
from collections import OrderedDict
import os
import pickle
import json
import datetime
import requests
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# 모델 API endpoint
url = 'http://api:5000'  # TODO: Docker 배포 시 설정
# url = 'http://127.0.0.1:5000'
predict_endpoint = '/model/predict/'
shap_endpoint = '/model/calculate-shap-values/'

# 기타 변수 초기화
last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_user_input_features():
    """
    메뉴 및 기타 input 값을 받기 위한 함수

    :return: json 형식의 user input 데이터
    """

    user_features = {"menu_name": st.sidebar.selectbox('Menu', ['predict'])}

    return [user_features]


def get_raw_input_features():
    """
    raw data input 값을 받기 위한 함수

    :return: json 형식의 raw input 데이터
    """
    raw_features = {"AP": st.sidebar.slider('Atomospheric Pressure', 990.0, 1040.0, (990.0 + 1040.0)/2),
                    "AT": st.sidebar.slider('Ambient Temperature', 1.0, 38.0, (1.0 + 38.0)/2),
                    "RH": st.sidebar.slider('Relative Humidity', 20.0, 105.0, (20.0 + 105.0)/2),
                    "V": st.sidebar.slider('Vaccum', 25.0, 85.0, (25.0 + 85.0)/2)
                    }

    return [raw_features]


def load_session_variable_from_model():
    """
    모델 종류가 바뀌거나 모델이 업데이트 되었을 때 새로 session variable 에 모델 정보 업데이트를 위한 함수

    :return: None
    """


def draw_shap_plot(base_value, shap_values, data, height=None):
    """
    shap plot 을 streamlit 어플리케이션 상에 표시하기 위한 함수

    :param data: 예측 결과값을 제외하고 사용된 변수 값만 포함하는 데이터
    :param base_value: shap 기준값
    :param shap_values: 변수별 shap 값
    :param height: 그림 height
    :return: None
    """
    p = shap.force_plot(base_value, shap_values, data)
    shap_html = f"<head>{shap.getjs()}</head><body>{p.html()}</body>"
    components.html(shap_html, height=height)


def streamlit_main():
    """
    streamlit main 함수

    :return: None
    """
    st.title('CCPP Power Output Predictor')
    # 화면 오른쪽에 last updated 표시
    components.html(
        f'''<p style="text-align:right; font-family:'IBM Plex Sans', sans-serif; font-size:0.8rem; color:#585858";>\
            Last Updated: {last_updated}</p>''', height=30)

    # sidebar input 값 선택 UI 생성
    st.sidebar.header('User Menu')
    user_input_data = get_user_input_features()

    st.sidebar.header('Raw Input Features')
    raw_input_data = get_raw_input_features()

    submit = st.sidebar.button('Get predictions')
    if submit:
        results = requests.post(url + predict_endpoint, json=raw_input_data)
        results = json.loads(results.text)

        # 예측 결과 표시
        st.subheader('Results')
        prediction = results["prediction"]
        st.write("Prediction: ", round(prediction, 2))

        # expander 형식으로 model input 표시
        st.subheader('Input Features')
        features_selected = ['AT', 'V', 'AP', 'RH']

        model_input_expander = st.beta_expander('Model Input')
        model_input_expander.write('Input Features: ')
        model_input_expander.text(", ".join(list(raw_input_data[0].keys())))
        model_input_expander.json(raw_input_data[0])
        model_input_expander.write('Selected Features: ')
        model_input_expander.text(", ".join(features_selected))
        selected_features_values = OrderedDict((k, results[k]) for k in features_selected)
        model_input_expander.json(selected_features_values)

        # shap 값 계산
        shap_results = requests.post(url + shap_endpoint, json=raw_input_data)
        shap_results = json.loads(shap_results.text)

        base_value = shap_results['base_value']
        shap_values = np.array(shap_results['shap_values'])

        # shap force plot 표시
        st.subheader('Interpretation Plot')
        draw_shap_plot(base_value, shap_values, pd.DataFrame(raw_input_data)[features_selected])

        # shap feature importance plot 표시
        # st.subheader('Shap Feature Importance (Absolute Value)')
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # shap.summary_plot(shap_values.reshape(1, -1), pd.DataFrame(raw_input_data)[features_selected], plot_type='bar')
        # st.pyplot(fig)

        # expander 형식으로 shap detail 값 표시
        shap_detail_expander = st.beta_expander('Shap Detail')
        for key, item in zip(features_selected, shap_values):
            shap_detail_expander.text('%s: %s' % (key, item))


if __name__ == '__main__':
    streamlit_main()

