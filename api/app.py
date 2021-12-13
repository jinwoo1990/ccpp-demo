from flask import Flask, request, jsonify
import pandas as pd
import helper


app = Flask(__name__)


@app.route('/model/predict/', methods=['POST'])
def predict():
    """
    예측 결과를 반환하는 함수

    :return: 전처리된 input 데이터, 예측 결과 및 확률을 포함한 dataframe
    """
    json_data = request.get_json()
    preprocessing_objects, model = helper.load_preprocessor_and_model()
    scaler = preprocessing_objects['scaler']
    features_selected = preprocessing_objects['features_selected']

    data = pd.DataFrame(json_data)[features_selected]
    preprocessed_data = helper.preprocess_record(data, scaler)
    prediction = helper.predict_record(preprocessed_data, model)

    output = data.to_dict('records')[0]
    output['prediction'] = float(prediction)
    output = jsonify(output)

    return output


@app.route('/model/calculate-shap-values/', methods=['POST'])
def calculate_shap_values():
    """
    shap force plot 객체와 shap values 를 반환하는 함수

    :return: 전처리된 input 데이터, 예측 결과 및 확률을 포함한 dataframe
    """
    json_data = request.get_json()
    preprocessing_objects, model = helper.load_preprocessor_and_model()
    scaler = preprocessing_objects['scaler']
    features_selected = preprocessing_objects['features_selected']

    data = pd.DataFrame(json_data)[features_selected]
    preprocessed_data = helper.preprocess_record(data, scaler)
    base_value, shap_values = helper.get_base_and_shap_values(preprocessed_data, model)

    output = data.to_dict('records')[0]
    output['base_value'] = base_value
    output['shap_values'] = shap_values
    output = jsonify(output)

    return output


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
