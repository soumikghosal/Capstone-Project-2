import cloudpickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify 
    
def predict_level(patient, pipe, xgb_model):
    patient = pd.DataFrame(data=patient, index=[0])
    X_test = pipe.transform(patient)
    pred = xgb_model.predict(X_test)
    
    # diabetes: 1 if the patient is in a risk of diabetes or 0 if the patient is not
    pred_label = ["No risk of diabetes" if ele==0 else "Risk of diabetes" for ele in pred]
    return pred_label

with open('../models/diabetes_pred_model.bin', 'rb') as f_in:
    pipe, xgb_model = cloudpickle.load(f_in)
    
app = Flask('diabetes-prediction')

@app.route('/')
def hello_world():
    return "<p>HWorld!</p>"

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    prediction = predict_level(patient, pipe, xgb_model)
    result = {'diabetes_prediction': str(prediction)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
 
 