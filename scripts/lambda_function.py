import cloudpickle
import pandas as pd
import numpy as np
    
with open('diabetes_pred_model.bin', 'rb') as f_in:
    pipe, xgb_model = cloudpickle.load(f_in)

def predict_level(patient, pipe=pipe, xgb_model=xgb_model):
    patient = pd.DataFrame(data=patient, index=[0])
    X_test = pipe.transform(patient)
    pred = xgb_model.predict(X_test)
    
    # stroke: 1 if the patient had a stroke or 0 if not
    pred_label = ["No risk of diabetes" if ele==0 else "Risk of diabetes" for ele in pred]
    return pred_label

def lambda_handler(event, context):
       
    patient = {
        'gender': event['gender'],
        'age': event['age'],
        'hypertension': event['hypertension'],
        'heart_disease': event['heart_disease'],
        'smoking_history': event['smoking_history'],
        'hba1c_level': event['hba1c_level'],
        'bmi': event['bmi'],
        'blood_glucose_level': event['blood_glucose_level']
    } 
    
    result = predict_level(patient)
    return result












 
 