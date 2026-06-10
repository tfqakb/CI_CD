import pandas as pd
from flask import Flask, request, render_template
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi import FastAPI
from sklearn.metrics import confusion_matrix, accuracy_score

app = FastAPI(title="Churn Prediction API", description="API for predicting customer churn using a trained model.")

#pickel_in = open('model_app/classifier.pkl', 'rb')
#classifier = pickle.load(pickel_in)
model = joblib.load('model_app/classifier.pkl')

@app.get('/')
def welcome_message():
    return {"message": "Welcome to the Churn Prediction App!"}

# @app.get('/predict')
# def predict_churn():
#     y_pred = model.predict(sc_xtest)
#     y_pred = (y_pred > 0.5)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     acc_score = accuracy_score(y_test, y_pred)
#     return {"conf_matrix": conf_matrix.tolist(), "acc_score": acc_score}
class CustomerDataChurn(BaseModel):
    features: list

@app.post('/predict')
def predict_churn(data: CustomerDataChurn):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    app.run(debug=True)