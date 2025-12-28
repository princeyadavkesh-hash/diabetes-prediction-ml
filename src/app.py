from fastapi import FastAPI
import joblib
import numpy as np


app = FastAPI(title="Diabetes Prediction API")

model = joblib.load("models/diabetes_pipeline.pkl")



@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.post("/predict")
def predict_diabetes(data: dict):
    features = np.array([[
        data["Pregnancies"],
        data["Glucose"],
        data["BloodPressure"],
        data["SkinThickness"],
        data["Insulin"],
        data["BMI"],
        data["DiabetesPedigreeFunction"],
        data["Age"]
    ]])

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0]),
        "result": "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    }





