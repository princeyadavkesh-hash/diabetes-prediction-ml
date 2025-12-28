import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

model = joblib.load("models/diabetes_pipeline.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("Machine Learning based diabetes risk prediction")

st.divider()

pregnancies = st.number_input("Pregnancies", 0, 20, 2)
glucose = st.number_input("Glucose", 0, 300, 130)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 85)
bmi = st.number_input("BMI", 0.0, 70.0, 28.5)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 35)

if st.button("Predict Diabetes"):
    X = np.array([[pregnancies, glucose, blood_pressure,
                   skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(X)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

st.caption("⚠️ Educational use only. Not medical advice.")
