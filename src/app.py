import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

# Load model
model = joblib.load("models/diabetes_pipeline.pkl")

# Title
st.title("🩺 Diabetes Prediction App")
st.write("Machine Learning based diabetes risk prediction")
st.divider()

# Inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 95)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 22.5)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.3)
age = st.number_input("Age", 1, 120, 25)

# Prediction
if st.button("Predict Diabetes"):
    X = np.array([[pregnancies, glucose, blood_pressure,
                   skin_thickness, insulin, bmi, dpf, age]])

    prob = model.predict_proba(X)[0][1]

    st.write(f"Diabetes Risk Probability: {prob:.2f}")

    if prob < 0.4:
        st.success("✅ Low Risk of Diabetes")
    elif prob < 0.6:
        st.warning("⚠️ Medium Risk of Diabetes")
    else:
        st.error("🚨 High Risk of Diabetes")

    st.caption("Educational purpose only. Not medical advice.")
