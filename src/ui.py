import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load ML model
model = joblib.load("models/diabetes_pipeline.pkl")

st.markdown(
    "<h1 style='text-align:center; color:#2E86C1;'>🩺 Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Machine Learning based health risk prediction</p>",
    unsafe_allow_html=True
)

st.divider()

st.subheader("🧾 Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose Level", 0, 300, 130)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 85)
    bmi = st.number_input("BMI", 0.0, 70.0, 28.5)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 35)

st.divider()

if glucose == 0 or bmi == 0:
    st.warning("⚠️ Glucose and BMI must be greater than zero")

if st.button("🔍 Predict Diabetes", use_container_width=True):
    features = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
