import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load trained ML model
model = joblib.load("models/diabetes_pipeline.pkl")

# App title
st.markdown(
    "<h1 style='text-align:center; color:#2E86C1;'>🩺 Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Machine Learning based health risk prediction</p>",
    unsafe_allow_html=True
)

st.divider()

# Input section
st.subheader("🧾 Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=130)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.5)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=35)

st.divider()

# Basic validation
if glucose == 0 or bmi == 0:
    st.warning("⚠️ Glucose and BMI must be greater than zero")

# Prediction
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

    st.divider()

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

# Footer
st.markdown(
    "<p style='text-align:center; font-size:12px;'>⚠️ Educational purpose only. Not medical advice.</p>",
    unsafe_allow_html=True
)
