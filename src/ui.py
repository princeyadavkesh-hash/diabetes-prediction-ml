import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes risk")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=130)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.5)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=35)

if st.button("🔍 Predict Diabetes"):
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            if result["prediction"] == 1:
                st.error("⚠️ High Risk: Diabetic")
            else:
                st.success("✅ Low Risk: Non-Diabetic")
        else:
            st.error("API Error. Please check FastAPI server.")

    except Exception as e:
        st.error("Could not connect to FastAPI server.")
