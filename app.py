import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction App ❤️")

# Collect user input
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", [0, 1])
chest_pain_type = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
resting_blood_pressure = st.number_input("Resting Blood Pressure", 80, 250, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
max_heart_rate = st.number_input("Max Heart Rate", 60, 220, 150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", [0, 1])
st_depression = st.number_input("ST Depression", 0.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])
num_major_vessels = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thalassemia = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Create DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "chest_pain_type": [chest_pain_type],
    "resting_blood_pressure": [resting_blood_pressure],
    "cholesterol": [cholesterol],
    "fasting_blood_sugar": [fasting_blood_sugar],
    "resting_ecg": [resting_ecg],
    "max_heart_rate": [max_heart_rate],
    "exercise_induced_angina": [exercise_induced_angina],
    "st_depression": [st_depression],
    "st_slope": [st_slope],
    "num_major_vessels": [num_major_vessels],
    "thalassemia": [thalassemia]
})

# Predict
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(f"⚠️ **has heart disease**. Probability: {probability:.2f}")
    else:
        st.success(f"✅ **does NOT have heart disease**. Probability: {probability:.2f}")
