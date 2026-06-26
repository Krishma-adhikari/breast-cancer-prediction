import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model (2).pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺")

st.title("🩺 Breast Cancer Prediction")
st.write("Enter the patient's tumor measurements below.")

# User Inputs
mean_radius = st.number_input("Mean Radius", min_value=0.0, format="%.2f")
mean_texture = st.number_input("Mean Texture", min_value=0.0, format="%.2f")
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, format="%.2f")
mean_area = st.number_input("Mean Area", min_value=0.0, format="%.2f")
mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, format="%.5f")

# Predict button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "mean_radius": [mean_radius],
        "mean_texture": [mean_texture],
        "mean_perimeter": [mean_perimeter],
        "mean_area": [mean_area],
        "mean_smoothness": [mean_smoothness]
    })

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Prediction: No Cancer")
    else:
        st.error("⚠️ Prediction: Cancer")