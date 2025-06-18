import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model, encoders, and feature order
model = pickle.load(open("best_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
feature_order = pickle.load(open("feature_order.pkl", "rb"))

st.title("Autism Spectrum Disorder Prediction")

# Create input form
with st.form("prediction_form"):
    st.subheader("Enter Patient Details")

    inputs = {}
    for feature in feature_order:
        if feature in encoders:
            options = encoders[feature].classes_
            selected = st.selectbox(f"{feature}:", options)
            encoded = encoders[feature].transform([selected])[0]
            inputs[feature] = encoded
        elif feature == "age":
            inputs[feature] = st.slider("Age", 1, 100, 25)
        elif feature == "result":
            inputs[feature] = st.slider("Test Result", 0, 20, 10)
        else:
            inputs[feature] = st.number_input(f"{feature}:", min_value=0, max_value=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare the input in correct order
    input_df = pd.DataFrame([inputs])[feature_order]

    # Predict
    prediction = model.predict(input_df)[0]
    label = "ASD Positive" if prediction == 1 else "ASD Negative"
    st.success(f"Prediction: {label}")
