import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model, encoders, and feature order
model = pickle.load(open("best_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
feature_order = pickle.load(open("feature_order.pkl", "rb"))

st.title("🧠 Autism Spectrum Disorder (ASD) Prediction")
st.write("This app predicts whether a person is **ASD Positive** or **ASD Negative** based on the given details.")

# Input form
with st.form("prediction_form"):
    st.subheader("🔹 Enter Patient Details")

    inputs = {}
    for feature in feature_order:
        if feature in encoders:  # categorical feature
            options = encoders[feature].classes_
            selected = st.selectbox(f"{feature}:", options)
            encoded = encoders[feature].transform([selected])[0]
            inputs[feature] = encoded

        elif feature == "age":
            inputs[feature] = st.slider("Age", min_value=1, max_value=100, value=25)

        elif feature == "result":
            inputs[feature] = st.slider("Test Result", min_value=0, max_value=20, value=10)

        else:  # binary features like A1_Score, etc.
            inputs[feature] = st.radio(f"{feature}:", [0, 1], index=0)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # Prepare the input in correct order
    input_df = pd.DataFrame([inputs])[feature_order]

    st.subheader("📊 Processed Input Data")
    st.write(input_df)

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    label = "✅ ASD Positive" if prediction == 1 else "❌ ASD Negative"
    confidence = np.max(probability) * 100

    st.subheader("🔎 Prediction Result")
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")
