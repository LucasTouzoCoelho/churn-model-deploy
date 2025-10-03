import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# --- Load train artfacts ---
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("src/models/model.pkl")
    return artifacts["model"], artifacts["scaler"], artifacts["features"]

model, scaler, features = load_artifacts()

# --- Loading json file ---
@st.cache_resource
def load_features_info():
    with open("src/features_info.json") as f:
        return json.load(f)

features_info = load_features_info()

st.title("📊 Churn Prediction App")

# --- Choice mode ---
option = st.radio(
    "How would you like to predict?",
    ("CSV Upload", "Manually insert the values")
)

# --- Modo 1: CSV upload ---
if option == "CSV Upload":
    uploaded_file = st.file_uploader("Please upload the CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            # Ensure the model is taking only the expected features
            X = data[features]
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            data["Churn_Prediction"] = predictions
            data["Churn_Probability"] = probabilities

            st.success("✅ Predictions generated successfully!")
            st.dataframe(data)

            # Alow the download
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download results", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error when generating the file: {e}")

# --- Mode 2: Manually insert ---
else:
    st.subheader("🔎 Fill the values to predict churn:")

    input_data = {}
    for feature in features:
        info = features_info.get(feature, {})

        # Input numérico
        if info.get("type") == "numeric":
            min_val = float(info.get("min", 0.0))
            max_val = float(info.get("max", 100.0))
            default = float(info.get("mean", (min_val + max_val)/2))
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=default
            )

        # Input categórico
        elif info.get("type") == "categorical":
            options = info.get("values", [])
            input_data[feature] = st.selectbox(f"{feature}", options)

        # Fallback para qualquer feature não especificada
        else:
            input_data[feature] = st.text_input(f"{feature}", value="")

    # Converter para DataFrame
    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        try:
            X_scaled = scaler.transform(input_df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0, 1]

            st.write("### Prediction Results:")
            st.write(f"**Predicted Class:** {'Churn' if prediction == 1 else 'No Churn'}")
            st.write(f"**Churn probability:** {probability:.2%}")

        except Exception as e:
            st.error(f"Error when generating the prediction: {e}")
