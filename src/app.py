import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("src/models/model.pkl")
    return artifacts["model"], artifacts["scaler"], artifacts["features"]

model, scaler, features = load_artifacts()

st.title("📊 Churn Prediction App")

# --- Escolha do modo ---
option = st.radio(
    "How would you like to predict?",
    ("CSV Upload", "Manually insert values")
)

# --- Modo 1: Upload de CSV ---
if option == "Upload de CSV":
    uploaded_file = st.file_uploader("Please upload the CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            # Garante que só pega as features que o modelo espera
            X = data[features]
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            data["Churn_Prediction"] = predictions
            data["Churn_Probability"] = probabilities

            st.success("✅ Predictions successfully generated!")
            st.dataframe(data)

            # Permitir download do resultado
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download the results", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

# --- Modo 2: Inserção manual ---
else:
    st.subheader(" Please fill the values to predict a churn")

    # Criar inputs dinamicamente para cada feature
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}", value=0)

    # Converter para DataFrame
    input_df = pd.DataFrame([input_data])

    if st.button("Prever"):
        try:
            X_scaled = scaler.transform(input_df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0, 1]

            st.write("### Prediction Result:")
            st.write(f"**Predicted Class:** {'Churn' if prediction == 1 else 'No Churn'}")
            st.write(f"**Probability of  Churn:** {probability:.2%}")

        except Exception as e:
            st.error(f"Error generating prediction: {e}")
