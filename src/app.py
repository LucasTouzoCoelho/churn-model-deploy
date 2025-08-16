import streamlit as st
import joblib
import numpy as np
import json
import os

# ---------------------------
# Função para carregar modelo e scaler
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("src/models/model.pkl")
    scaler = joblib.load("src/models/scaler.pkl")
    with open("src/features_info.json", "r") as f:
        features_info = json.load(f)
    return model, scaler, features_info

# ---------------------------
# App Streamlit
# ---------------------------
st.title("Churn Prediction App")

model, scaler, features_info = load_artifacts()

st.write("Insira os valores para prever se o cliente terá churn:")

user_input = {}

# Criar os inputs dinamicamente
for feature, info in features_info.items():
    if info["type"] == "numeric":
        user_input[feature] = st.number_input(
            f"{feature} (range: {info['min']} - {info['max']})",
            min_value=info["min"],
            max_value=info["max"],
            value=(info["min"] + info["max"]) / 2.0
        )
    elif info["type"] == "categorical":
        user_input[feature] = st.selectbox(
            f"{feature}",
            options=info["values"]
        )

# Botão de previsão
if st.button("Prever"):
    # Criar vetor de features na ordem correta
    features = list(features_info.keys())
    input_array = np.array([[user_input[feat] for feat in features]])

    # Escalar numéricas
    input_scaled = scaler.transform(input_array)

    # Fazer previsão
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Resultado da previsão:")
    st.write("Churn" if prediction == 1 else "Não Churn")
    st.write(f"Probabilidade de churn: {prob:.2f}")
