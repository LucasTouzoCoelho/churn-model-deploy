# app.py
import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_data
import json

st.set_page_config(page_title="Previsão de Churn", layout="wide")
st.title("📊 Previsão de Churn de Clientes")

# 1️⃣ Carregar artefatos do treino
@st.cache_data
def load_artifacts():
    artifacts = joblib.load("src/models/model.pkl")
    model = artifacts["model"]
    features = artifacts["features"]
    features_info = artifacts.get("features_info", None)
    return model, features, features_info

model, features, features_info = load_artifacts()

# 2️⃣ Seleção do modo de input
input_mode = st.radio("Como você quer fornecer os dados?", ("Upload CSV", "Inserção Manual"))

if input_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Escolha um CSV com dados de clientes", type="csv")
    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        st.write("Dados carregados:")
        st.dataframe(df_new.head())

        try:
            df_processed, _, _ = preprocess_data(df_new, fit_scaler=False)
        except ValueError as e:
            st.error(f"Erro ao processar os dados: {e}")
        else:
            X = df_processed[features]
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

            st.subheader("Resultados das Previsões")
            results = pd.DataFrame({
                "customerID": df_new.get("customerID", range(len(predictions))),
                "Churn_Prediction": predictions
            })
            if probabilities is not None:
                results["Churn_Probability"] = probabilities.round(3)

            st.dataframe(results)
            st.download_button(
                label="⬇️ Baixar previsões",
                data=results.to_csv(index=False),
                file_name="churn_predictions_app.csv",
                mime="text/csv"
            )

elif input_mode == "Inserção Manual":
    st.subheader("Preencha os dados do cliente para previsão")
    manual_input = {}
    if features_info is not None:
        for feature, info in features_info.items():
            if info["type"] == "categorical":
                manual_input[feature] = st.selectbox(feature, info["values"])
            else:
                manual_input[feature] = st.number_input(
                    feature,
                    min_value=info.get("min", 0.0),
                    max_value=info.get("max", 100.0),
                    value=info.get("mean", 0.0)
                )

        if st.button("Prever Churn"):
            df_manual = pd.DataFrame([manual_input])
            df_processed, _, _ = preprocess_data(df_manual, fit_scaler=False)
            X_manual = df_processed[features]
            pred = model.predict(X_manual)[0]
            prob = model.predict_proba(X_manual)[:, 1][0] if hasattr(model, "predict_proba") else None

            st.write(f"**Previsão de Churn:** {pred}")
            if prob is not None:
                st.write(f"**Probabilidade de Churn:** {prob:.3f}")
