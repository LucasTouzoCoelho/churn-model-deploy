import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# --- Carregar artefatos do treino ---
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("src/models/model.pkl")
    return artifacts["model"], artifacts["scaler"], artifacts["features"]

model, scaler, features = load_artifacts()

# --- Carregar info das features do JSON ---
@st.cache_resource
def load_features_info():
    with open("src/features_info.json") as f:
        return json.load(f)

features_info = load_features_info()

st.title("📊 Churn Prediction App")

# --- Escolha do modo ---
option = st.radio(
    "Como você gostaria de prever?",
    ("Upload de CSV", "Inserir valores manualmente")
)

# --- Modo 1: Upload de CSV ---
if option == "Upload de CSV":
    uploaded_file = st.file_uploader("Faça upload de um arquivo CSV", type=["csv"])

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

            st.success("✅ Previsões geradas com sucesso!")
            st.dataframe(data)

            # Permitir download do resultado
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar resultados", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

# --- Modo 2: Inserção manual ---
else:
    st.subheader("🔎 Preencher valores para prever um cliente")

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

    if st.button("Prever"):
        try:
            # Para features categóricas com LabelEncoder, aplicar codificação
            for feature in features:
                if features_info.get(feature, {}).get("type") == "categorical":
                    # Assumindo que LabelEncoders estão salvos no modelo, ou você pode aplicar manualmente
                    # Aqui deixamos como string, se necessário ajustar depois
                    pass

            X_scaled = scaler.transform(input_df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0, 1]

            st.write("### Resultado da Previsão:")
            st.write(f"**Classe prevista:** {prediction}")
            st.write(f"**Probabilidade de Churn:** {probability:.2%}")

        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")
