import streamlit as st
import pandas as pd
import joblib
import json
from preprocessing import preprocess_data

# --- Carregar artefatos do treino ---
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("src/models/model.pkl")
    return artifacts["model"], artifacts["scaler"], artifacts["features"]

model, scaler, features = load_artifacts()

# --- Carregar info das features do JSON ---
@st.cache_resource
def load_features_info():
    with open("src/models/features_info.json") as f:
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
            # Pré-processamento
            df_processed, _, _ = preprocess_data(data, fit_scaler=False, scaler=scaler)
            X = df_processed[features]

            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            data["Churn_Prediction"] = predictions
            data["Churn_Probability"] = probabilities.round(3)

            st.success("✅ Previsões geradas com sucesso!")
            st.dataframe(data)

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

        # Input categórico
        if info.get("type") == "categorical":
            options = info.get("values", [])
            input_data[feature] = st.selectbox(f"{feature}", options)

        # Input numérico
        elif info.get("type") == "numeric":
            min_val = info.get("min", 0)
            max_val = info.get("max", 100)
            default = (min_val + max_val)/2
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default)
            )

        # Fallback
        else:
            input_data[feature] = st.text_input(f"{feature}", value="")

    # Converter para DataFrame
    input_df = pd.DataFrame([input_data])

    if st.button("Prever"):
        try:
            # Pré-processamento com o mesmo scaler
            df_processed, _, _ = preprocess_data(input_df, fit_scaler=False, scaler=scaler)
            X = df_processed[features]

            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0, 1]

            st.write("### Resultado da Previsão:")
            st.write(f"**Classe prevista:** {'Churn' if prediction == 1 else 'Não Churn'}")
            st.write(f"**Probabilidade de Churn:** {probability:.2%}")

        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")
