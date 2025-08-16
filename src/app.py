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

    # Carregar informações das features
    import json
    with open("src/features_info.json") as f:
        features_info = json.load(f)

    input_data = {}

    for feature in features:
        info = features_info.get(feature, {})

        # Se for feature categórica (lista de opções)
        if "options" in info:
            options = info["options"]
            default = options[0] if options else None
            input_data[feature] = st.selectbox(f"{feature}", options, index=0)
        # Se for feature numérica
        else:
            min_val = info.get("min")
            max_val = info.get("max")
            default = info.get("mean")

            # Garantir que são números válidos
            min_val = float(min_val) if min_val is not None else 0.0
            max_val = float(max_val) if max_val is not None else 100.0
            default = float(default) if default is not None else (min_val + max_val)/2

            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=default
            )

    # Converter para DataFrame
    input_df = pd.DataFrame([input_data])

    if st.button("Prever"):
        try:
            # Aplicar transformações (LabelEncoder + StandardScaler)
            # Para isso, você precisará reaplicar a mesma lógica do preprocess_data
            from preprocessing import preprocess_data
            df_processed, _, _ = preprocess_data(input_df, fit_scaler=False, scaler=scaler)

            # Selecionar apenas as features do modelo
            X_scaled = df_processed[features]

            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0, 1]

            st.write("### Resultado da Previsão:")
            st.write(f"**Classe prevista:** {'Churn' if prediction == 1 else 'Não Churn'}")
            st.write(f"**Probabilidade de Churn:** {probability:.2%}")

        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")

