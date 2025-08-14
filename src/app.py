# app.py
import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_data

st.set_page_config(page_title="Previsão de Churn", layout="wide")
st.title("📊 Previsão de Churn de Clientes")

# 1. Carregar artefatos do treino
@st.cache_data
def load_artifacts():
    artifacts = joblib.load("src/models/model.pkl")
    return artifacts["model"], artifacts["scaler"], list(artifacts["features"])

model, scaler, features = load_artifacts()

# 2. Upload do CSV
uploaded_file = st.file_uploader("Escolha um CSV com dados de clientes", type="csv")
if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    st.write("Dados carregados:")
    st.dataframe(df_new.head())

    # 2.1 Validar se todas as colunas necessárias existem
    missing_cols = [col for col in features if col not in df_new.columns]
    if missing_cols:
        st.error(f"Erro: O CSV enviado não contém as colunas necessárias: {missing_cols}")
    else:
        # 3. Pré-processar com o scaler já treinado
        try:
            df_processed, _, _ = preprocess_data(df_new, fit_scaler=False, scaler=scaler)
        except ValueError as e:
            st.error(f"Erro ao processar os dados: {e}")
        else:
            # 4. Garantir que temos apenas as features do treino
            X = df_processed[features]

            # 5. Fazer previsões
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

            # 6. Exibir resultados
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
