import pandas as pd
import joblib
from preprocessing import preprocess_data

# 1. Ler dados de teste
df_new = pd.read_csv("data/new_data.csv")

# 2. Carregar modelo e scaler
artifacts = joblib.load("models/model.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]  # usar o scaler fitado do treino

# 3. Pré-processar os dados (aplicar transform)
df_processed, _ , _= preprocess_data(df_new, fit_scaler=False, scaler=scaler)  # fit_scaler=False porque o scaler já está fitado

# 4. Prever
predictions = model.predict(df_processed)

# Se o modelo suportar probabilidades, calcular
probability = None
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(df_processed)[:, 1]

# Adicionar previsões ao DataFrame original
df_new["prediction"] = predictions
if probability is not None:
    df_new["probability"] = probability

# Salvar no CSV
df_new.to_csv("data/predictions.csv", index=False)
print(f"✅ Previsões salvas")

