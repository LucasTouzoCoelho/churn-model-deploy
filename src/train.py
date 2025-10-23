# train.py
import os
import joblib
import pandas as pd
from preprocessing import preprocess_data
from model import create_model

# 1. Ler dados
df = pd.read_csv("data/train_dataset.csv")

# 2. Pré-processar com fit do scaler e dos encoders
x, y, scaler, encoders = preprocess_data(df, fit_scaler=True)

# 3. Criar e treinar modelo
model = create_model()
model.fit(x, y)

# 4. Criar pasta de saída se não existir
os.makedirs("src/models", exist_ok=True)

# 5. Salvar artefatos com joblib
joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": list(x.columns),
    "encoders": encoders
}, "src/models/model.pkl")

print(" Modelo treinado e salvo com joblib em src/models/model.pkl")

