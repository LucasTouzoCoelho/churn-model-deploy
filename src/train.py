# train.py
import joblib
import pandas as pd
from preprocessing import preprocess_data
from model import create_model

# 1. Ler dados
df = pd.read_csv("data/dataset.csv")

# 2. Pré-processar (com fit do scaler)
features, target, scaler, le = preprocess_data(df, fit_scaler=True)

# 3. Separar X e y
x = features
y = target

# 4. Criar e treinar modelo
model = create_model()
model.fit(x, y)

# 5. Salvar artefatos (modelo + scaler + features)
joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": features,
    "le":le
}, "models/model.pkl")

print("✅ Modelo treinado e salvo em models/model.pkl")

