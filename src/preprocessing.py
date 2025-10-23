# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, fit_scaler=False, scaler=None, encoders=None):
    df_copy = df.copy()
    
    # Limpeza básica
    df_copy['SeniorCitizen'] = df_copy['SeniorCitizen'].astype(str)
    df_copy = df_copy[df_copy['TotalCharges'] != ' ']
    df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'])
    df_copy['PaymentMethod'] = (df_copy['PaymentMethod']
                                .str.replace('(', '', regex=False)
                                .str.replace(')', '', regex=False)
                                .str.replace('automatic', '', regex=False))
    
    # Detecta colunas categóricas (exceto 'Churn')
    categorical_columns = [col for col in df_copy.select_dtypes(include=['object']).columns if col != 'Churn']

    if fit_scaler:
        encoders = {}

    # Aplica ou carrega mapeamentos (encoders)
    for col in categorical_columns:
        if fit_scaler:
            uniques = df_copy[col].unique()
            encoders[col] = {val: idx for idx, val in enumerate(uniques)}
        df_copy[col] = df_copy[col].map(encoders[col])
    
    # Target
    y = df_copy['Churn'].map({'No': 0, 'Yes': 1}) if 'Churn' in df_copy.columns else None

    # Features selecionadas
    features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges']

    # Escala os dados
    if fit_scaler:
        scaler = StandardScaler()
        x_scaled = pd.DataFrame(scaler.fit_transform(df_copy[features]), columns=features)
    else:
        if scaler is None or encoders is None:
            raise ValueError("Scaler or encoders not provided.")
        x_scaled = pd.DataFrame(scaler.transform(df_copy[features]), columns=features)

    return x_scaled, y, scaler, encoders
