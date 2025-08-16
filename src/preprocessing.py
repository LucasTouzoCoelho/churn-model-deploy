# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoders = {}

def preprocess_data(df, fit_scaler=False, scaler=None, le=None):
    df_copy = df.copy()
    
    # Limpeza básica
    df_copy['SeniorCitizen'] = df_copy['SeniorCitizen'].astype(str)
    df_copy = df_copy[df_copy['TotalCharges'] != ' ']
    df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'])
    df_copy['PaymentMethod'] = (df_copy['PaymentMethod']
                                .str.replace('(', '', regex=False)
                                .str.replace(')', '', regex=False)
                                .str.replace('automatic', '', regex=False))
    
    # Colunas categóricas (exceto target)
    categorical_columns = [col for col in df_copy.select_dtypes(include=['object']).columns if col != 'Churn']
    
    # Label Encoding
    for col in categorical_columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col])
        label_encoders[col] = le

    # Target
    y = df_copy['Churn'].map({'No': 0, 'Yes': 1}) if 'Churn' in df_copy.columns else None

    # Features para scaler
    features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV',
                'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
                'MonthlyCharges','TotalCharges']

    # Se for treino, criar scaler e dar fit; se for predict, usar o scaler passado
    if fit_scaler:
        scaler = StandardScaler()
        x_scaled = pd.DataFrame(scaler.fit_transform(df_copy[features]), columns=features)
    else:
        if scaler is None:
            raise ValueError("No scaler provided for transform in predict!")
        x_scaled = pd.DataFrame(scaler.transform(df_copy[features]), columns=features)

    return x_scaled, y, scaler, le
