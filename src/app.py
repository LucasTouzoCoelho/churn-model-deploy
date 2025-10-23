import streamlit as st
import pandas as pd
import joblib
import json
from preprocessing import preprocess_data

# --- Load train artifacts ---
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("src/models/model.pkl")
    return artifacts["model"], artifacts["scaler"], artifacts["features"], artifacts["encoders"]

model, scaler, features, encoders = load_artifacts()

# --- Load JSON file with feature info ---
@st.cache_resource
def load_features_info():
    with open("src/features_info.json") as f:
        return json.load(f)

features_info = load_features_info()

# --- App interface ---
st.title("📊 Churn Prediction App")
st.subheader("🔎 Fill the values to predict churn:")

input_data = {}
for feature in features:
    info = features_info.get(feature, {})

    if info.get("type") == "numeric":
        min_val = float(info.get("min", 0.0))
        max_val = float(info.get("max", 100.0))
        default = float(info.get("mean", (min_val + max_val) / 2))
        input_data[feature] = st.number_input(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=default
        )
    elif info.get("type") == "categorical":
        options = info.get("values", [])
        input_data[feature] = st.selectbox(feature, options)
    else:
        input_data[feature] = st.text_input(feature, value="")

# --- Create DataFrame from user input ---
input_df = pd.DataFrame([input_data])

# --- Prediction ---
if st.button("Predict"):
    try:
        # ✅ Aplica o mesmo pré-processamento usado no treino
        X_processed, _, _, _ = preprocess_data(
            input_df,
            fit_scaler=False,
            scaler=scaler,
            encoders=encoders
        )

        # ✅ Usa diretamente o resultado do preprocessamento
        prediction = model.predict(X_processed)[0]

        # --- Exibir resultados ---
        st.write("### Prediction Results:")
        st.write(f"**Predicted Class:** {'Churn' if prediction == 1 else 'No Churn'}")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_processed)[0, 1]
            st.write(f"**Churn Probability:** {probability:.2%}")
        else:
            st.write("**Churn probability not available.**")

    except Exception as e:
        st.error(f"Error when generating the prediction: {e}")





































# import streamlit as st
# import pandas as pd
# import joblib
# import json
# from preprocessing import preprocess_data

# # --- Load train artifacts ---
# @st.cache_resource
# def load_artifacts():
#     artifacts = joblib.load("models/model.pkl")
#     return artifacts["model"], artifacts["scaler"], artifacts["features"], artifacts["encoders"]

# model, scaler, features, encoders = load_artifacts()

# # --- Load JSON file with feature info ---
# @st.cache_resource
# def load_features_info():
#     with open("features_info.json") as f:
#         return json.load(f)

# features_info = load_features_info()

# # --- App interface ---
# st.title("📊 Churn Prediction App")
# st.subheader("🔎 Fill the values to predict churn:")

# input_data = {}
# for feature in features:
#     info = features_info.get(feature, {})

#     if info.get("type") == "numeric":
#         min_val = float(info.get("min", 0.0))
#         max_val = float(info.get("max", 100.0))
#         default = float(info.get("mean", (min_val + max_val) / 2))
#         input_data[feature] = st.number_input(
#             label=feature,
#             min_value=min_val,
#             max_value=max_val,
#             value=default
#         )
#     elif info.get("type") == "categorical":
#         options = info.get("values", [])
#         input_data[feature] = st.selectbox(feature, options)
#     else:
#         input_data[feature] = st.text_input(feature, value="")

# # --- Create DataFrame from user input ---
# input_df = pd.DataFrame([input_data])

# # --- Prediction ---
# if st.button("Predict"):
#     try:
#         # Pré-processamento, se necessário:
#         input_df_processed, _, _, _ = preprocess_data(input_df, fit_scaler=False, scaler=scaler, encoders=encoders)

#         # Aplicar o scaler nos dados de entrada
#         X_scaled = scaler.transform(input_df[features])
#         prediction = model.predict(X_scaled)[0]

#         # Exibir resultados
#         st.write("### Prediction Results:")
#         st.write(f"**Predicted Class:** {'Churn' if prediction == 1 else 'No Churn'}")

#         if hasattr(model, "predict_proba"):
#             probability = model.predict_proba(X_scaled)[0, 1]
#             st.write(f"**Churn Probability:** {probability:.2%}")
#         else:
#             st.write("**Churn probability not available.**")

#     except Exception as e:
#         st.error(f"Error when generating the prediction: {e}")
