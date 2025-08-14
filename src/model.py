from sklearn.ensemble import RandomForestClassifier

def create_model():
    """Cria o modelo de Machine Learning."""
    return RandomForestClassifier(bootstrap=True,
 max_depth= 20,
 min_samples_leaf= 1,
 min_samples_split= 2,
 random_state=42,
 n_estimators= 200)

