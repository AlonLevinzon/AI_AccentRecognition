from xgboost import XGBClassifier

def get_xgboost_classifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42):
    """
    Returns an initialized XGBoost Classifier instance optimized for tabular acoustic
    features with constraints to prevent overfitting.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,          # Controlled depth protects against recording noise memorization
        learning_rate=learning_rate,
        objective='multi:softprob',   # Configured for multiclass probability outputs
        eval_metric='mlogloss',
        random_state=random_state
    )