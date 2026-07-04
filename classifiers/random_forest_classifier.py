from sklearn.ensemble import RandomForestClassifier

def get_random_forest_classifier(n_estimators=100, max_depth=None, random_state=42):
    """
    Returns an initialized Random Forest Classifier instance matching the unified Pipeline API.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Utilizes all available CPU cores for parallel execution
    )