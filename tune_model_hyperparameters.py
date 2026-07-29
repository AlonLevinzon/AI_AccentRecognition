import numpy as np
from sklearn.model_selection import RandomizedSearchCV


def tune_model_hyperparameters(model_name, X_train, y_train, n_iter=10, cv_folds=3):
    """
    Executes an efficient hyperparameter optimization routine using RandomizedSearchCV
    strictly inside the training boundaries of the current cross-validation fold.

    Parameters:
    -----------
    model_name : str
        The identifier of the classifier to tune ('SVM (RBF)', 'Random Forest', or 'XGBoost').
    X_train : np.ndarray
        Standardized training matrix containing only the Top-K selected acoustic features.
    y_train : np.ndarray
        Ground-truth target labels for the training set.
    n_iter : int, default=10
        Number of parameter settings sampled (controls search time vs. optimization quality).
    cv_folds : int, default=3
        Internal Cross-Validation folds for hyperparameter evaluation.

    Returns:
    --------
    best_estimator : object
        The optimized classifier fitted with the best hyperparameter configuration.
    """
    if model_name == 'SVM (RBF)':
        from sklearn.svm import SVC
        base_model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10, 50, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }

    elif model_name == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

    elif model_name == 'XGBoost':
        from xgboost import XGBClassifier
        base_model = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }

    else:
        # Fallback for models that do not require complex tuning (e.g., Gaussian NB, Logistic Regression)
        return None

    # Instantiate fast randomized search over parameter space
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Fit search strictly on training data
    search.fit(X_train, y_train)

    return search.best_estimator_