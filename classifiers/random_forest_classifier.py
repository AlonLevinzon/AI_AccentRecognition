from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
    """
    Trains a generalized Random Forest classifier and evaluates its performance.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature vectors (e.g., flattened 55-dimensional acoustic arrays).
    y_train : array-like of shape (n_samples,)
        Target class labels for training (e.g., accent string or integer IDs).
    X_test : array-like of shape (n_samples, n_features)
        Testing feature vectors.
    y_test : array-like of shape (n_samples,)
        Target class labels for testing.
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int or None, default=None
        The maximum depth of the trees. If None, nodes are expanded until leaves are pure.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    random_state : int, default=42
        Controls the randomness of bootstrapping and feature selection for reproducibility.
        
    Returns:
    --------
    model : RandomForestClassifier
        The trained scikit-learn Random Forest model instance.
    metrics : dict
        A dictionary containing calculated evaluation metrics: accuracy, precision, recall, f1_score, and confusion_matrix.
    """
    # Initialize the Random Forest Classifier
    # n_jobs=-1 ensures the model utilizes all available CPU cores for parallel tree building
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Run predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate global evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Using 'macro' average treats all classes equally, making it ideal for multi-class setups
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, 
        y_pred, 
        average='macro', 
        zero_division=0
    )
    
    # Generate the raw confusion matrix array
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }
    
    return model, metrics
