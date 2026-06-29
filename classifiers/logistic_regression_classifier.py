from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def train_logistic_regression(X_train, y_train, X_test, y_test, C=1.0, max_iter=1000, solver='lbfgs'):
    """
    Trains a generalized Logistic Regression classifier and evaluates its performance.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature vectors (e.g., extracted acoustic feature arrays).
    y_train : array-like of shape (n_samples,)
        Target class labels for training.
    X_test : array-like of shape (n_samples, n_features)
        Testing feature vectors.
    y_test : array-like of shape (n_samples,)
        Target class labels for testing.
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
    max_iter : int, default=1000
        Maximum number of iterations taken for the solvers to converge.
    solver : str, default='lbfgs'
        Algorithm to use in the optimization problem. 'lbfgs' is robust and handles multinomial loss natively.
        
    Returns:
    --------
    model : LogisticRegression
        The trained scikit-learn model instance.
    metrics : dict
        A dictionary containing calculated evaluation metrics: accuracy, precision, recall, f1_score, and confusion_matrix.
    """
    # Initialize the model. 'multi_class=auto' chooses multinomial loss if the data is multi-class
    model = LogisticRegression(
        C=C, 
        max_iter=max_iter, 
        solver=solver, 
        multi_class='auto', 
        random_state=42
    )
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Run predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate global evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Using 'macro' average treats all classes equally, making it ideal for multi-class classification setups
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
