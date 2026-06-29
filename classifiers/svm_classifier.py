from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def train_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf', gamma='scale', random_state=42):
    """
    Trains a generalized Support Vector Machine (SVM) classifier and evaluates its performance.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature vectors (e.g., 55-dimensional handcrafted acoustic metrics).
    y_train : array-like of shape (n_samples,)
        Target class labels for training.
    X_test : array-like of shape (n_samples, n_features)
        Testing feature vectors.
    y_test : array-like of shape (n_samples,)
        Target class labels for testing.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is inversely proportional to C.
        (Note: Literature suggests higher values like C=128 for optimized accent boundary parsing).
    kernel : str, default='rbf'
        Specifies the kernel type to be used in the algorithm. 'rbf' represents the Gaussian kernel.
    gamma : str or float, default='scale'
        Kernel coefficient for 'rbf'. If 'scale', it uses 1 / (n_features * X.var()) as value of gamma.
    random_state : int, default=42
        Controls the pseudo-random number generation for shuffling the data for probability estimation.
        
    Returns:
    --------
    model : SVC
        The trained scikit-learn Support Vector Classifier instance.
    metrics : dict
        A dictionary containing calculated evaluation metrics: accuracy, precision, recall, f1_score, and confusion_matrix.
    """
    # Initialize the Support Vector Classifier
    # probability=True can be enabled if soft probability vectors are explicitly required for weighted voting schemas
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        random_state=random_state
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
