from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier

def get_svm_classifier(C=1.0, kernel='rbf', gamma='scale', random_state=42):
    """
    Returns an SVM classifier wrapped in a OneVsOneClassifier strategy
    for optimal multiclass accent boundary splitting.
    """
    base_svc = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,  # Enables soft probability vectors if needed
        random_state=random_state
    )
    # Wrapping with OneVsOne handles multiclass text/accent structures robustly
    return OneVsOneClassifier(base_svc)