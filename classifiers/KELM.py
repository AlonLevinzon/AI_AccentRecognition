import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KernelELMClassifier(BaseEstimator, ClassifierMixin):
    """
    Kernel Extreme Learning Machine (KELM) custom classifier fully compatible
    with scikit-learn estimators, RandomizedSearchCV, and Pipeline cloning.
    """

    def __init__(self, C=1.0, gamma=0.1):
        self.C = C
        self.gamma = gamma

    def rbf_kernel(self, X1, X2):
        """ Computes RBF Kernel between two matrices """
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        distances_sq = np.maximum(distances_sq, 0.0)  # מניעת ערכים שליליים קטנים מחמת דיוק צף
        return np.exp(-self.gamma * distances_sq)

    def fit(self, X, y):
        """ Trains the Kernel ELM model using analytical linear system solving """
        # Convert to Numpy Array in case of Pandas DataFrame
        self.X_train_ = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        y_arr = y_arr.flatten()

        # Store unique classes
        self.classes_ = np.unique(y_arr)
        num_samples = self.X_train_.shape[0]
        num_classes = len(self.classes_)

        # Create robust One-Hot binary matrix for all label types
        y_label_matrix = np.zeros((num_samples, num_classes), dtype=np.float64)
        for idx, cls_val in enumerate(self.classes_):
            y_label_matrix[y_arr == cls_val, idx] = 1.0

        # Compute Kernel matrix (N x N)
        K = self.rbf_kernel(self.X_train_, self.X_train_)

        # Numerically stable analytical solution with added regularization jitter
        reg_matrix = K + (1.0 / self.C) * np.eye(num_samples) + 1e-7 * np.eye(num_samples)

        try:
            self.beta_ = np.linalg.solve(reg_matrix, y_label_matrix)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.pinv(reg_matrix) @ y_label_matrix

        return self

    def predict(self, X):
        """ Predicts the class with the highest analytical output """
        X_data = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        K_test = self.rbf_kernel(X_data, self.X_train_)
        raw_output = np.dot(K_test, self.beta_)
        class_indices = np.argmax(raw_output, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        """ Predicts class probabilities using softmax over raw outputs """
        X_data = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        K_test = self.rbf_kernel(X_data, self.X_train_)
        raw_output = np.dot(K_test, self.beta_)
        exp_scores = np.exp(raw_output - np.max(raw_output, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)