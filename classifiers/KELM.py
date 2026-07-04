import numpy as np


class KernelELMClassifier:
    """
    Kernel Extreme Learning Machine custom classifier optimized for matrix operations.
    """

    def __init__(self, C=1.0, gamma=0.1):
        self.C = C
        self.gamma = gamma
        self.X_train = None
        self.beta = None
        self.n_classes = None

    def rbf_kernel(self, X1, X2):
        """ Computes RBF Kernel between two matrices """
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * distances_sq)

    def convert_to_label_matrix(self, y):
        """ Converts integer array labels to a One-Hot binary matrix """
        # Force conversion to integer indices (0, 1, 2...) for array indexing
        y_int = y.astype(int)
        self.n_classes = len(np.unique(y_int))
        one_hot = np.zeros((len(y_int), self.n_classes))
        for idx, val in enumerate(y_int):
            one_hot[idx, val] = 1.0
        return one_hot

    def fit(self, X, y):
        """ Trains the Kernel ELM model using analytical linear system solving """
        # Convert X to a numpy array if it comes as a Pandas DataFrame
        self.X_train = X.values if hasattr(X, 'values') else X
        y_label_matrix = self.convert_to_label_matrix(y)
        num_samples = self.X_train.shape[0]

        # Calculate the kernel matrix (N x N)
        K = self.rbf_kernel(self.X_train, self.X_train)
        identity_matrix = np.eye(num_samples)

        # Analytically solve for weights beta: (K + I/C) * beta = Y_matrix
        self.beta = np.linalg.solve(K + identity_matrix / self.C, y_label_matrix)
        return self

    def predict(self, X):
        """ Predicts the class with the highest analytical output """
        X_data = X.values if hasattr(X, 'values') else X
        K_test = self.rbf_kernel(X_data, self.X_train)
        raw_output = np.dot(K_test, self.beta)
        return np.argmax(raw_output, axis=1)