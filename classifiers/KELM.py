import numpy as np


class KernelELMClassifier:
    def __init__(self, C=1.0, gamma=0.1):
        """
        C regulazation paramater
        gamma: width paramater RBF Kernel (1 / (2 * sigma^2))
        """
        self.C = C
        self.gamma = gamma
        self.X_train = None
        self.beta = None
        self.n_classes = None

    def rbf_kernel(self, X1, X2):
        """  RBF Kernel between two samples matrics """
        #algebraic expansion of squared Euclidean distance
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * distances_sq)

    def convert_to_label_matrix(self, y):
        """ turn labels to matrix of [1. 0. 0.] etc...so we can multiplay with  """
        classes = np.unique(y)
        self.n_classes = len(classes)
        one_hot = np.zeros((len(y), self.n_classes))
        for idx, val in enumerate(y):
            one_hot[idx, val] = 1
        return one_hot

    def fit(self, X, y):
        """ training the model """
        self.X_train = X
        y_label_matrix = self.convert_to_label_matrix(y)
        num_samples = X.shape[0]

        # 1. (N x N) kernel matrix for training
        K = self.rbf_kernel(X, X)

        #  we calculate beta from  y_pred=K_test * beta
        # beta = (K + I/C)^(-1) * Y
        identity_matrix = np.eye(num_samples)
        self.beta = np.linalg.solve(K + identity_matrix / self.C, y_label_matrix)
        return self

    def predict(self, X):
        # kernel calculation between X data and X we were training on
        K_test = self.rbf_kernel(X, self.X_train)

        #raw output from   y_pred=K_test * beta
        raw_output = np.dot(K_test, self.beta)

        # taking the argmax with the highest value
        return np.argmax(raw_output, axis=1)