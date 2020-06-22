import numpy as np
from scipy.linalg import eigh

class PCA_:
    def __init__(self, num_comp=None):
        self.num_comp = num_comp
        self.pc_loads = None

    def fit(self, X):
        data_len, data_dim = X.shape

        # column centred
        X -= np.mean(X, axis=0)
        cov_x = np.cov(np.transpose(X))
        S, V = eigh(cov_x, eigvals=(data_dim-self.num_comp, data_dim-1))
        self.pc_loads = np.fliplr(V)

    def transform(self, X):
        return np.dot(X, self.pc_loads)
