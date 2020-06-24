import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd


class PCA_:

    def __init__(self, n_components=None, solver='EVD', power=0.99, whiten=False):
        self.num_comp = n_components    # Number of PC
        self.power = power   # Reserve 99% power
        self.pc_loads = None
        self.solver = solver
        self.whiten = whiten
        self.mean_ = None

    def fit(self, X):
        data_len, data_dim = X.shape
        if self.solver == 'EVD':
            # calculate covariance
            self.mean_ = np.mean(X, axis=0)
            cov_x = np.cov(X.T)
            # Number of PC have been assigned
            if self.num_comp:
                _, V = eigh(cov_x, eigvals=(data_dim-self.num_comp, data_dim-1))
                V = np.fliplr(V)
                self.pc_loads = normalize(V, norm='l2', axis=0) if self.whiten else V
            else:
                # Preserve 99% power
                var = eigh(cov_x, eigvals_only=True)
                power = 0
                k = data_dim - 1
                while power <= 0.99 * sum(var):
                    power += var[k]
                    k -= 1
                _, V = eigh(cov_x, eigvals=(k, data_dim-1))
                V = np.fliplr(V)
                self.pc_loads = normalize(V, norm='l2', axis=0) if self.whiten else V

        elif self.solver == 'SVD':
            self.mean_ = np.mean(X, axis=0)
            X_0 = X - self.mean_
            _, S, VT = randomized_svd(X_0, k=self.num_comp)
            self.pc_loads = normalize(VT.T, norm='l2', axis=0) if self.whiten else VT.T

    def transform(self, data):
        return np.dot(data, self.pc_loads)
