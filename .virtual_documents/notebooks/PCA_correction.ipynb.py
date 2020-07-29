import numpy as np
from sklearn import datasets
from ..src.utils import biplot, plot_pca


iris = datasets.load_iris()
X = iris['data']
y = iris['target']
y_name = iris['target_names'][iris['target']]
feature_names = iris['feature_names']


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean = X.mean(axis=0)
        X_tilde = X-self.mean
        self.U, self.d, self.V = np.linalg.svd(X_tilde, full_matrices=False)
        self.components = self.V[:self.n_components]
        eigen = self.d**2
        self.explained_variance = eigen[:self.n_components]/eigen.sum()

    def transform(self, X):
        X_tilde = X-self.mean
        X_trans = X_tilde@self.components.T
        return X_trans

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


pca = PCA(2)
pca.fit(X)
X_reduced = pca.fit_transform(X)
plot_pca(X_reduced,y,y_name)


biplot(X_reduced, y,y_name, pca.components, feature_names)


pca.explained_variance


digits = datasets.load_digits()
X = digits['data']
y = digits['target']
y_name = digits['target_names'][digits['target']]


X.shape


pca = PCA(2)
pca.fit(X)
X_reduced = pca.fit_transform(X)
plot_pca(X_reduced, y, y_name)
