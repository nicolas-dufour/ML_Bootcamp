import numpy as np
from sklearn import datasets
from src.utils import biplot, plot_pca


iris = datasets.load_iris()
X = iris['data']
y = iris['target']
y_name = iris['target_names'][iris['target']]
feature_names = iris['feature_names']


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean = #Mean
        X_tilde = X-self.mean
        ### Compute the needed transformations
        self.components = #Components
        self.explained_variance = # vector of the explained variance of  each compoasant

    def transform(self, X):
        X_tilde = X-self.mean
        ### Compute the PCA of the given data
        return X_red

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


### code goes here


### code goes here


### code goes here


digits = datasets.load_digits()
X = digits['data']
y = digits['target']
y_name = digits['target_names'][digits['target']]


### code goes here


### code goes here
