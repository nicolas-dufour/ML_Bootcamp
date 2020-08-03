import numpy as np
import plotly.express as px
from src.utils import plot_regression


class RegressionDataset:
    def __init__(self, theta, x_min, x_max):
        self.theta = theta
        self.n_dimension = theta.shape[0] - 1
        self.x_min, self.x_max = x_min, x_max

    def linear(self, n, sigma):
        X = np.concatenate(
            (np.ones((n, 1)),
             np.random.uniform(self.x_min, self.x_max, (n, self.n_dimension))),
            axis=1)
        y = X@self.theta + sigma*np.random.randn(n)
        return X, y

    def polymomial(self, n, sigma):
        x0 = np.random.uniform(self.x_min, self.x_max, (n, 1))
        X = np.ones((n, 1))
        for i in range(1, self.n_dimension+1):
            X = np.concatenate((X, x0**i), axis=1)
        y = X@self.theta + sigma*np.random.randn(n)
        return X, y


theta = np.array([1, 0.5])
rdata = RegressionDataset(theta, -10, 10)
X, y = rdata.linear(100, 1)
px.scatter(x=X[:,1], y=y)


class LSRegression:
    def __init__(self):
        self.theta = np.nan


get_ipython().run_cell_magic("time", "", """ls = LSRegression()
ls.fit(X, y)
y_hat = ls.predict(X)
plot_regression(X[:, 1], y, y_hat)""")


class LSRegression_GD:
    def __init__(self, d):
        self.theta = np.random.randn(d)
