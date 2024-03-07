import numpy as np

EPSILON = 1e-8

# Mean Squared Error


def mse(y_pred, y_true, derivative=False):
    if derivative:
        return (y_pred - y_true)
    return np.mean(np.square(y_pred - y_true))
