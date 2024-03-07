import numpy as np
import matplotlib.pyplot as plt

# Sigmoid


def sigmo(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Step function


def stepfn(x):
    return np.where(x >= 0, 1, 0)

# ReLU


def relu(x, derivative=False):
    if derivative:
        return stepfn(x)
    return np.maximum(0, x)

# Leaky ReLu(if a remains unchanged) and Parametric ReLu


def lprelu(x, a=0.01, derivative=False):
    if derivative:
        return np.where(x < 0, a, 1)

    return np.maximum(a * x, x)


# Gaussian ReLu


def grelu(x, a=0.1, derivative=False):
    if derivative:
        return stepfn(x)
    noise = np.random.normal(loc=0, scale=a, size=x.shape)
    return np.maximum(0, x + noise)

# Hyperbolic tangent


def htan(x, derivative=False):
    if derivative:
        ...
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Softplus


def softplus(x, derivative=False):
    if derivative:
        return sigmo(x)
    return np.log(1 + np.exp(x))

# Softmax


def softmax(x, derivative=False):
    if derivative:
        ...
    x_max = np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=-1, keepdims=True)
    return x_exp / x_sum

# Swish


def swish(x, a=1, derivative=False):
    if derivative:
        exp_x = np.exp(x)
        return (a * exp_x) / ((1 + exp_x) ** 2) + (x * a * exp_x) / (1 + exp_x)
    return x * sigmo(a*x)

# Mish


def mish(x, derivative=False):
    if derivative:
        exp_x = np.exp(x)
        exp_2x = np.exp(2 * x)
        num = (4 * exp_2x) + (4 * x * exp_x) + exp_x
        den = (exp_2x + 2 * x * exp_x + 2) ** 2
        return exp_x * num / den
    return x * htan(softplus(x))

"""def linspaceME(start, stop, sample):
    space = (stop - start) / (sample - 1)
    spaceACUM = start

    array = []

    for i in range(sample):
        array.append(spaceACUM)
        spaceACUM += space

    return np.array(array)"""

if __name__ == "__main__":
    x = np.linspace(-5, 5, 1000)
    plt.plot(x)
    # plt.axis("equal")
    plt.show()
