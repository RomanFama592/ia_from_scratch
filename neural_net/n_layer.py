import numpy as np


class n_layer:
    def __init__(self, n_neur_previous, n_neur, act_fn):
        self.bias = np.random.rand(1, n_neur) * 2 - 1
        self.weights = np.random.rand(n_neur_previous, n_neur) * 2 - 1
        self.act_fn = act_fn

    def forward(self, input):
        self.input = input
        # el output se rompe
        sum_pon = np.dot(input, self.weights) + self.bias
        self.output = self.act_fn(sum_pon)
        
        return self.output

    def gradient_descent(self, delta, lr):
        wT = self.weights

        self.bias -= np.mean(delta, axis=0, keepdims=True) * lr
        self.weights -= np.dot(self.output.T, delta) * lr

        return wT
