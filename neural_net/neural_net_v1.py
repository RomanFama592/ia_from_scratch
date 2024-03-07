from n_layer import n_layer
import pickle as p, numpy as np, sklearn.datasets as ds
from cost_fn import mse
from act_fn import sigmo 



# input format: [[neur, act_fn], [neur, act_fn], [neurOutput, act_fn]]
def create_nn(topology):
    nn = []
    for i, topo in enumerate(topology[:-1]):
        nn.append(n_layer(topo[0], topology[i+1][0], topo[1]))
    return nn


def predict(nn, input):
    act_fn_products = [input]
    for i in range(0, len(nn)):
        # print(i, "F")
        sum_pon = np.dot(act_fn_products[-1], nn[i].weights) + nn[i].bias
        act_fn_products.append(nn[i].act_fn(sum_pon))
    return act_fn_products


def train(nn, output_neurs, output_true, fn_cost, lr=0.5):
    deltas = []
    for i in reversed(range(0, len(nn))):
        print(nn[i].weights.shape)
        print(output_neurs[i].shape)
        # print(i, "B")
        if i == len(nn) - 1:
            delta = fn_cost(output_neurs[i+1], output_true, derivative=True) * nn[i].act_fn(output_neurs[i+1], derivative=True)
        else:
            delta = np.dot(deltas[0], wT.T) * nn[i].act_fn(output_neurs[i+1], derivative=True)
        deltas.insert(0, delta)

        wT = nn[i].weights

        nn[i].bias -= np.mean(deltas[0], axis=0, keepdims=True) * lr
        nn[i].weights -= np.dot(output_neurs[i].T, deltas[0]) * lr


# extension .pickle

def exportNN(nn, path):
    with open(path, 'wb') as file:
        p.dump(nn, file)


def importNN(path):
    with open(path, 'rb') as file:
        return p.load(file)
    
if __name__ == "__main__":
    x, y = ds.make_circles(500, factor=0.7, noise=0.06)
    y = y[:, np.newaxis]

    nn = create_nn([[2, sigmo], [4, sigmo], [8, sigmo], [1]])


    output = predict(nn, x)
    train(nn, output, y, mse, 0.05)