from os.path import exists as existsP
import n_layer, pickle as p, numpy as np


class neural_net():
    def __init__(self, topology, path=None):
        self.n_layers = []
        if path != None and existsP(path):
            self.import_nn(path)
        else:
            self.create_nn(topology)

    def create_nn(self, topology):
        self.n_layers = []
        for i, topo in enumerate(topology[:-1]):
            self.n_layers.append(n_layer(topo[0], topology[i+1][0], topo[1]))
        return self.n_layers

    def predict(self, input):
        self.outputs = [input]
        for i in range(0, len(self.n_layers)):
            self.outputs.append(self.n_layers[i].forward(self.outputs[-1]))
        return self.outputs[-1]

    # ver si hacer que train tenga las iteraciones de entrenamiento y usarla como decorador
    def train(self, dataset, fn_cost, lr=0.05):
        for e in range(0, len(dataset)):
            wT = None
            deltas = []
            self.predict(dataset[e][0])
            for i in reversed(range(0, len(self.n_layers))):
                if i == len(self.n_layers) - 1:
                    delta = fn_cost(self.outputs[i+1], dataset[e][1], derivative=True) * self.act_fn(self.outputs[i+1], derivative=True)
                else:
                    delta = np.dot(deltas[0], wT.T) * self.act_fn(self.outputs[i+1], derivative=True)
                
                deltas.insert(0, delta)

                wT = self.n_layers[i].gradient_descent(deltas[0], lr)
            
        return fn_cost(self.outputs[i], dataset[e][1]) #probar su derivada

    def import_nn(self, path):
        with open(path, 'rb') as file:
            self.n_layers = p.load(file)

    def export_nn(self, path):
        with open(path, 'wb') as file:
            p.dump(self.n_layers, file)
