import matplotlib.pyplot as plt, sklearn.datasets as ds, numpy as np
from neural_net_v2 import neural_net
from cost_fn import mse
from act_fn import sigmo


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title("Inputs")
ax2.set_title("Predict")
ax3.set_title("DG")

ax1.axis("equal")
ax2.axis("equal")

RES = 50
SIZE = 1.5
loss = []
dataset = []
nn = neural_net([[2, sigmo], [4, sigmo], [8, sigmo], [1]])


for i in range(1000):
    print(i)
    x, y = ds.make_circles(500, factor=0.7, noise=0.06)
    y = y[:, np.newaxis]
    dataset.append(np.array(x, y))

    ll = nn.train(dataset, fn_cost=mse)

    if i % 25 == 0:
        ax1.clear()
        ax1.set_title("Inputs")
        
        ax2.clear()
        ax2.set_title("Predict")

        loss.append(ll)
                
        _x0 = np.linspace(-SIZE, SIZE, RES)
        _x1 = np.linspace(-SIZE, SIZE, RES)
        _y = np.zeros((RES, RES))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _y[i0, i1] = nn.predict(np.array([[x0, x1]]))[-1][0][0]

        ax1.pcolormesh(_x0, _x1, _y, cmap="coolwarm")
        ax1.scatter(x[y[:, 0] == 0, 0], x[y[:, 0] == 0, 1], c="skyblue")
        ax1.scatter(x[y[:, 0] == 1, 0], x[y[:, 0] == 1, 1], c="salmon")

        ax2.plot(range(len(loss)), loss)

        plt.pause(0.001)
plt.show()