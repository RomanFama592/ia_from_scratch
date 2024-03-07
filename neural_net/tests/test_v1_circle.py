import matplotlib.pyplot as plt, numpy as np, sklearn.datasets as ds
from neural_net_v1 import train, create_nn, predict
from act_fn import sigmo 
from cost_fn import mse



fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title("Inputs")
ax2.set_title("Predict")
ax3.set_title("DG")

ax1.axis("equal")
ax2.axis("equal")


# NO FUNCIONA CORRECTAMENTE, VERIFICAR EL MOTIVO

I = 1000
interval = 25
N_SAMPLES = 500
INPUTS_NEURS = 2
LOSS = []
RES = 50
SIZE = 1.5
lr = 0.01


nn = create_nn([[INPUTS_NEURS, sigmo], [4, sigmo], [8, sigmo], [1]])
#nn = importNN("NN.pickle")

for i in range(I):
    print("round: ", i+1)
    #no entiendo que es el X e Y
    x, y = ds.make_circles(N_SAMPLES, factor=0.7, noise=0.06)
    y = y[:, np.newaxis]

    output = predict(nn, x)
    train(nn, output, y, mse, lr)

    if i % interval == 0:
        ax1.clear()
        ax1.set_title("Inputs")
        
        ax2.clear()
        ax2.set_title("Predict")

        LOSS.append(mse(output[-1], y))
        #lr = lr - (lr * 0.25) if LOSS[-1] < LOSS[-2] else lr + (lr * 0.25)

        _x0 = np.linspace(-SIZE, SIZE, RES)
        _x1 = np.linspace(-SIZE, SIZE, RES)
        _y = np.zeros((RES, RES))

        #no lo entiendo del todo
        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _y[i0, i1] = predict(nn, np.array([[x0, x1]]))[-1][0][0]

        ax1.scatter(x[y[:, 0] == 0, 0], x[y[:, 0] == 0, 1], c="skyblue")
        ax1.scatter(x[y[:, 0] == 1, 0], x[y[:, 0] == 1, 1], c="salmon")
        ax2.pcolormesh(_x0, _x1, _y, cmap="coolwarm")

        ax3.plot(range(len(LOSS)), LOSS)

        plt.pause(0.001)
plt.show()
