import os.path as path, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from neural_net.cost_fn import mse
from neural_net.neural_net_v1 import create_nn, train, create_nn, predict, exportNN, importNN
from neural_net.act_fn import relu, sigmo


PORT = 8000
datasetpath = 'models/dataset_bgcolor.pickle'
nnpath = 'models/nn_textcolor.pickle'

INPUTS_NEURS = 3

# [[r, g, b], sg]
DATASET = importNN(datasetpath) if path.exists(datasetpath) else []

# [[INPUTS_NEURS, sigmo], [6, sigmo], [3, sigmo], [1]]
NN = create_nn([[INPUTS_NEURS, sigmo], [8, relu], [4, sigmo], [1]])
print(f"{DATASET}/{NN}")

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def Rpredict():
    data = request.get_json()

    data = np.reshape(data, [1, 3])
    output = predict(NN, data)
    return jsonify({"out":output[-1][0][0]})

@app.route('/train-<samples>-<lr>')
def Rtrain(samples, lr):
    if(len(DATASET) == 0):
        return jsonify({"result": "no examples in dataset"})
    
    samples = int(samples)
    lr = float(lr)
    LOSS = []
    
    print("TRAIN STARTED...")
    
    for e in range(samples):
        print("ROUND:", e)
        for i in range(len(DATASET)):
            output = predict(NN, DATASET[i][0])
            train(NN, output, DATASET[i][1], mse, lr)
        if e % 25 == 0:
            LOSS.append(mse(output[-1], DATASET[-1][1]))
    
    print("TRAIN FINISHED")
    
    return jsonify({"loss": LOSS})

@app.route('/recivedexample', methods=['POST'])
def Rrecived():
    data = request.get_json()
    data = (np.reshape(data[0], [1, 3]), np.reshape(data[1], [1, 1]))

    DATASET.append(data)
    print("EXAMPLE ADDED", data)

    return {"state": "new example add in dataset"}

@app.route('/removelastexample')
def Rremovelast():
    print("EXAMPLE REMOVED", DATASET.pop())

    return {"state": "remove last in dataset"}

@app.route('/save')
def Rsave():
    exportNN(NN, nnpath)
    exportNN(DATASET, datasetpath)

    return {"state": "nn & dataset saved"}

if __name__ == '__main__':
    app.run(debug=True, port=PORT)