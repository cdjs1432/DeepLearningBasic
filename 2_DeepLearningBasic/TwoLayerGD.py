import numpy as np
import pandas as pd
import Layers

# load data
train = pd.read_csv("../Data/MNIST_data/mnist_train.csv")
y_train = train["label"]
x_train = train.drop("label", 1)
x_train = x_train.values / x_train.values.max()
y_train = y_train.values

# y to one-hot
one_hot = np.zeros((y_train.shape[0], y_train.max() + 1))
one_hot[np.arange(y_train.shape[0]), y_train] = 1
y_train = one_hot

# initialize parameters
num_classes = y_train.shape[1]
hidden = 50

keys = ['w1', 'b1', 'sigmoid', 'w2', 'b2']
params = {}
params['w1'] = np.random.uniform(-1, 1, (x_train.shape[1], hidden))
params['b1'] = np.random.uniform(-1, 1, (hidden))
params['w2'] = np.random.uniform(-1, 1, (hidden, num_classes))
params['b2'] = np.random.uniform(-1, 1, (num_classes))

layers = {}
layers['w1'] = Layers.MulLayer(params['w1'])
layers['b1'] = Layers.AddLayer(params['b1'])
layers['sigmoid'] = Layers.SigmoidLayer()
layers['w2'] = Layers.MulLayer(params['w2'])
layers['b2'] = Layers.AddLayer(params['b2'])
lastlayer = Layers.SoftmaxLayer()

grads = {}

# initialize hyperparameters
learning_rate = 0.01
epochs = 10000
batch_size = 512

for epoch in range(epochs):
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x = x_train[batch_mask]
    y = y_train[batch_mask]

    for key in keys:
        x = layers[key].forward(x)

    loss = lastlayer.forward(x, y)

    if epoch % (epochs / 10) == 0:
        pred = Layers.softmax(x)
        print("ACC on epoch %d : " % epoch, (pred.argmax(1) == y.argmax(1)).mean())
        print("LOSS on epoch %d : " % epoch, loss)

    dout = lastlayer.backward()

    for key in reversed(keys):
        dout = layers[key].backward(dout)
        if key != 'sigmoid':
            grads[key] = layers[key].grad
            params[key] -= learning_rate * grads[key]

# load data
test = pd.read_csv("../Data/MNIST_data/mnist_test.csv")
y_test = test["label"]
x_test = test.drop("label", 1)
x_test = x_test.values / x_test.values.max()
y_test = y_test.values

# y to one-hot
one_hot = np.zeros((y_test.shape[0], y_test.max() + 1))
one_hot[np.arange(y_test.shape[0]), y_test] = 1
y_test = one_hot

x = x_test
for key in keys:
    x = layers[key].forward(x)
pred = Layers.softmax(x)
print("ACC : ", (pred.argmax(1) == y_test.argmax(1)).mean())
