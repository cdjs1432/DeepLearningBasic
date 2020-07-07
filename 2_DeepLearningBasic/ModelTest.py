import numpy as np
import pandas as pd
from Model import Model
import Layers
import Optimizer

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

model = Model()
model.addlayer(Layers.MulLayer(), input_size=(784, 32), name="w1")
model.addlayer(Layers.AddLayer(), input_size=32, name='b1')
model.addlayer(Layers.SigmoidLayer(), activation=True, name='sigmoid1')
model.addlayer(Layers.MulLayer(), input_size=(32, 10), name="w2")
model.addlayer(Layers.AddLayer(), input_size=10, name='b2')
model.addlayer(Layers.SoftmaxLayer(), activation=True, name='softmax')

optimizer = Optimizer.Adam(batch_size=128)
model.train(x_train, y_train, optimizer, 10000, 0.01)
