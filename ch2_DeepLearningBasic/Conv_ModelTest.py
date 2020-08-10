import numpy as np
import pandas as pd
from Model import Model
import Layers
import Optimizer
from ch3_ConvNet.ConvLayers import ConvLayer

# load data
train = pd.read_csv("../Data/MNIST_data/mnist_train.csv")
y_train = train["label"]
x_train = train.drop("label", 1)
x_train = x_train.values / x_train.values.max()
y_train = y_train.values

# y to one-hot
one_hot = np.zeros((y_train.shape[0], y_train.max() + 1))
one_hot[np.arange(y_train.shape[0]), y_train] = 1
y_train = one_hot[:50000]
y_test = one_hot[50000:60000]
x_test = x_train[50000:60000]
x_train = x_train[:50000]

# initialize parameters
num_classes = y_train.shape[1]
hidden = 50
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
print(x_train.shape)

model = Model()
model.addlayer(ConvLayer(32, (3, 3)), name='conv1')
model.addlayer(Layers.ReLU(), name='relu1')
model.addlayer(Layers.Dropout(), name='dropout1')
model.addlayer(Layers.Flatten(), name='flatten')
model.addlayer(Layers.MulLayer(10), name="w1")
model.addlayer(Layers.AddLayer(10), name='b1')
model.addlayer(Layers.ReLU(), name='relu3')
model.addlayer(Layers.Dropout(0.5), name='dropout3')
model.addlayer(Layers.SoftmaxLayer(), name='softmax')

optimizer = Optimizer.Adam(batch_size=32)
model.train(x_train, y_train, optimizer, 10000, 0.01)

model.save()
print("--TRAIN EVAL--")
model.eval(x_train, y_train)
print("--TEST EVAL--")
model.eval(x_test, y_test)
