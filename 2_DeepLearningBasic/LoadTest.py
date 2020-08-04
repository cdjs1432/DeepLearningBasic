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
y_train = one_hot[:50000]
y_test = one_hot[50000:60000]
x_test = x_train[50000:60000]
x_train = x_train[:50000]

# initialize parameters
num_classes = y_train.shape[1]

model = Model()
model.load('model.txt')
optimizer = Optimizer.Adam(batch_size=128)
model.train(x_train, y_train, optimizer, 1000, 0.01, True)

print(model.predict(x_train[1]))
print(y_train[1])
