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
hidden = 50

model = Model()
model.addlayer(Layers.MulLayer(100, initializer='he'), name="w1")
model.addlayer(Layers.AddLayer(100), name='b1')
model.addlayer(Layers.ReLU(), name='relu1')
model.addlayer(Layers.Dropout(), name='dropout1')
model.addlayer(Layers.MulLayer(100, initializer='he'), name="w2")
model.addlayer(Layers.AddLayer(100), name='b2')
model.addlayer(Layers.ReLU(), name='relu2')
model.addlayer(Layers.Dropout(), name='dropout2')
model.addlayer(Layers.MulLayer(100, initializer='he'), name="w3")
model.addlayer(Layers.AddLayer(100), name='b3')
model.addlayer(Layers.ReLU(), name='relu3')
model.addlayer(Layers.Dropout(), name='dropout3')
model.addlayer(Layers.MulLayer(10, initializer='he'), name="w4")
model.addlayer(Layers.AddLayer(10), name='b4')
model.addlayer(Layers.ReLU(), name='relu4')
model.addlayer(Layers.Dropout(), name='dropout4')
model.addlayer(Layers.SoftmaxLayer(), name='softmax')

optimizer = Optimizer.Adam(batch_size=128)
model.train(x_train, y_train, optimizer, 10000, 0.01)

model.save()

print("--TEST EVAL--")
model.eval(x_test, y_test)

print("--Predict--")
print(y_test[2])
pred = model.predict(x_test[2].reshape(1, *x_test[2].shape))
print(pred)