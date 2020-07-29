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
y_train = one_hot[:300]
y_test = one_hot[300:600]
x_test = x_train[300:600]
x_train = x_train[:300]

# initialize parameters
num_classes = y_train.shape[1]
hidden = 50

model = Model()
model.addlayer(Layers.MulLayer(), input_size=(784, 100), name="w1", initialization='he', reg=0.01)
model.addlayer(Layers.AddLayer(), input_size=100, name='b1')
model.addlayer(Layers.ReLU(), activation=True, name='relu1')
model.addlayer(Layers.Dropout(), activation=True, name='dropout1')
model.addlayer(Layers.MulLayer(), input_size=(100, 100), name="w2", initialization='he', reg=0.01)
model.addlayer(Layers.AddLayer(), input_size=100, name='b2')
model.addlayer(Layers.ReLU(), activation=True, name='relu2')
model.addlayer(Layers.Dropout(), activation=True, name='dropout2')
model.addlayer(Layers.MulLayer(), input_size=(100, 100), name="w3", initialization='he', reg=0.01)
model.addlayer(Layers.AddLayer(), input_size=100, name='b3')
model.addlayer(Layers.ReLU(), activation=True, name='relu3')
model.addlayer(Layers.Dropout(), activation=True, name='dropout3')
model.addlayer(Layers.MulLayer(), input_size=(100, 100), name="w4", initialization='he', reg=0.01)
model.addlayer(Layers.AddLayer(), input_size=100, name='b4')
model.addlayer(Layers.ReLU(), activation=True, name='relu4')
model.addlayer(Layers.Dropout(), activation=True, name='dropout4')
model.addlayer(Layers.MulLayer(), input_size=(100, 100), name="w5", initialization='he', reg=0.01)
model.addlayer(Layers.AddLayer(), input_size=100, name='b5')
model.addlayer(Layers.ReLU(), activation=True, name='relu5')
model.addlayer(Layers.Dropout(), activation=True, name='dropout5')
model.addlayer(Layers.MulLayer(), input_size=(100, 10), name="w6", initialization='he', reg=0.01)
model.addlayer(Layers.AddLayer(), input_size=10, name='b6')
model.addlayer(Layers.Dropout(), activation=True, name='dropout6')
model.addlayer(Layers.SoftmaxLayer(), activation=True, name='softmax')

optimizer = Optimizer.SGD(batch_size=128)
model.train(x_train, y_train, optimizer, 3000, 0.01)

model.predict(x_test, y_test, train_flag=False)
print("Final test_ACC : ", (model.pred.argmax(1) == y_test.argmax(1)).mean())
print("Final test_LOSS : ", model.loss)
