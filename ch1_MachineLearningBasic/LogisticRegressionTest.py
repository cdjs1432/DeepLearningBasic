import pandas as pd
import numpy as np
import ComputeGrad


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


data = pd.read_csv("./Data/diabetes.csv")
train_data = data[:600]
test_data = data[600:]

train_y = train_data["Outcome"]
train_x = train_data.drop("Outcome", 1)
train_x = train_x.values
train_x /= train_x.max(axis=0)
train_y = train_y.values

test_y = test_data["Outcome"]
test_x = test_data.drop("Outcome", 1)
test_x = test_x.values
test_x /= test_x.max(axis=0)
test_y = test_y.values

w = np.ones(8)
b = 0

w, b = ComputeGrad.LogisticGD(train_x, train_y, w, b, learning_rate=0.01)

z = train_x.dot(w) + b
pred = sigmoid(z)
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
print("Train Acc: ", (train_y == pred).mean())
z = test_x.dot(w) + b
pred = sigmoid(z)
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
print("Test Acc: ", (test_y == pred).mean())
