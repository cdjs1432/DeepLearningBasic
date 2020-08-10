import numpy as np
from sklearn import datasets
import ComputeGrad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


cancer = datasets.load_breast_cancer()
x = cancer.data
x /= x.mean()
y = cancer.target
w = np.random.uniform(-1, 1, x.shape[1])
b = 0
learning_rate = 0.0001
epoch = 10000

w, b = ComputeGrad.LogisticGD(x, y, w, b, learning_rate, epoch)

z = x.dot(w) + b
pred = sigmoid(z)

pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
print("ACC : ", (pred == y).mean())
