import numpy as np
import pandas as pd
import ComputeGrad
from sklearn import datasets


def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)
    if a.ndim == 1:
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
    else:
        sum_exp_a = np.sum(exp_a, 1)
        sum_exp_a = sum_exp_a.reshape(sum_exp_a.shape[0], 1)
        y = exp_a / sum_exp_a
    return y


def cross_entropy_loss(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

iris = datasets.load_iris()
x = iris.data
x /= x.mean()
y = iris.target

print(x.shape, y.shape)
one_hot = np.zeros((y.shape[0], y.max() + 1))
one_hot[np.arange(y.shape[0]), y] = 1

y = one_hot

print(y.shape)
num_classes = 3
w = np.random.uniform(-1, 1, (x.shape[1], num_classes))
b = np.zeros(num_classes)
w, b = ComputeGrad.SoftmaxGD(x, y, w, b, 0.1, 10000, 128)

pred = x.dot(w) + b
pred = softmax(pred)
print("ACC : ", (pred.argmax(1) == y.argmax(1)).mean())
