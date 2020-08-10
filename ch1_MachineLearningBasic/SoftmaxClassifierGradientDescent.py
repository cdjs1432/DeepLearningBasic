import numpy as np
import pandas as pd
import ComputeGrad


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


a = np.array([1.3, 2.4, 3.7])
print(softmax(a))


def cross_entropy_loss(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size


y = softmax(a)
print(y)
t = np.array([1, 0, 0])
print(cross_entropy_loss(y, t))
t = np.array([0, 1, 0])
print(cross_entropy_loss(y, t))
t = np.array([0, 0, 1])
print(cross_entropy_loss(y, t))

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

# initialize parameters and hyperparameters
num_classes = y_train.shape[1]
w = np.random.uniform(-1, 1, (x_train.shape[1], num_classes))
b = np.zeros(num_classes)

learning_rate = 0.01
epoch = 10000
batch_size = 512

w, b = ComputeGrad.SoftmaxGD(x_train, y_train, w, b, learning_rate, epoch, batch_size)

pred = x_train.dot(w) + b
pred = softmax(pred)
print("TRAIN ACC : ", (pred.argmax(1) == y_train.argmax(1)).mean())

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

pred = x_test.dot(w) + b
pred = softmax(pred)
print("TEST ACC : ", (pred.argmax(1) == y_test.argmax(1)).mean())
