import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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
    C = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + C)) / batch_size


class MulLayer:
    def __init__(self, w):
        self.x = None
        self.w = w
        self.grad = None

    def forward(self, x):
        self.x = x
        return x.dot(self.w)

    def backward(self, dout):
        self.grad = np.dot(self.x.T, dout)
        return np.dot(dout, self.w.T)


class AddLayer:
    def __init__(self, b):
        self.x = None
        self.b = b
        self.grad = None

    def forward(self, x):
        self.x = x
        return x + self.b

    def backward(self, dout):
        self.grad = dout.mean()
        return dout


class MSELayer:
    def __init__(self, y):
        self.x = None
        self.y = y
        self.loss = None

    def forward(self, x):
        self.x = x
        self.loss = np.square(x - self.y).mean()
        return self.loss

    def backward(self):
        return self.x - self.y


class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, z):
        self.out = sigmoid(z)
        return self.out

    def backward(self, dout):
        return (1 - self.out) * self.out * dout


class SoftmaxLayer:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_loss(self.y, self.t)
        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx