import numpy as np
import ComputeGrad

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# same as MultiVariableGradientDescent
train_x = np.random.uniform(-2, 2, (100, 10))
ans_w = np.random.uniform(-3, 3, 10)
ans_b = np.random.uniform(-3, 3, 1)
train_y = train_x.dot(ans_w) + ans_b
train_y = sigmoid(train_y)
train_y[train_y > 0.5] = 1
train_y[train_y <= 0.5] = 0

w = np.ones(10)
b = 0

learning_rate = 0.01

epoch = 10000

w, b = ComputeGrad.LogisticGD(train_x, train_y, w, b, learning_rate, epoch)

z = train_x.dot(w) + b
pred = sigmoid(z)

pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
print((pred == train_y).mean())