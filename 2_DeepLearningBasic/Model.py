import numpy as np
from Layers import *


class Model:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.keys = []
        self.layers = {}
        self.num = 0
        self.loss = None
        self.pred = None

    def addlayer(self, layer, activation=False, input_size=None, name=None):
        if name is None:
            name = str(self.num)

        self.keys.append(name)
        self.num += 1
        self.layers[name] = layer

        if not activation:
            self.params[name] = np.random.uniform(-1, 1, input_size)
            self.layers[name].param = self.params[name]

    def predict(self, x, y):
        for i in range(len(self.keys) - 1):
            key = self.keys[i]
            x = self.layers[key].forward(x)
        self.loss = self.layers[self.keys[-1]].forward(x, y)
        self.pred = softmax(x)

    def train(self, x_train, y_train, epoch, learning_rate, batch_size):
        for epochs in range(epoch):
            batch_mask = np.random.choice(x_train.shape[0], batch_size)
            x = x_train[batch_mask]
            y = y_train[batch_mask]

            self.predict(x, y)
            dout = self.layers[self.keys[-1]].backward()
            for i in reversed(range(len(self.keys) - 1)):
                key = self.keys[i]
                dout = self.layers[key].backward(dout)
                if key in self.params:
                    self.grads[key] = self.layers[key].grad
                    self.params[key] -= learning_rate * self.grads[key]

            if epochs % (epoch / 10) == 0:
                print("ACC on epoch %d : " % epochs, (self.pred.argmax(1) == y.argmax(1)).mean())
                print("LOSS on epoch %d : " % epochs, self.loss)
