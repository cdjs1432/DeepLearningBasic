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

    def train(self, x_train, y_train, optimizer, epoch, learning_rate):
        optimizer.train(x_train, y_train, epoch, learning_rate, self)
