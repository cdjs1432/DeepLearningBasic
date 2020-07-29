from Layers import *


class Model:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.keys = []
        self.layers = {}
        self.num = 0
        self.l2 = {}
        self.weight_decay_lambda = {}
        self.loss = None
        self.pred = None

    def addlayer(self, layer, activation=False, input_size=None, name=None, initialization=None, reg=0):
        if name is None:
            name = str(self.num)

        self.keys.append(name)
        self.num += 1
        self.layers[name] = layer

        if not activation:
            self.weight_decay_lambda[name] = reg
            if isinstance(layer, AddLayer):
                self.params[name] = np.zeros(input_size)
            elif initialization is 'xavier':
                self.params[name] = xavier_initialization(input_size)
            elif initialization is 'he':
                self.params[name] = he_initialization(input_size)
            else:
                self.params[name] = np.random.uniform(-1, 1, input_size)

            self.layers[name].param = self.params[name]

    def predict(self, x, y, train_flag=True):
        for i in range(len(self.keys) - 1):
            key = self.keys[i]
            if isinstance(self.layers[key], Dropout):
                x = self.layers[key].forward(x, train_flag)
            else:
                x = self.layers[key].forward(x)
            if key in self.weight_decay_lambda:
                self.l2[key] = np.sum(np.square(self.params[key])) * self.weight_decay_lambda[key]
        self.loss = self.layers[self.keys[-1]].forward(x, y)
        self.loss += sum(self.l2.values())/2
        self.pred = softmax(x)

    def train(self, x_train, y_train, optimizer, epoch, learning_rate):
        optimizer.train(x_train, y_train, epoch, learning_rate, self)
