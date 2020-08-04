from Layers import *


class Model:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.keys = []
        self.layers = {}
        self.num = 0
        self.l2 = {}
        self.size = {}
        self.weight_decay_lambda = {}
        self.loss = None
        self.pred = None

    def addlayer(self, layer, name=None):
        if name is None:
            name = str(self.num)

        self.keys.append(name)
        self.num += 1
        self.layers[name] = layer

    def forward(self, x, train_flag=True):
        for i in range(len(self.keys) - 1):
            key = self.keys[i]
            if isinstance(self.layers[key], Dropout):
                x = self.layers[key].forward(x, train_flag)
            else:
                x = self.layers[key].forward(x)
            if key in self.weight_decay_lambda:
                self.l2[key] = np.sum(np.square(self.params[key])) * self.weight_decay_lambda[key]
        return x

    def predict(self, x):
        x = self.forward(x)
        self.pred = softmax(x)
        return self.pred

    def eval(self, x, y, epoch=None):
        x = self.forward(x, False)
        self.loss = self.layers[self.keys[-1]].forward(x, y)
        self.loss += sum(self.l2.values()) / 2
        self.pred = softmax(x)

        if epoch is None:
            print("ACC : ", (self.pred.argmax(1) == y.argmax(1)).mean())
            print("LOSS : ", self.loss)
        else:
            print("ACC on epoch %d : " % epoch, (self.pred.argmax(1) == y.argmax(1)).mean())
            print("LOSS on epoch %d : " % epoch, self.loss)

    def train(self, x_train, y_train, optimizer, epoch, learning_rate, skip_init=False):
        if not skip_init:
            in_size = x_train.shape[1:]
            for name in self.layers:
                if not self.layers[name].activation:
                    out_size = self.layers[name].out
                    if type(in_size) is int:
                        size = (in_size, out_size)
                    else:
                        size = (*in_size, out_size)
                    self.size[name] = size

                    self.weight_decay_lambda[name] = self.layers[name].reg
                    if isinstance(self.layers[name], AddLayer):
                        self.params[name] = np.zeros(self.layers[name].out)
                    elif self.layers[name].init is 'xavier':
                        self.params[name] = xavier_initialization(size)
                    elif self.layers[name].init is 'he':
                        self.params[name] = he_initialization(size)
                    else:
                        self.params[name] = np.random.uniform(size)
                    in_size = out_size

                    self.layers[name].param = self.params[name]
        optimizer.train(x_train, y_train, epoch, learning_rate, self)

    def save(self, path=''):
        f = open(path + "model.txt", 'w')
        f.write(path + "weight.npz\n")

        params = {}
        for name in self.layers:
            data = self.layers[name].__class__.__name__ + "\n" + name + "\n"
            if not self.layers[name].activation:
                params[name] = self.layers[name].param
            f.write(data)
        np.savez(path + "weight", **params)

    def load(self, path):
        f = open(path)
        weight_path = f.readline()[:-1]
        load = np.load(weight_path)
        while True:
            layer = f.readline()[:-1]
            if not layer:
                break
            name = f.readline()[:-1]

            self.addlayer(eval(layer)(), name)
            if name in load:
                self.layers[name].param = load[name]
