import numpy as np


class SGD:
    def __init__(self, batch_size):
        self.x = None
        self.y = None
        self.model = None
        self.batch_size = batch_size

    def train(self, x_train, y_train, epoch, learning_rate, model):
        for epochs in range(epoch):
            batch_mask = np.random.choice(x_train.shape[0], self.batch_size)
            x = x_train[batch_mask]
            y = y_train[batch_mask]

            z = model.forward(x)
            model.loss = model.layers[model.keys[-1]].forward(z, y)
            model.loss += sum(model.l2.values()) / 2
            dout = model.layers[model.keys[-1]].backward()
            for i in reversed(range(len(model.keys) - 1)):
                key = model.keys[i]
                dout = model.layers[key].backward(dout)
                if key in model.params:
                    model.grads[key] = model.layers[key].grad + model.weight_decay_lambda[key] * model.params[key]
                    model.params[key] -= learning_rate * model.grads[key]

            if epochs % (epoch / 10) == 0:
                model.eval(x, y, epoch=epochs)
        model.eval(x_train, y_train)


class Momentum:
    def __init__(self, batch_size, momentum, nesterov=False):
        self.x = None
        self.y = None
        self.model = None
        self.batch_size = batch_size
        self.momentum = momentum
        self.nesterov = nesterov

    def train(self, x_train, y_train, epoch, learning_rate, model):
        velocity = {}
        for p in model.params:
            velocity[p] = np.zeros(model.params[p].shape)
        for epochs in range(epoch):
            batch_mask = np.random.choice(x_train.shape[0], self.batch_size)
            x = x_train[batch_mask]
            y = y_train[batch_mask]

            z = model.forward(x)
            model.loss = model.layers[model.keys[-1]].forward(z, y)
            model.loss += sum(model.l2.values()) / 2
            dout = model.layers[model.keys[-1]].backward()
            for i in reversed(range(len(model.keys) - 1)):
                key = model.keys[i]
                dout = model.layers[key].backward(dout)
                if key in model.params:
                    model.grads[key] = model.layers[key].grad
                    if self.nesterov:
                        velocity[key] = self.momentum * velocity[key] - learning_rate * model.grads[key]
                        model.params[key] += self.momentum * velocity[key] - learning_rate * model.grads[key]
                    else:
                        velocity[key] = self.momentum * velocity[key] - learning_rate * model.grads[key]
                        model.params[key] += velocity[key]

            if epochs % (epoch / 10) == 0:
                model.eval(x, y, epochs)
        model.eval(x_train, y_train)


class Adagrad:
    def __init__(self, batch_size):
        self.x = None
        self.y = None
        self.model = None
        self.batch_size = batch_size
        self.epsilon = 10e-8

    def train(self, x_train, y_train, epoch, learning_rate, model):
        G = {}
        for p in model.params:
            G[p] = np.zeros(model.params[p].shape)
        for epochs in range(epoch):
            batch_mask = np.random.choice(x_train.shape[0], self.batch_size)
            x = x_train[batch_mask]
            y = y_train[batch_mask]

            z = model.forward(x)
            model.loss = model.layers[model.keys[-1]].forward(z, y)
            model.loss += sum(model.l2.values()) / 2
            dout = model.layers[model.keys[-1]].backward()
            for i in reversed(range(len(model.keys) - 1)):
                key = model.keys[i]
                dout = model.layers[key].backward(dout)
                if key in model.params:
                    model.grads[key] = model.layers[key].grad
                    G[key] += np.square(model.grads[key])
                    model.params[key] -= np.multiply(learning_rate / (np.sqrt(G[key] + self.epsilon)), model.grads[key])

            if epochs % (epoch / 10) == 0:
                model.eval(x, y, epochs)
        model.eval(x_train, y_train)


class RMSProp:
    def __init__(self, batch_size, gamma=0.9):
        self.x = None
        self.y = None
        self.model = None
        self.batch_size = batch_size
        self.epsilon = 10e-8
        self.gamma = gamma

    def train(self, x_train, y_train, epoch, learning_rate, model):
        G = {}
        for p in model.params:
            G[p] = np.zeros(model.params[p].shape)
        for epochs in range(epoch):
            batch_mask = np.random.choice(x_train.shape[0], self.batch_size)
            x = x_train[batch_mask]
            y = y_train[batch_mask]

            z = model.forward(x)
            model.loss = model.layers[model.keys[-1]].forward(z, y)
            model.loss += sum(model.l2.values()) / 2
            dout = model.layers[model.keys[-1]].backward()
            for i in reversed(range(len(model.keys) - 1)):
                key = model.keys[i]
                dout = model.layers[key].backward(dout)
                if key in model.params:
                    model.grads[key] = model.layers[key].grad
                    G[key] = self.gamma * G[key] + (1 - self.gamma) * np.square(model.grads[key])
                    model.params[key] -= np.multiply(learning_rate / (np.sqrt(G[key] + self.epsilon)), model.grads[key])

            if epochs % (epoch / 10) == 0:
                model.eval(x, y, epoch=epochs)
        model.eval(x_train, y_train)


class AdaDelta:
    def __init__(self, batch_size, gamma=0.9):
        self.x = None
        self.y = None
        self.model = None
        self.batch_size = batch_size
        self.epsilon = 10e-8
        self.gamma = gamma

    def train(self, x_train, y_train, epoch, learning_rate, model):
        G = {}
        s = {}
        for p in model.params:
            G[p] = np.zeros(model.params[p].shape)
            s[p] = np.zeros(model.params[p].shape)
        for epochs in range(epoch):
            batch_mask = np.random.choice(x_train.shape[0], self.batch_size)
            x = x_train[batch_mask]
            y = y_train[batch_mask]

            z = model.forward(x)
            model.loss = model.layers[model.keys[-1]].forward(z, y)
            model.loss += sum(model.l2.values()) / 2
            dout = model.layers[model.keys[-1]].backward()
            for i in reversed(range(len(model.keys) - 1)):
                key = model.keys[i]
                dout = model.layers[key].backward(dout)
                if key in model.params:
                    model.grads[key] = model.layers[key].grad
                    d_t = np.multiply(np.sqrt(s[key] + self.epsilon) / np.sqrt(G[key] + self.epsilon), model.grads[key])
                    G[key] = self.gamma * G[key] + (1 - self.gamma) * np.square(model.grads[key])
                    s[key] = self.gamma * s[key] + (1 - self.gamma) * np.square(d_t)
                    model.params[key] -= d_t

            if epochs % (epoch / 10) == 0:
                model.eval(x, y, epoch=epochs)
        model.eval(x_train, y_train)


class Adam:
    def __init__(self, batch_size, beta_1 = 0.9, beta_2 = 0.999):
        self.x = None
        self.y = None
        self.model = None
        self.batch_size = batch_size
        self.epsilon = 10e-8
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def train(self, x_train, y_train, epoch, learning_rate, model):
        m = {}
        m_hat = {}
        v = {}
        v_hat = {}
        for p in model.params:
            m[p] = np.zeros(model.params[p].shape)
            m_hat[p] = np.zeros(model.params[p].shape)

            v[p] = np.zeros(model.params[p].shape)
            v_hat[p] = np.zeros(model.params[p].shape)
        for epochs in range(epoch):
            batch_mask = np.random.choice(x_train.shape[0], self.batch_size)
            x = x_train[batch_mask]
            y = y_train[batch_mask]

            z = model.forward(x)
            model.loss = model.layers[model.keys[-1]].forward(z, y)
            model.loss += sum(model.l2.values()) / 2
            dout = model.layers[model.keys[-1]].backward()
            for i in reversed(range(len(model.keys) - 1)):
                key = model.keys[i]
                dout = model.layers[key].backward(dout)
                if key in model.params:
                    model.grads[key] = model.layers[key].grad
                    m[key] = self.beta_1 * m[key] + (1 - self.beta_1)*model.grads[key]
                    m_hat[key] = m[key] / (1 - self.beta_1 * self.beta_1)
                    v[key] = self.beta_2 * v[key] + (1 - self.beta_2) * model.grads[key] * model.grads[key]
                    v_hat[key] = v[key] / (1 - self.beta_2 * self.beta_2)
                    model.params[key] -= learning_rate * m_hat[key] / np.sqrt(v_hat[key] + self.epsilon)


            if epochs % (epoch / 10) == 0:
                model.eval(x, y, epoch=epochs)
        model.eval(x_train, y_train)