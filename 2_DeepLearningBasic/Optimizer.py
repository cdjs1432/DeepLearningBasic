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

            model.predict(x, y)
            dout = model.layers[model.keys[-1]].backward()
            for i in reversed(range(len(model.keys) - 1)):
                key = model.keys[i]
                dout = model.layers[key].backward(dout)
                if key in model.params:
                    model.grads[key] = model.layers[key].grad
                    model.params[key] -= learning_rate * model.grads[key]

            if epochs % (epoch / 10) == 0:
                print("ACC on epoch %d : " % epochs, (model.pred.argmax(1) == y.argmax(1)).mean())
                print("LOSS on epoch %d : " % epochs, model.loss)
        model.predict(x_train, y_train)
        print("Final train_ACC : ", (model.pred.argmax(1) == y_train.argmax(1)).mean())
        print("Final train_LOSS : ", model.loss)


class Momentum:
    def __init__(self, batch_size, momentum):
        self.x = None
        self.y = None
        self.model = None
        self.batch_size = batch_size
        self.momentum = momentum

    def train(self, x_train, y_train, epoch, learning_rate, model):
        velocity = {}
        for p in model.params:
            velocity[p] = np.zeros(model.params[p].shape)
        for epochs in range(epoch):
            batch_mask = np.random.choice(x_train.shape[0], self.batch_size)
            x = x_train[batch_mask]
            y = y_train[batch_mask]

            model.predict(x, y)
            dout = model.layers[model.keys[-1]].backward()
            for i in reversed(range(len(model.keys) - 1)):
                key = model.keys[i]
                dout = model.layers[key].backward(dout)
                if key in model.params:
                    model.grads[key] = model.layers[key].grad
                    velocity[key] = self.momentum * velocity[key] + (1 - self.momentum) * model.grads[key]
                    model.params[key] -= learning_rate * velocity[key]

            if epochs % (epoch / 10) == 0:
                print("ACC on epoch %d : " % epochs, (model.pred.argmax(1) == y.argmax(1)).mean())
                print("LOSS on epoch %d : " % epochs, model.loss)
        model.predict(x_train, y_train)
        print("Final train_ACC : ", (model.pred.argmax(1) == y_train.argmax(1)).mean())
        print("Final train_LOSS : ", model.loss)
