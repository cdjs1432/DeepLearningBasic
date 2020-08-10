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


def GradientDescent(x, y, b, w, learning_rate=0.01, epoch=10000):
    if type(w) != np.ndarray:
        w = float(w)
        w = np.reshape(w, (1, 1))
    if x.size == x.shape[0]:
        x = x.reshape(x.shape[0], 1)
    if y.size == y.shape[0]:
        y = y.reshape(y.shape[0], 1)
    for epochs in range(epoch):
        pred = x.dot(w) + b  # y = w1x1 + w2x2 + ... + wmxm + b
        """
        err = 1/m * ((Wx1 + b - y1)^2 + (Wx2 + b - y2)^2 + ... + (Wxm + b - ym)^2)
        to minimize err --> differentiate w and b
        dw = 2/m * ((Wx1 + b - y1) * x1 + (Wx2 + b - y2) * x2 + ... + (Wxm + b - ym) * xm) 
        db = 2/m * ((Wx1 + b - y1) * 1 + (Wx2 + b - y2) * 1 + ... + (Wxm + b - ym))  
        """
        dw = ((pred - y) * x).mean(0)
        db = (pred - y).mean()
        dw = dw.reshape(dw.shape[0], 1)
        w -= dw * learning_rate
        b -= db * learning_rate
        if epochs % 1000 == 0:
            err = np.mean(np.square(pred - y))  # err = 1/m * (w1x1 + w2x2 + ... + wmxm + b - y)^2
            print("error : ", err)
    return w, b


def SGD(x, y, b, w, learning_rate=0.01, epoch=1000, batch_size=16):
    if type(w) != np.ndarray:
        w = float(w)
        w = np.reshape(w, (1, 1))
    if x.size == x.shape[0]:
        x = x.reshape(x.shape[0], 1)
    if y.size == y.shape[0]:
        y = y.reshape(y.shape[0], 1)

    for epochs in range(epoch):
        batch_mask = np.random.choice(x.shape[0], batch_size)
        x_batch = x[batch_mask]
        y_batch = y[batch_mask]

        pred = x_batch.dot(w) + b
        dw = ((pred - y_batch) * x_batch).mean(0)
        dw = dw.reshape(dw.shape[0], 1)
        db = (pred - y_batch).mean()
        w -= dw * learning_rate
        b -= db * learning_rate
        if epochs % (epoch / 10) == 0:
            pred = x.dot(w) + b
            err = np.mean(np.square(pred - y))
            print("error : ", err)
    return w, b


def LogisticGD(x, y, w, b=0, learning_rate=0.001, epoch=10000):
    for epochs in range(epoch):
        z = x.dot(w) + b
        pred = sigmoid(z)
        err = -np.log(pred) * y - np.log(1 - pred) * (1 - y)
        err = err.mean()
        """
        to minimize err --> differentiate pred
        err = -ln(pred)*y - ln(1-pred)*(1-y)
        --> -y/pred + (1-y) / (1-pred)

        differentiate pred by z
        pred = 1 / (1+e^-z)
        --> e^z / (1+e^-z)^2
        --> e^z / (1+e^-z) * 1 / (1+e^-z)
        --> pred * (1 - pred)

        chain rule : dl/dpred * dpred/dz = dl/dz

        as the chain rule, derivative of the loss function respect of z
        --> (-y/pred + (1-y) / (1-pred)) * (pred * 1-pred)
        --> (1-pred) * -y + pred * (1-y)
        --> -y +y*pred +pred -y*pred
        --> pred - y

         now, let's find dl/dw
         dl/dw = dl/dz * dz/dw
               = (pred - y) * dz/dw
         let's find dz/dw...

         z = wT.x + b
         --> dz/dw = x

         so, dl/dw = (pred - y) * dz/dw
                   = (pred - y) * x


        similarly, find dl/db...
        dl/db = dl/dz * dz/db
              =  (pred - y) * dz/db
        z = wT.x + b
        --> dz/db = 1

        so, dl/db = (pred - y)

        """
        dw = (pred - y).dot(x)
        db = (pred - y).mean()
        w -= dw * learning_rate
        b -= db * learning_rate
        if epochs % 1000 == 0:
            print(err)
    return w, b


def SoftmaxGD(x, y, w, b, learning_rate=0.01, epoch=100000, batch_size=128):
    for epochs in range(epoch):
        batch_mask = np.random.choice(x.shape[0], batch_size)
        x_batch = x[batch_mask]
        y_batch = y[batch_mask]

        z = x_batch.dot(w) + b
        pred = softmax(z)
        dz = (pred - y_batch) / batch_size
        dw = np.dot(x_batch.T, dz)
        db = dz * 1.0
        w -= dw * learning_rate
        b -= (db * learning_rate).mean(0)

        if epochs % (epoch / 10) == 0:
            pred = softmax(x.dot(w) + b)
            print("ACC on epoch %d : " % epochs, (pred.argmax(1) == y.argmax(1)).mean())
            err = cross_entropy_loss(pred, y)
            print("ERR on epoch %d : " % epochs, err)

    return w, b
