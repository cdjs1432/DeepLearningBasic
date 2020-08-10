import numpy as np
import Layers

train_x = np.random.uniform(-5, 5, (100, 10))
ans_w = np.random.uniform(-5, 5, (10, 1))
ans_b = 3
train_y = train_x.dot(ans_w) + ans_b
w = np.random.uniform(-5, 5, (10, 1))
b = np.random.uniform(-5, 5, (1, 1))
learning_rate = 0.001
epoch = 20000

keys = ['w', 'b']
layers = {}
layers['w'] = Layers.MulLayer(w)
layers['b'] = Layers.AddLayer(b)
lastlayer = Layers.MSELayer(train_y)
grads = {}

for epoch in range(epoch):
    x = train_x
    y = train_y

    for key in keys:
        x = layers[key].forward(x)

    loss = lastlayer.forward(x)
    if epoch % 100 == 0:
        print("err : ", loss)

    dout = lastlayer.backward()

    for key in reversed(keys):
        dout = layers[key].backward(dout)
        grads[key] = layers[key].grad

    db = layers['b'].grad
    dw = layers['w'].grad

    w -= learning_rate * dw
    b -= learning_rate * db
