import numpy as np
import ComputeGrad
data = np.array([[1, 2, 3], [4, 6, 8]])  # y=2x+2 --> w=2, b=2
train_x = data[0]
train_y = data[1]

w = 1
b = 1

learning_rate = 0.1
epoch = 10000
w, b = ComputeGrad.GradientDescent(train_x, train_y, b, w, learning_rate, epoch)


print(w);
print(b)
