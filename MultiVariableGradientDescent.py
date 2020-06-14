import numpy as np
import ComputeGrad
train_x = np.random.uniform(-5, 5, (100, 10))
ans_w = np.random.uniform(-5, 5, (10, 1))
ans_b = 3
train_y = train_x.dot(ans_w) + ans_b
w = np.random.uniform(-5, 5, (10, 1))
b = np.random.uniform(-5, 5, (1, 1))
learning_rate = 0.00001

w, b = ComputeGrad.GradientDescent(train_x, train_y, 3, w)

print(ans_w)
print(w)

print(ans_b)
print(b)
