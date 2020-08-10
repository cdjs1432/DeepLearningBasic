import numpy as np
import pandas as pd
import cv2
from Model import Model
model = Model()
model.load('model.txt')
num = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

for i in range(10):
    gray = cv2.imread("images/" + str(num[i]) + ".png", cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(255-gray, (28, 28))
    x = gray.flatten() / 255.0
    pred = np.argmax(model.predict(x))
    print(i, pred)