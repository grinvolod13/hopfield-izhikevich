import numpy as np
import cv2
import matplotlib.pyplot as plt

import hopfield


def f_none(s):
    return float(s)


def draw(mas):
    a, b = mas.shape
    for i in range(a):
        for j in range(b):
            if mas[i][j] > 0.5:
                print('#', end='', sep='')
            else:
                print(' ', end='', sep='')


h = hopfield.Hopfield()

l = []
img = None
for i in range(8):
    img = cv2.imread('C:\\Users\\User\\Desktop\\hopfield-classic\\data\\train\\num\\' + str(i) + '.png', cv2.IMREAD_GRAYSCALE) / 255.0
    img[img > 0.5] = 1.0
    img[img <= 0.5] = -1.0
    img = -img
    l.append(img.flatten())

l = np.array(l)

# read and process image
inp = cv2.imread("C:\\Users\\User\\Desktop\\hopfield-classic\\data\\train\\num\\9.png", cv2.IMREAD_GRAYSCALE) / 255.0
inp[inp >= 0.5] = 1.0
inp[inp <= 0.6] = -1.0
inp = -inp

plt.imshow(inp)
plt.show()

t,out = h.start(l, inp.flatten())

# shows out image
print(t)
plt.imshow(out.reshape(20,20))
plt.show()

