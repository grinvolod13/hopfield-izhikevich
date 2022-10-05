q = 0.4

import numpy as np
import cv2
import matplotlib.pyplot as plt
import hopfield


def j(x, h):
    k = 2.0 * q / (abs(h) + q)
    if k != 0.0:
        if k * (x - 1) + 2 < 0:
            x = k * x - k + 3
        else:
            x = k * x - k + 1
    else:
        x = 1.0
    return x


def f(s, h):
    if h > 0:
        return j(s, h)
    else:
        return -j(-s, h)


def f_none(s):
    return float(s)


n = hopfield.Hopfield(f)

l = []
img = None
for i in range(3):
    img = cv2.imread('C:\\Users\\User\\Desktop\\hopfield-classic\\data\\train\\num\\' + str(i) + '.png',
                     cv2.IMREAD_GRAYSCALE) / 255.0
    img[img > 0.5] = 1.0
    img[img <= 0.5] = -1.0
    img = -img
    l.append(img.flatten())

l = np.array(l)

# read and process image
inp = cv2.imread("C:\\Users\\User\\Desktop\\hopfield-classic\\data\\train\\num\\2.png", cv2.IMREAD_GRAYSCALE) / 255.0
inp[inp >= 0.5] = 1.0
inp[inp <= 0.6] = -1.0
inp = -inp

t, out = n.start(l, inp.flatten(), 250)

# shows out image
print(t)
plt.imshow(out.reshape(20, 20))
plt.show()
