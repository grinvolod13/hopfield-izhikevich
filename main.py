import numpy as np
import cv2
import matplotlib.pyplot as plt

import net


def f_none(s):
    return float(s)

def draw(mas):
    a, b = mas.shape
    for i in range(a):
        for j in range(b):
            if mas[i][j]>0.5:
                print('#', end='',sep='')
            else:
                print(' ', end='', sep='')


h = net.Net()

# training
l = []
img = None
for i in range(8):
    img = cv2.imread("data/train/fig/" + str(i) + '.png', cv2.IMREAD_GRAYSCALE) / 255.0
    img[img > 0.5] = 1.0
    img[img <= 0.5] = -1.0
    img = -img
    # plt.imshow(img)
    # plt.show()
    l.append(img.flatten())

l = np.array(l)
print("memorising...")
h.memorize(l)

# read and process image
inp = cv2.imread("data/train/fig/5.png", cv2.IMREAD_GRAYSCALE) / 255.0
# cv2.imshow("img",inp)
inp[inp >= 0.5] = 1.0
inp[inp <= 0.6] = -1.0
inp = -inp
plt.imshow(inp)
plt.show()

print("processing data...")
out = h.work_mode(inp.flatten(), 1000).reshape(20, 20)

# shows out image

plt.imshow(out)
plt.show()
# print(out)
# print(hopf._Net__W)
