import numpy as np
import matplotlib.pyplot as plt
import hopfield

q = 0.4


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


n = hopfield.Hopfield(f)

learn_data = np.array(
    [[1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]], np.float64)

inp = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1], np.float64)
print(inp)
plt.imshow([inp])
plt.show()

t, out = n.start(learn_data, inp, 250)

print("t:", t)
print(out)

plt.imshow([out])
plt.show()
