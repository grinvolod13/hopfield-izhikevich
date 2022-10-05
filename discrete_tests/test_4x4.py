import numpy as np
import matplotlib.pyplot as plt
import hopfield

n = hopfield.Hopfield()

learn_data = np.array(
    [[1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]], np.float64)

plt.imshow([learn_data[0]])
plt.show()
plt.imshow([learn_data[1]])
plt.show()

inp = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1], np.float64)
print(inp)
plt.imshow([inp])
plt.show()

t, out = n.start(learn_data, inp, 10000)

print("t:", t)
print(out)

plt.imshow([out])
plt.show()
