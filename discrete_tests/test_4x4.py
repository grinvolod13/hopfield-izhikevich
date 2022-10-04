import numpy as np
import matplotlib.pyplot as plt
import hopfield

n = hopfield.Hopfield()

learn_data = np.array(
    [[1.0, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
     [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]])

""",
     [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
"""

inp = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1],np.float64)
print(inp)
plt.imshow([inp])
plt.show()

t, out = n.start(learn_data, inp)

print("t:", t)
print(out)

plt.imshow([out])
plt.show()
