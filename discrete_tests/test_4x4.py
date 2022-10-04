import numpy as np
import matplotlib.pyplot as plt
import net

n = net.Net()

learn_data = np.array(
    [[1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
     [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1]])

inp = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1])

n.memorize(learn_data)

print(inp)
plt.imshow([inp])
plt.show()

out = [n.work_mode(inp)]
print(out)
plt.imshow(out)
plt.show()
