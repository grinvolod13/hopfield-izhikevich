import numpy as np
import matplotlib.pyplot as plt
import net
n = net.Net()
inp = np.array([np.array([1.0,1,1,-1.0])])

n.memorize(inp)

print(inp)

out = n.work_mode(inp,1000)

print(out)



