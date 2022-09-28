import numpy as np
import cv2
import matplotlib.pyplot as plt

import net


h = net.Net()

# training
l = []
img = 0
for i in range(4):
    img = cv2.imread("data/train/num/"+str(i)+'.png',cv2.IMREAD_GRAYSCALE)
    l.append(1 - img.flatten()*2/255.0)

l = np.array(l,np.float64)
print("memorising...")
h.memorize(l)



# read and process image
inp = 1 - cv2.imread("data/train/num/0.png",cv2.IMREAD_GRAYSCALE).flatten()*2/255.0
print("processing data...")
out = h.work_mode(inp, 100)


# shows out image
img_out = (out).reshape(20,20)
plt.imshow(img_out)
plt.show()



# print(out)
# print(hopf._Net__W)


