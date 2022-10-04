import numpy as np
import matplotlib.pyplot as plt


def f_basic(s):
    if s > 0:
        return 1.0
    else:
        return -1.0




class Net:

    def __init__(self):
        self.__N = 0
        self.__M = 0
        self.i = 0
        # self.__S
        # self.__W
        # self.__image

    def memorize(self, train_set):
        (M, N) = train_set.shape
        self.__N = N
        self.__M = M
        self.W = np.zeros((N, N))
        self.S = np.zeros(N)
        self.image = train_set

        for i in range(M):
            self.W += np.outer(self.image[i], self.image[i].T)
        self.W /= np.float64(N)
        for i in range(N):
            self.W[i][i] = 0.0

    def process(self, func=None):
        if func == None:
            func = f_basic

        self.S[self.S >= 1.0] = 1.0
        self.S[self.S <= -1.0] = -1.0

        self.S = np.array([i for i in map(func, np.dot(self.W, self.S))])
        print(self.i)
        self.i+=1
        plt.title(str(self.i))
        plt.imshow(self.S.reshape(20,20))
        plt.show()

        self.S[self.S >= 1.0] = 1.0
        self.S[self.S <= -1.0] = -1.0

        return self.S.copy()

    def work_mode(self, X, t: int = 1000, func=None):
        self.S = np.copy(X)
        b = False
        for i in range(t):
            X = np.copy(self.S)
            self.process(func)

            if (X == self.S).all() and i>=2:
                if b:
                    print("t:", i + 1)
                    return self.S
                else:
                    b = True
            else:
                b = False

        print("timeout:", t)

        return self.S.copy()
