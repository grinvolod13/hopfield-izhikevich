
import numpy as np

class Net:

    def __init__(self):
        self.__N = 0
        self.__M = 0
        # self.__S
        # self.__W
        # self.__image

    def memorize(self, train_set):
        (M, N) = train_set.shape
        self.__N = N
        self.__M = M
        self.W = np.zeros((N, N),np.float64)
        self.S = np.zeros(N,np.float64)
        self.image = train_set

        # for i in range(N):
        #     print(i)
        #     for j in range(N):
        #         if i == j:
        #             continue
        #         sum = 0.0
        #         for k in range(M):
        #             sum += self.image[k][i] * self.image[k][j]
        #         self.W[i][j] = sum / float(N)
        for i in range(M):
            self.W+=np.outer(self.image[i].T,self.image[i])
        self.W/=float(N)
        for i in range(N):
            self.W[i][i]=0.0

    @staticmethod
    def f_basic(s):
        if s > 0:
            return 1.0
        elif s < 0:
            return -1.0
        else:
            return 0.0
    @staticmethod
    def f_none(s):
        return float(s)


    def process(self):
        self.S = np.array([i for i in map(self.f_basic, self.S.dot(self.W))])
        return self.S

    def is_image(self, X):
        for i in range(self.__M):
            if (X == self.image[i]).all():
                print("is image: ",X)
                return True
        return False

    def work_mode(self, X, t: int):
        self.S = X.copy()
        b = False
        for i in range(t):
            X = self.S.copy()
            self.process()
            if (X == self.S).all() and self.is_image(self.S):
                if b:
                    print("t:", i+1)
                    print(self.S)
                    return self.S
                else:
                    b = True
            else:
                b = False

        print("timeout:",t)

        return self.S
