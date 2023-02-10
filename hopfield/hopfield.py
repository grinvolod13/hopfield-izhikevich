import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import random
import copy


class Hopfield:

    def __init__(self, np_type=np.float16, q=0.0):
        """
        H - вектор вхідних збуджень системи,
        S - вектор внутрішнього стану нейрону,
        W - матриця звязків нейронів,
        F - фукнція переходу нейрона, S(t+1) = F(S(t), H(t)) ,
        """
        self.q = q
        self.np_type = np_type
        self.f = np.vectorize(self.f)
        self.H = np.array([], self.np_type)
        self.S = np.array([], self.np_type)
        self.W = np.array([], self.np_type)
        self.gif = []
        

    def iterate(self, mode="sync", q=0):
        """
        S(t+1) = F(S(t), H(t))
        """
        if mode == "sync":
            self.S = self.f(self.S, self.S @ self.W, q)  # активація нейронів
        else:
            order = np.arange(float(self.S.shape[0])) # type: ignore
            np.random.shuffle(order)
            for i in order:
                self.S[i] = self.f(self.S[i], self.S @ self.W[i], q).clip(-1,1) # type: ignore

    def train(self, images: np.ndarray, zeroDiagonal=True):
        # images = np.array(images, self.np_type)
        self.W = images.T @ images
        if zeroDiagonal:
            np.fill_diagonal(self.W, 0)
        self.W /= images.shape[1]  # w/=n

        self.S = np.zeros(self.W.shape[0], dtype=self.np_type)

    def run(self, X: np.ndarray, time: int = 250, mode="sync", q=None, animate=None, notS=False):
        if q is None:
            q = self.q
            
        X = X.copy()
        
        if self.W.shape[1] != len(X):
            raise ValueError
        if notS:
            # self.S = np.random.randint(0, 2, len(X))*2-1
            self.S = np.zeros_like(X)
        else:
            self.S = X.copy()

        if animate:
            self.gif = [np.ceil(self.S * 127 + 127).reshape(animate, animate)]

        for i in range(time):
            self.iterate(mode, q)
            
            if animate:
                self.gif.append(np.ceil(self.S * 127 + 127).reshape(animate, animate))
                
            if (self.S == X).all() and i > 0:  # перевірка зміни виходу з минулої ітерації
                return {"time": i, "timeout": False, "output":self.S}
            
            X = self.S.copy()
            
        return {"time": i, "timeout": True, "output":self.S}

    @staticmethod
    def f(s, h, q):
        def j(sj, hj):
            if q == 0.0:
                sj = 1.0
            else:
                k = 2.0 * q / (abs(hj) + q)
                if k * (sj - 1) + 2 < 0:
                    sj = k * sj - k + 3
                else:
                    sj = k * sj - k + 1

            return sj

        if h == 0:
            return s
        if h > 0:
            return j(s, h)
        else:
            return -j(-s, h)
