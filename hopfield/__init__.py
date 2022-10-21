import numpy as np
from tqdm import tqdm


def vectorize(func):
    return np.vectorize(func)


class Hopfield:

    def __init__(self, q=0.0):
        """
        H - вектор вхідних збуджень системи,
        S - вектор внутрішнього стану нейрону,
        W - матриця звязків нейронів,
        F - фукнція переходу нейрона, S(t+1) = F(S(t), H(t)) ,
        """
        self.q = q
        self.f = np.vectorize(self.f)
        self.H = np.array([], float)
        self.S = np.array([], float)
        self.W = np.array([], float)

    def iterate(self, mode="async", q=0.0):
        """
        S(t+1) = F(S(t), H(t))
        """
        if mode == "sync":
            self.H = self.W @ self.H  # прохід ваг
            self.S = self.f(self.S, self.H, self.q)  # активація нейронів
        else:
            order = np.arange(len(self.H))
            np.random.shuffle(order)
            for i in order:
                inp_i = self.H @ self.W[i]
                self.S[i] = self.f(self.S[i], inp_i, q)
                self.H[i] = self.S[i]

    def train(self, images: np.ndarray):
        images = np.array(images, float)
        self.W = np.zeros((images.shape[1], images.shape[1]), float)

        self.W = images.T @ images
        np.fill_diagonal(self.W, 0)
        self.W /= images.shape[1]  # w/=n

    def run(self, X: np.array, time: int = 250, mode="async", q=None):
        if q is None:
            q = self.q
        X = np.array(X)
        if self.W.shape[1] != len(X):
            raise ValueError
        self.S = np.zeros(len(X), float)

        self.H = X.copy()
        for i in range(time):
            self.iterate(mode, q)
            if (self.H == X).all() and i > 0:  # перевірка зміни виходу з минулої ітерації
                return i, self.H.copy()
            X = self.H.copy()

        return time, self.H.copy()

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
