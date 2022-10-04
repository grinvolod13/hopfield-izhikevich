import numpy as np


class Hopfield:

    def __init__(self, f=None, g=None):
        """
        H - вектор вхідних збуджень системи,
        S - вектор внутрішнього стану нейрону,
        W - матриця звязків нейронів,
        F - фукнція переходу нейрона, S(t+1) = F(S(t), H(t)) ,
        G - функція активації виходу нейрона, H(t+1) = G(W*S(t+1)*1/n)
        """
        self.H = np.array([], np.float64)
        self.S = np.array([], np.float64)
        self.W = np.array([], np.float64)

        if f is None:
            f = self.f_basic
        if g is None:
            g = self.g_basic

        self.f = np.vectorize(f)
        self.g = np.vectorize(g)

    @staticmethod
    def f_basic(s: np.float64, h: np.float64):
        if h >= 0:
            return np.float64(1.0)
        return np.float64(-1.0)

    @staticmethod
    def g_basic(s: np.float64):
        return s

    def iteration(self):
        """
        S(t+1) = F(S(t), H(t))
        H(t+1) = G(W*S(t+1)*1/n)
        """
        self.S = self.f(self.S, self.H).copy()
        self.H = self.g(np.dot(self.W, self.S) / len(self.S)).copy()

    def memorisation(self, images: np.ndarray):
        self.W = np.zeros((images.shape[1], images.shape[1]), np.float64)
        for i in range(images.shape[0]):
            self.W += np.outer(images[i], images[i].T)

        np.fill_diagonal(self.W, 0.0)

        self.W /= images.shape[1]  # w/=n

        self.H = np.zeros(images.shape[1], np.float64)
        self.S = np.zeros(images.shape[1], np.float64)

    def start(self, images, X: np.array, t: int = 1000):
        X = np.array(X)
        if images.shape[1] != len(X):
            raise ValueError

        self.memorisation(images)

        self.H = X.copy()
        self.S = X.copy()
        for i in range(t):

            self.iteration()
            if (self.S == X).all() and i > 0:  # перевірка зміни внутрішнього стану з минулої ітерації
                return i, self.S
            X = self.S.copy()

        return t, self.S
