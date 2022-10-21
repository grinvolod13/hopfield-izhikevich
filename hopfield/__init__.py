import numpy as np
from tqdm import tqdm

class Hopfield:

    def __init__(self, f=None, g=None):
        """
        H - вектор вхідних збуджень системи,
        S - вектор внутрішнього стану нейрону,
        W - матриця звязків нейронів,
        F - фукнція переходу нейрона, S(t+1) = F(S(t), H(t)) ,
        G - функція активації виходу нейрона, H(t+1) = W @ S(t+1)*1/n
        """
        self.H = np.array([], np.float64)
        self.S = np.array([], np.float64)
        self.W = np.array([], np.float64)

        if f is None:
            f = self.f_basic


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
        H(t+1) = S(t+1) @ W/n
        """
        self.H = self.H @ self.W  # прохід ваг
        self.S = self.f(self.S, self.H)  # активація нейронів
        self.H = self.S.copy()

    def memorisation(self, images: np.ndarray):
        self.W = np.zeros((images.shape[1], images.shape[1]), np.float64)
        for i in range(images.shape[0]):
            self.W += np.outer(images[i], images[i])
        np.fill_diagonal(self.W, 0)

        self.W /= images.shape[1]  # w/=n

        self.H = np.zeros(images.shape[1], np.float64)
        self.S = np.zeros(images.shape[1], np.float64)

    def start(self, images, X: np.array, t: int = 250):
        X = np.array(X)
        if images.shape[1] != len(X):
            raise ValueError

        self.memorisation(images)

        self.H = X.copy()
        self.S = np.zeros_like(self.H)
        for i in tqdm(range(t)):
            self.iteration()
            if (self.H == X).all() and i > 0:  # перевірка зміни виходу з минулої ітерації
                return i, self.H.copy()
            X = self.H.copy()

        return t, self.H.copy()
