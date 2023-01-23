import numpy as np

class Density(object):
    def __init__(self):
        pass


    @staticmethod
    def uniform_pdf(x, left=0, right=1):
        assert left < right
        return 1 / (right - left) if left < x < right else 0


    @staticmethod
    def gaussian(x, mu=0, sigma=1):
        assert sigma > 0
        return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))


    @staticmethod
    def multivariate_gaussian(x: np.array, mu: np.array, sigma: np.ndarray):
        assert len(x) == len(mu) == sigma.shape[0] == sigma.shape[1]

        return np.power(np.exp, -1 / 2 * np.matmul(np.matmul(x - mu, np.linalg.inv(sigma)), x - mu)) / \
            np.sqrt((2 * np.pi) ** len(x) * np.linalg.det(sigma))


