import numpy as np


def uniform_pdf(x, a=0, b=1):
    assert a < b
    return 1 / (b-a) if a < x < b else 0


def gaussian(x, mu=0, sd=1):
    assert sd > 0
    return 1 / np.sqrt(2 * np.pi * sd**2) * np.exp(-(x - mu)**2 / (2 * sd**2))



