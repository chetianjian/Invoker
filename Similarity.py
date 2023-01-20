import numpy as np

class Similarity(object):

    def __init__(self, x: np.array, y: np.array):
        """
        :param x: First vector.
        :param y: Second vector.
        """
        assert len(x) == len(y)
        assert len(x) != 1
        self.x = x
        self.y = y
        self.dimension = len(x)


    @property
    def cosine(self):
        """
        Cosine similarity.
        :return: (x · y) / ( ||x||_2 * ||y||_2 )
        """

        return np.dot(self.x, self.y) / \
            np.sqrt(np.nansum(np.power(self.x, 2))) / \
            np.sqrt(np.nansum(np.power(self.y, 2)))


