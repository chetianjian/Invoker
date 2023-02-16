import numpy as np

class Metric(object):

    def __init__(self, x=None, y=None):
        """
        :param x: np.array. First point.
        :param y: np.array. Second point.
        """

        if x is not None and y is not None:
            assert len(x) == len(y)
        self.x = x
        self.y = y
        self.dimension = len(x)


    @property
    def manhattan(self):
        """
        Manhattan distance.
        :return: ||x - y||_1
        """

        return np.nansum(np.absolute(self.x - self.y))


    @property
    def euclidean(self):
        """
        Euclidean distance.
        :return: ||x - y||_2
        """

        return np.sqrt(np.nansum(np.power(self.x - self.y, 2)))


    @property
    def discrete(self):
        """
        Discrete distance.
        :return: 1 if x == y else 0
        """

        if np.sum(self.x == self.y) + np.sum(np.isnan(self.x) == np.isnan(self.y)) \
                == self.dimension:
            return 1
        return 0


    @staticmethod
    def dotToPlane(point, plane):
        """
        Calculate the geometric distance from a point to a plane in the Euclidean space.
        :param point: np.array.
        :param plane:
        :return:
        """
