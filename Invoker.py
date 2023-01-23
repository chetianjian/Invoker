from Factor import Factor
from Strategy import Strategy
from Density import Density
from Copula import BivariateCopula
from Similarity import Similarity
from Metric import Metric
from ShareHolderFetcher import ShareHolderFetcher
from utils import *


class Invoker(Factor, Strategy):

    def __init__(self):
        # print("I am a beacon of knowledge blazing out across a black sea of ignorance.")
        super().__init__()



    # ------------------------------------------------------------------------------------------


    @property
    def Density(self):
        return Density()


    @property
    def BivariateCopula(self):
        return BivariateCopula()


    @staticmethod
    def Similarity(x: np.array, y: np.array):
        """
        :param x: First vector.
        :param y: Second vector.
        """
        return Similarity(x=x, y=y)


    @staticmethod
    def Metric(x: np.array, y: np.array):
        """
        :param x: First point.
        :param y: Second point.
        """
        return Metric(x=x, y=y)


    @staticmethod
    def ShareHolderFetcher(start_date=None, end_date=None):
        """
        :param start_date: str. Format: 'YYYYMMDD'
        :param end_date: str. Format: 'YYYYMMDD'
        """
        return ShareHolderFetcher(start_date=start_date, end_date=end_date)


    # ------------------------------------------------------------------------------------------


    @property
    def broker_recommend(self):
        return pd.Series(self.block2code("券商金股"),
                         [self.code2name(_) for _ in self.block2code("券商金股")])


    def yield_candle(self, code_list, n=0):
        """
        Usage:
                iteration = Invoker.yield_candle(code_list)
                next(iteration)
        :param code_list: An iterable that stores the target stock codes.
        :param n: Default to 0, which means all data are required to be included.
                  Draw the candle plot for the last-n-days.
        :return: yield one candle graph each time you call the function.
        """

        for code in code_list:
            try:
                print(f"{code}: {self.stock_list.loc[code, 'name']}")
                yield draw_candle(self.code2df(code).iloc[-n:, :])
            except:
                print("All candles have been displayed.")
                break