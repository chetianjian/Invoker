from MongoLoader import Mongo
import numpy as np


class TradingStrategy151(Mongo):
    def __init__(self):
        super().__init__()

    def Strategy_1_Price_Momemtum(self, H, S=1, T=12):
        """
        :param H: Holding Period.
        :param S: Skip Period, default to 1.
        :param T: Formation Perid, default to 12.
        :return: Strategy Yields.
        """

        self.slim()
        Rcum = self.close.shift(S) / self.close.shift(S+T) - 1
        Rmean = self.rate.shift(S).rolling(T).sum() / T
        df = np.square((self.rate.shift(S)-Rmean))
        sigma = np.sqrt(df.rolling(T).sum() / (T-1))
        R_risk_adjusted = Rmean / sigma

        pass








