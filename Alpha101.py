from utils import *
from MongoLoader import Mongo


class Alpha101(Mongo):
    def __init__(self):
        super().__init__()


    def alpha_001(self):
        """
        :return: (rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close),2.),5))-0.5)
        """

        data1 = self.volume.diff(periods=1).rank(axis=1, pct=True)
        data2 = ((self.close - self.open) / self.open).rank(axis=1, pct=True)
        alpha = -data1.iloc[-6:, :].corrwith(data2.iloc[-6:, :]).dropna()
        alpha = alpha.dropna()
        return alpha


    def alpha_002(self):
        """
        :return: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        """

        return -1 * np.log(self.volume).diff(2).rank(pct=True).corrwith(
            (((self.close - self.open) / self.open).rank(pct=True)).shift(6)
        )


    def alpha_003(self):
        """
        :return: (-1 * correlation(rank(open), rank(volume), 10))
        """

        return -1 * self.open.rank(pct=True).corrwith(
            self.volume.rank(pct=True).shift(10)
        )


    def alpha_004(self):
        """
        :return: (-1 * Ts_Rank(rank(low), 9))
        """

        return -1 * ts_rank(self.low.rank(pct=True), 9)



    def alpha_005(self):
        """
        :return: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
        """

        return -1 * (self.open - self.vwap.rolling(10).sum() / 10).rank(pct=True) * \
            np.absolute((self.close - self.vwap).rank(pct=True))


    def alpha_006(self):
        """
        :return: (-1 * correlation(open, volume, 10))
        """

        return -1 * self.open.corrwith(self.volume.shift(10))


    def alpha_007(self):
        """
        :return: ((adv20<volume)?((-1*ts_rank(abs(delta(close,7)),60))*sign(delta(close,7))):(-1*1))
        """

        pass


    def alpha_008(self):
        """
        :return: (-1*rank(((sum(open,5)*sum(returns,5))-delay((sum(open,5)*sum(returns,5)),10))))
        """

        sumprod = self.open.rolling(5).sum() * self.rate.rolling(5).sum()

        return -1 * (sumprod - sumprod.shift(10)).rank(pct=True)

    def alpha_022(self):
        """
        :return: (-1 *(delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))
        """

        return -1 * self.high.corrwith(self.volume.shift(5)).diff(5) * \
            stddev(self.close, 20).rank(pct=True)


    def alpha_041(self):
        """
        :return: (((high * low)^0.5) - vwap)
        """

        return np.power((self.high * self.low), 0.5) - self.vwap


    def alpha_042(self):
        """
        :return: (rank((vwap - close)) / rank((vwap + close)))
        """

        return (self.vwap - self.close).rank(pct=True) / (self.vwap + self.close).rank(pct=True)


    def alpha_044(self):
        """
        :return: (-1 * correlation(high, rank(volume), 5))
        """

        return -1 * self.high.corrwith(self.volume.rank(pct=True).shift(5))


    def alpha_053(self):
        """
        :return: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        """

        return -1 * ((self.close - self.low) - (self.high - self.close) /
                     (self.close - self.low)).diff(9)



    def alpha_054(self):
        """
        :return: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        """

        return -1 * ((self.low - self.close) * np.power(self.open, 5)) / \
            ((self.low - self.high) * np.power(self.close, 5))

    def alpha_101(self):
        """
        :return: ((close - open) / ((high - low) + .001))
        """

        return (self.close - self.open) / (self.high - self.low + 0.001)
