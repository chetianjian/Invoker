from utils import *
from MongoLoader import Mongo


class Alpha101(Mongo):
    def __init__(self):
        super().__init__()


    def alpha_001(self):
        """
        :return: (rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close),2.),5))-0.5)
        """

        tmp1 = self.volume.diff(periods=1).rank(axis=1, pct=True)
        tmp2 = ((self.close - self.open) / self.open).rank(axis=1, pct=True)
        tmp3 = -tmp1.iloc[-6:, :].corrwith(tmp2.iloc[-6:, :]).dropna()
        return tmp3.dropna()


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

        return -1 * ts_rank(self.low.rank(pct=True), window=9)


    def alpha_005(self):
        """
        :return: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
        """

        tmp1 = self.open - self.vwap.rolling(10).sum() / 10
        tmp2 = (self.close - self.vwap).rank(pct=True)

        return -1 * tmp1.rank(pct=True) * np.absolute(tmp2)


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

        tmp = self.open.rolling(5).sum() * self.rate.rolling(5).sum()
        return -1 * (tmp - tmp.shift(10)).rank(pct=True)




    def alpha_012(self):
        """
        :return: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        """

        tmp1 = np.sign(self.volume.diff(1))
        tmp2 = -1 * self.close.diff(1)
        return tmp1 * tmp2


    def alpha_013(self):
        """
        :return: (-1 * rank(covariance(rank(close), rank(volume), 5)))
        """

        tmp1 = self.close.rank(pct=True)
        tmp2 = self.volume.rank(pct=True)
        tmp = covwith(df1=tmp1, df2=tmp2, window=5)
        return -1 * tmp.rank(pct=True)


    def alpha_014(self):
        """
        :return: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        """

        tmp1 = self.rate.diff(3).rank(pct=True)
        tmp2 = self.open.corrwith(self.volume.shift(10))

        return -1 * tmp1 * tmp2


    def alpha_015(self):
        """
        :return: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        """

        tmp = self.high.rank(pct=True).corrwith(self.volume.rank(pct=True).shift(3))
        return -1 * tmp.rolling(window=3).sum()


    def alpha_016(self):
        """
        :return: (-1 * rank(covariance(rank(high), rank(volume), 5)))
        """

        tmp = covwith(df1=self.high.rank(pct=True),
                             df2=self.volume.rank(pct=True),
                             window=5)
        return -1 * tmp.rank(pct=True)


    def alpha_017(self):
        """
        :return: (((-1*rank(ts_rank(close,10)))*rank(delta(delta(close,1),1)))*rank(ts_rank((volume/adv20),5)))
        """

        adv20 = self.volume.rolling(20).mean()
        tmp1 = ts_rank(self.close, 10)
        tmp2 = self.close.diff(1).diff(1).rank(pct=True)
        tmp3 = ts_rank(self.volume / adv20, 5).rank(pct=True)

        return -1 * tmp1 * tmp2 * tmp3


    def alpha_018(self):
        """
        :return: (-1*rank(((stddev(abs((close-open)),5)+(close-open))+correlation(close,open,10))))
        """

        tmp = stddev(np.absolute(self.close-self.open), window=5) \
               + self.close - self.open \
               + self.close.corrwith(self.open.shift(10))
        return -1 * tmp.rank(pct=True)


    def alpha_019(self):
        """
        :return: ((-1*sign(((close-delay(close,7))+delta(close,7))))*(1+rank((1+sum(returns,250)))))
        """

        tmp1 = np.sign(self.close - self.close.shift(7) + self.close.diff(7))
        tmp2 = 1 + (1 + self.rate.rolling(window=250).sum()).rank(pct=True)

        return -1 * tmp1 * tmp2


    def alpha_020(self):
        """
        :return: (((-1*rank((open-delay(high,1))))*rank((open-delay(close,1))))*rank((open-delay(low,1))))
        """

        tmp1 = (self.open - self.high.shift(1)).rank(pct=True)
        tmp2 = (self.open - self.close.shift(1)).rank(pct=True)
        tmp3 = (self.open - self.low.shift(1)).rank(pct=True)

        return -1 * tmp1 * tmp2 * tmp3


    def alpha_022(self):
        """
        :return: (-1 *(delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))
        """

        tmp1 = self.high.corrwith(self.volume.shift(5)).diff(5)
        tmp2 = stddev(self.close, 20).rank(pct=True)

        return -1 * tmp1 * tmp2


    def alpha_025(self):
        """
        :return: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
        """

        adv20 = self.volume.rolling(20).mean()
        tmp = -1 * self.rate * adv20 * self.vwap * (self.high - self.close)
        return tmp.rank(pct=True)



    def alpha_038(self):
        """
        :return: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        """

        tmp1 = ts_rank(self.close, 10).rank(pct=True)
        tmp2 = (self.close / self.open).rank(pct=True)

        return -1 * tmp1 * tmp2








    def alpha_041(self):
        """
        :return: (((high * low)^0.5) - vwap)
        """

        return np.power((self.high * self.low), 0.5) - self.vwap







    def alpha_042(self):
        """
        :return: (rank((vwap - close)) / rank((vwap + close)))
        """

        tmp1 = (self.vwap - self.close).rank(pct=True)
        tmp2 = (self.vwap + self.close).rank(pct=True)

        return tmp1 / tmp2






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

        tmp1 = (self.low - self.close) * np.power(self.open, 5)
        tmp2 = (self.low - self.high) * np.power(self.close, 5)
        return -1 * tmp1 / tmp2







    def alpha_101(self):
        """
        :return: ((close - open) / ((high - low) + .001))
        """

        return (self.close - self.open) / (self.high - self.low + 0.001)
