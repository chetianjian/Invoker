from utils import *
from MongoLoader import Mongo


class Alpha101(Mongo):
    def __init__(self):
        super().__init__()
        self.alpha101 = dict.fromkeys(["0"*(3-len(str(_)))+str(_) for _ in range(1, 102)])


    @property
    def alpha_001(self):
        """
        :return: (rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close),2.),5))-0.5)
        """

        tmp1 = self.volume.diff(periods=1).rank(axis=1, pct=True)
        tmp2 = ((self.close - self.open) / self.open).rank(axis=1, pct=True)
        tmp3 = -tmp1.iloc[-6:, :].corrwith(tmp2.iloc[-6:, :]).dropna()

        return tmp3.dropna()


    @property
    def alpha_002(self):
        """
        :return: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        """

        if self.alpha101["002"] is None:
            tmp1 = np.log(self.volume).diff(2).rank(pct=True)
            tmp2 = (((self.close - self.open) / self.open).rank(pct=True))

            self.alpha101["002"] = -1 * corr(tmp1, tmp2, 6)

        return self.alpha101["002"]


    @property
    def alpha_003(self):
        """
        :return: (-1 * correlation(rank(open), rank(volume), 10))
        """

        if self.alpha101["003"] is None:

            self.alpha101["003"] = -1 * corr(self.open.rank(pct=True),
                                             self.volume.rank(pct=True),
                                             10)

        return self.alpha101["003"]


    @property
    def alpha_004(self):
        """
        :return: (-1 * Ts_Rank(rank(low), 9))
        """

        if self.alpha101["004"] is None:

            self.alpha101["004"] = -1 * ts_rank(self.low.rank(pct=True), window=9)

        return self.alpha101["004"]


    @property
    def alpha_005(self):
        """
        :return: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
        """

        if self.alpha101["005"] is None:
            tmp1 = self.open - self.vwap.rolling(10).sum() / 10
            tmp2 = (self.close - self.vwap).rank(pct=True)

            self.alpha101["005"] = -1 * tmp1.rank(pct=True) * np.absolute(tmp2)

        return self.alpha101["005"]


    @property
    def alpha_006(self):
        """
        :return: (-1 * correlation(open, volume, 10))
        """

        if self.alpha101["006"] is None:

            self.alpha101["006"] = -1 * corr(self.open, self.volume, 10)

        return self.alpha101["006"]


    @property
    def alpha_007(self):
        """
        :return: ((adv20<volume)?((-1*ts_rank(abs(delta(close,7)),60))*sign(delta(close,7))):(-1*1))
        """

        return None


    @property
    def alpha_008(self):
        """
        :return: (-1*rank(((sum(open,5)*sum(returns,5))-delay((sum(open,5)*sum(returns,5)),10))))
        """

        if self.alpha101["008"] is None:
            tmp = self.open.rolling(5).sum() * self.rate.rolling(5).sum()

            self.alpha101["008"] = -1 * (tmp - tmp.shift(10)).rank(pct=True)

        return self.alpha101["008"]






    @property
    def alpha_011(self):
        """
        :return: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
        """

        if self.alpha101["011"] is None:
            tmp1 = ts_max(self.vwap - self.close, 3).rank(pct=True)
            tmp2 = ts_min(self.vwap - self.close, 3).rank(pct=True)
            tmp3 = self.volume.diff(3).rank(pct=True)

            self.alpha101["011"] = (tmp1 + tmp2) * tmp3

        return self.alpha101["011"]


    @property
    def alpha_012(self):
        """
        :return: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        """

        if self.alpha101["012"] is None:
            tmp1 = np.sign(self.volume.diff(1))
            tmp2 = -1 * self.close.diff(1)

            self.alpha101["012"] = tmp1 * tmp2

        return self.alpha101["012"]


    @property
    def alpha_013(self):
        """
        :return: (-1 * rank(covariance(rank(close), rank(volume), 5)))
        """

        if self.alpha101["013"] is None:
            tmp1 = self.close.rank(pct=True)
            tmp2 = self.volume.rank(pct=True)
            tmp = covar(df1=tmp1, df2=tmp2, window=5)

            self.alpha101["013"] = -1 * tmp.rank(pct=True)

        return self.alpha101["013"]


    @property
    def alpha_014(self):
        """
        :return: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        """

        if self.alpha101["014"] is None:
            tmp1 = self.rate.diff(3).rank(pct=True)
            tmp2 = corr(self.open, self.volume, 10)

            self.alpha101["014"] = -1 * tmp1 * tmp2

        return self.alpha101["014"]


    @property
    def alpha_015(self):
        """
        :return: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        """

        if self.alpha101["015"] is None:
            tmp1 = self.high.rank(pct=True)
            tmp2 = self.volume.rank(pct=True)
            tmp = corr(tmp1, tmp2, 3)

            self.alpha101["015"] = -1 * tmp.rolling(window=3).sum()

        return self.alpha101["015"]


    @property
    def alpha_016(self):
        """
        :return: (-1 * rank(covariance(rank(high), rank(volume), 5)))
        """

        if self.alpha101["016"] is None:
            tmp = covar(df1=self.high.rank(pct=True),
                                 df2=self.volume.rank(pct=True),
                                 window=5)

            self.alpha101["016"] = -1 * tmp.rank(pct=True)

        return self.alpha101["016"]


    @property
    def alpha_017(self):
        """
        :return: (((-1*rank(ts_rank(close,10)))*rank(delta(delta(close,1),1)))*rank(ts_rank((volume/adv20),5)))
        """

        if self.alpha101["017"] is None:
            adv20 = self.volume.rolling(20).mean()
            tmp1 = ts_rank(self.close, 10)
            tmp2 = self.close.diff(1).diff(1).rank(pct=True)
            tmp3 = ts_rank(self.volume.div(adv20), 5).rank(pct=True)

            self.alpha101["017"] = -1 * tmp1 * tmp2 * tmp3

        return self.alpha101["017"]


    @property
    def alpha_018(self):
        """
        :return: (-1*rank(((stddev(abs((close-open)),5)+(close-open))+correlation(close,open,10))))
        """

        if self.alpha101["018"] is None:
            tmp1 = stddev(np.absolute(self.close-self.open), window=5)
            tmp2 = self.close - self.open
            tmp3 = corr(self.close, self.open, 10)
            tmp = tmp1 + tmp2 + tmp3

            self.alpha101["018"] = -1 * tmp.rank(pct=True)

        return self.alpha101["018"]


    @property
    def alpha_019(self):
        """
        :return: ((-1*sign(((close-delay(close,7))+delta(close,7))))*(1+rank((1+sum(returns,250)))))
        """

        if self.alpha101["019"] is None:
            tmp1 = np.sign(self.close - self.close.shift(7) + self.close.diff(7))
            tmp2 = 1 + (1 + self.rate.rolling(window=250).sum()).rank(pct=True)

            self.alpha101["019"] = -1 * tmp1 * tmp2

        return self.alpha101["019"]


    @property
    def alpha_020(self):
        """
        :return: (((-1*rank((open-delay(high,1))))*rank((open-delay(close,1))))*rank((open-delay(low,1))))
        """

        if self.alpha101["020"] is None:
            tmp1 = (self.open - self.high.shift(1)).rank(pct=True)
            tmp2 = (self.open - self.close.shift(1)).rank(pct=True)
            tmp3 = (self.open - self.low.shift(1)).rank(pct=True)

            self.alpha101["020"] = -1 * tmp1 * tmp2 * tmp3

        return self.alpha101["020"]







    @property
    def alpha_022(self):
        """
        :return: (-1 *(delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))
        """

        if self.alpha101["022"] is None:
            tmp1 = corr(self.high, self.volume, 5).diff(5)
            tmp2 = stddev(self.close, 20).rank(pct=True)

            self.alpha101["022"] = -1 * tmp1 * tmp2

        return self.alpha101["022"]







    @property
    def alpha_025(self):
        """
        :return: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
        """

        if self.alpha101["025"] is None:
            adv20 = self.volume.rolling(20).mean()
            tmp = -1 * self.rate * adv20 * self.vwap * (self.high - self.close)

            self.alpha101["025"] = tmp.rank(pct=True)

        return self.alpha101["025"]







    @property
    def alpha_028(self):
        """
        :return: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        """

        if self.alpha101["028"] is None:
            adv20 = self.volume.rolling(20).mean()
            tmp1 = corr(adv20, self.low, 5)
            tmp2 = (self.high + self.low) / 2

            self.alpha101["028"] = scale(tmp1 + tmp2 - self.close)

        return self.alpha101["028"]







    @property
    def alpha_032(self):
        """
        :return: (scale(((sum(close,7)/7)-close))+(20*scale(correlation(vwap,delay(close,5),230))))
        """

        if self.alpha101["032"] is None:
            tmp1 = scale(self.close.rolling(7).mean() - self.close)
            tmp2 = 20 * scale(corr(self.vwap, self.close.shift(5), 230))

            self.alpha101["032"] = tmp1 + tmp2

        return self.alpha101["032"]







    @property
    def alpha_034(self):
        """
        :return: rank(((1-rank((stddev(returns,2)/stddev(returns,5))))+(1-rank(delta(close,1)))))
        """

        if self.alpha101["034"] is None:
            tmp1 = 1 - (stddev(self.rate, 2).div(stddev(self.rate, 5))).rank(pct=True)
            tmp2 = 1 - self.close.diff(1).rank(pct=True)

            self.alpha101["034"] = (tmp1 + tmp2).rank(pct=True)

        return self.alpha101["034"]


    @property
    def alpha_035(self):
        """
        :return: ((Ts_Rank(volume,32)*(1-Ts_Rank(((close+high)-low),16)))*(1-Ts_Rank(returns,32)))
        """

        if self.alpha101["035"] is None:
            tmp1 = ts_rank(self.volume, 32)
            tmp2 = 1 - ts_rank(self.close+self.high-self.low, 16)
            tmp3 = 1 - ts_rank(self.rate, 32)

            self.alpha101["035"] = tmp1 * tmp2 * tmp3

        return self.alpha101["035"]







    @property
    def alpha_038(self):
        """
        :return: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        """

        if self.alpha101["038"] is None:
            tmp1 = ts_rank(self.close, 10).rank(pct=True)
            tmp2 = (self.close.div(self.open)).rank(pct=True)

            self.alpha101["038"] = -1 * tmp1 * tmp2

        return self.alpha101["038"]






    @property
    def alpha_040(self):
        """
        :return: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
        """

        if self.alpha101["040"] is None:
            tmp1 = stddev(self.high, 10).rank(pct=True)
            tmp2 = corr(self.high, self.volume, 10)

            self.alpha101["040"] = -1 * tmp1 * tmp2

        return self.alpha101["040"]


    @property
    def alpha_041(self):
        """
        :return: (((high * low)^0.5) - vwap)
        """

        if self.alpha101["041"] is None:

            self.alpha101["041"] = np.power((self.high * self.low), 0.5) - self.vwap

        return self.alpha101["041"]


    @property
    def alpha_042(self):
        """
        :return: (rank((vwap - close)) / rank((vwap + close)))
        """

        if self.alpha101["042"] is None:
            tmp1 = (self.vwap - self.close).rank(pct=True)
            tmp2 = (self.vwap + self.close).rank(pct=True)

            self.alpha101["042"] = tmp1.div(tmp2)

        return self.alpha101["042"]


    @property
    def alpha_043(self):
        """
        :return: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        """

        if self.alpha101["043"] is None:
            adv20 = self.volume.rolling(20).mean()
            tmp1 = ts_rank(self.volume.div(adv20), 20)
            tmp2 = ts_rank(-1 * self.close.diff(7), 8)

            self.alpha101["043"] = tmp1 * tmp2

        return self.alpha101["043"]


    @property
    def alpha_044(self):
        """
        :return: (-1 * correlation(high, rank(volume), 5))
        """

        if self.alpha101["044"] is None:

            self.alpha101["044"] = -1 * corr(self.high, self.volume.rank(pct=True), 5)

        return self.alpha101["044"]






    @property
    def alpha_045(self):
        """
        :return: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) *
                 correlation(close, volume, 2)) *
                 rank(correlation(sum(close, 5), sum(close, 20), 2))))
        """

        if self.alpha101["045"] is None:
            tmp1 = self.close.shift(5).rolling(20).mean().rank(pct=True)
            tmp2 = corr(self.close, self.volume, 2)
            tmp3 = corr(self.close.rolling(5).sum(), self.close.rolling(20).sum(), 2).rank(pct=True)

            self.alpha101["045"] = -1 * tmp1 * tmp2 * tmp3

        return self.alpha101["045"]


    @property
    def alpha_047(self):
        """
        :return: ((((rank((1 / close)) * volume) / adv20) *
                 ((high * rank((high - close))) /
                 (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
        """

        if self.alpha101["047"] is None:
            adv20 = self.volume.rolling(20).mean()
            tmp1 = (1 / self.close).rank(pct=True) * self.volume / adv20
            tmp2 = self.high * (self.high - self.close).rank(pct=True)
            tmp3 = self.high.rolling(5).mean()
            tmp4 = self.vwap.diff(5).rank(pct=True)

            self.alpha101["047"] = (tmp1 * tmp2).div(tmp3) - tmp4

        return self.alpha101["047"]






    @property
    def alpha_050(self):
        """
        :return: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        """

        if self.alpha101["050"] is None:
            self.alpha101["050"] = -1 * ts_max(corr(self.volume.rank(pct=True),
                                                    self.vwap.rank(pct=True), 5), 5)

        return self.alpha101["050"]







    @property
    def alpha_052(self):
        """
        :return: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
                 rank(((sum(returns, 240) - sum(returns, 20)) / 220))) *
                 ts_rank(volume, 5))
        """

        if self.alpha101["052"] is None:
            tmp1 = ts_min(self.low, 5)
            tmp2 = tmp1.shift(5)
            tmp3 = self.rate.rolling(240).sum()
            tmp4 = self.rate.rolling(20).sum()
            tmp5 = ts_rank(self.volume, 5)

            self.alpha101["052"] = (-1 * tmp1 + tmp2) * ((tmp3 - tmp4) / 220).rank(pct=True) * tmp5

        return self.alpha101["052"]


    @property
    def alpha_053(self):
        """
        :return: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        """

        if self.alpha101["053"] is None:
            self.alpha101["053"] = -1 * ((self.close - self.low) - (self.high - self.close).div(
                (self.close - self.low)).diff(9))

        return self.alpha101["053"]







    @property
    def alpha_054(self):
        """
        :return: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        """

        if self.alpha101["054"] is None:
            tmp1 = (self.low - self.close) * np.power(self.open, 5)
            tmp2 = (self.low - self.high) * np.power(self.close, 5)

            self.alpha101["054"] = -1 * tmp1.div(tmp2)

        return self.alpha101["054"]



    @property
    def alpha_056(self):
        """
        :return: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) *
                 rank((returns * cap)))))
        """

        if self.alpha101["056"] is None:
            tmp1 = self.rate.rolling(10).sum()
            tmp2 = (self.rate.rolling(2).sum()).rolling(3).sum()
            tmp3 = (self.rate * self.mv).rank(pct=True)

            self.alpha101["056"] = -1 * (tmp1.div(tmp2)).rank(pct=True) * tmp3

        return self.alpha101["056"]


    @property
    def alpha_057(self):
        """
        :return: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
        """

        # TO DO:
        return None





    @property
    def alpha_061(self):
        """
        :return: (rank((vwap-ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
        """

        if self.alpha101["061"] is None:
            adv180 = self.volume.rolling(window=180).mean()
            tmp1 = self.vwap - ts_min(self.vwap, 16).rank(pct=True)
            tmp2 = corr(self.vwap, adv180, 18).rank(pct=True)

            self.alpha101["061"] = (tmp1 < tmp2) * 1

        return self.alpha101["061"]





    @property
    def alpha_075(self):
        """
        :return: (rank(correlation(vwap,volume,4.24304)) <
                 rank(correlation(rank(low),rank(adv50),12.4413)))
        """

        if self.alpha101["075"] is None:
            adv50 = self.volume.rolling(window=50).mean()
            tmp1 = corr(self.vwap, self.volume, 4).rank(pct=True)
            tmp2 = corr(self.low.rank(pct=True), adv50.rank(pct=True), 12)

            self.alpha101["075"] = (tmp1 < tmp2) * 1

        return self.alpha101["075"]








    @property
    def alpha_101(self):
        """
        :return: ((close - open) / ((high - low) + .001))
        """

        if self.alpha101["101"] is None:
            self.alpha101["101"] = (self.close - self.open).div(self.high - self.low + 0.001)

        return self.alpha101["101"]


