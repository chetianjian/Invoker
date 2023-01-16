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
    def alpha_026(self):
        """
        :return: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        """

        if self.alpha101["026"] is None:
            tmp1 = ts_rank(self.volume, 5)
            tmp2 = ts_rank(self.high, 5)
            tmp = corr(tmp1, tmp2, 5)

            self.alpha101["026"] = -1 * ts_max(tmp, 3)

        return self.alpha101["026"]







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
    def alpha_029(self):
        """
        :return: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank(
                    (-1 * rank(delta((close - 1), 5)))
                )), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
        """

        if self.alpha101["029"] is None:
            tmp = -1 * (self.close - 1).diff(5).rank(pct=True)
            tmp = np.log(ts_min(tmp.rank(pct=True).rank(pct=True), 2))
            tmp = scale(tmp).rank(pct=True).rank(pct=True)
            tmp = tmp.rolling(window=5, min_periods=1).min()

            self.alpha101["029"] = tmp + ts_rank(-1*self.rate.shift(6), 5)

        return self.alpha101["029"]


    @property
    def alpha_030(self):
        """
        :return: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +
                    sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
        """

        if self.alpha101["030"] is None:
            tmp1 = np.sign(self.close - self.close.shift(1))
            tmp2 = np.sign(self.close.shift(1) - self.close.shift(2))
            tmp3 = np.sign(self.close.shift(2) - self.close.shift(3))
            tmp4 = (tmp1 + tmp2 + tmp3).rank(pct=True)
            tmp5 = (1 - tmp4) * self.volume.rolling(5).sum()

            self.alpha101["030"] = tmp5 / self.volume.rolling(20).sum()

        return self.alpha101["030"]


    @property
    def alpha_031(self):
        """
        :return: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) +
                    rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
        """

        if self.alpha101["031"] is None:
            adv20 = self.volume.rolling(20).mean()
            tmp1 = -1 * self.close.diff(10).rank(pct=True).rank(pct=True)
            tmp1 = decay_linear(tmp1, 10).rank(pct=True).rank(pct=True).rank(pct=True)
            tmp2 = (-1 * self.close.diff(3)).rank(pct=True)
            tmp3 = np.sign(scale(corr(adv20, self.low, 12)))

            self.alpha101["031"] = tmp1 + tmp2 + tmp3

        return self.alpha101["031"]


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
    def alpha_033(self):
        """
        :return: rank((-1 * ((1 - (open / close))^1)))
        """

        if self.alpha101["033"] is None:

            tmp = 1 - self.open / self.close

            self.alpha101["033"] = -1 * tmp.rank(pct=True)

        return self.alpha101["033"]


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
    def alpha_036(self):
        """
        :return:  (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) +
                  (0.7 * rank((open - close)))) +
                  (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) +
                  rank(abs(correlation(vwap, adv20, 6)))) +
                  (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
        """

        if self.alpha101["036"] is None:
            adv20 = self.volume.rolling(20).mean()
            tmp1 = 2.21 * corr((self.close - self.open), self.volume.shift(1), 15).rank(pct=True)
            tmp2 = 0.7 * (self.open - self.close).rank(pct=True)
            tmp3 = 0.73 * ts_rank((-1 * self.rate).shift(6), 5).rank(pct=True)
            tmp4 = corr(self.vwap, adv20, 6)
            tmp5 = 0.6 * ((self.close.rolling(200).sum() - self.open) *
                          (self.close-self.open)).rank(pct=True)

            self.alpha101["036"] = tmp1 + tmp2 + tmp3 + tmp4 + tmp5

        return self.alpha101["036"]


    @property
    def alpha_037(self):
        """
        :return:  (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        """

        if self.alpha101["037"] is None:
            tmp1 = corr((self.open - self.close).shift(1), self.close, 200).rank(pct=True)
            tmp2 = (self.open - self.close).rank(pct=True)

            self.alpha101["037"] = tmp1 + tmp2

        return self.alpha101["037"]


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
    def alpha_039(self):
        """
        :return: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) *
                    (1 + rank(sum(returns, 250))))
        """

        if self.alpha101["039"] is None:
            adv20 = self.volume.rolling(20).mean()
            tmp1 = self.close.diff(7)
            tmp2 = 1 - decay_linear(self.volume / adv20, 9).rank(pct=True)
            tmp3 = 1 + self.rate.rolling(250).sum().rank(pct=True)

            self.alpha101["039"] = -1 * (tmp1 * tmp2).rank(pct=True) * tmp3

        return self.alpha101["039"]


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
    def alpha_051(self):
        """
        :return: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) <
        (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        """

        tmp1 = (self.close.shift(20) - self.close.shift(10)) / 10
        tmp2 = (self.close.shift(10) - self.close) / 1010
        print(tmp1, tmp2)
        return None





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
    def alpha_055(self):
        """
        :return: (-1 * correlation(
                       rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))),
                       rank(volume),
                       6))
        """

        if self.alpha101["055"] is None:
            tmp1 = self.close - ts_min(self.low, 12)
            tmp2 = ts_max(self.high, 12) - ts_min(self.low, 12)
            tmp3 = self.volume.rank(pct=True)

            self.alpha101["055"] = -1 * corr((tmp1 / tmp2).rank(pct=True), tmp3, 6)

        return self.alpha101["055"]


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

        if self.alpha101["057"] is None:
            tmp1 = self.close - self.vwap
            tmp2 = decay_linear(ts_argmax(self.close, 30).rank(pct=True), 2)

            self.alpha101["057"] = -1 * tmp1 / tmp2

        return self.alpha101["057"]


    @property
    def alpha_061(self):
        """
        :return: (rank((vwap-ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
        """

        if self.alpha101["061"] is None:
            adv180 = self.volume.rolling(window=180).mean()
            tmp1 = self.vwap - ts_min(self.vwap, 16).rank(pct=True)
            tmp2 = corr(self.vwap, adv180, 18).rank(pct=True)

            self.alpha101["061"] = (tmp1 < tmp2).astype(int)

        return self.alpha101["061"]


    @property
    def alpha_062(self):
        """
        :return: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) <
                  rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
        """

        if self.alpha101["062"] is None:
            adv20 = self.volume.rolling(window=20).mean()
            tmp1 = corr(self.vwap, adv20.rolling(window=22).sum(), 10)
            tmp2 = 2 * self.open.rank(pct=True)
            tmp3 = ((self.high + self.low) / 2).rank(pct=True) + self.high.rank(pct=True)
            tmp4 = (tmp2 < tmp3).astype(int)
            tmp5 = -1 * tmp4.rank(pct=True)

            self.alpha101["062"] = (tmp1 < tmp5).astype(int)

        return self.alpha101["062"]



    @property
    def alpha_064(self):
        """
        :return: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),
                    sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) +
                    (vwap * (1 - 0.178404))), 3.69741))) * -1)
        """

        if self.alpha101["064"] is None:
            adv120 = self.volume.rolling(window=120).mean()
            tmp1 = (0.178404 * self.open + 0.821596 * self.low).rolling(window=13).sum()
            tmp2 = adv120.rolling(window=13).sum()
            tmp3 = corr(tmp1, tmp2, 17).rank(pct=True)
            tmp4 = 0.178404 * ((self.high + self.low) / 2) + 0.821596 * self.vwap
            tmp5 = tmp4.diff(4).rank(pct=True)

            self.alpha101["064"] = -1 * (tmp3 < tmp5)

        return self.alpha101["064"]


    @property
    def alpha_065(self):
        """
        :return: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))),
                                    sum(adv60, 8.6911),
                                    6.40374)) <
                        rank((open - ts_min(open, 13.635)))) * -1)
        """

        if self.alpha101["065"] is None:
            adv60 = self.volume.rolling(window=60).mean()
            tmp1 = 0.00817205 * self.open + 0.99182795 * self.vwap
            tmp2 = corr(tmp1, adv60.rolling(window=9).sum(), 6).rank(pct=True)
            tmp3 = self.open - ts_min(self.open, 14).rank(pct=True)

            self.alpha101["065"] = -1 * (tmp2 < tmp3)

        return self.alpha101["065"]


    @property
    def alpha_068(self):
        """
        :return: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <
                   rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
        """

        if self.alpha101["068"] is None:
            adv15 = self.volume.rolling(window=15).mean()
            tmp1 = ts_rank(corr(self.high.rank(pct=True), adv15.rank(pct=True), 9), 14)
            tmp2 = 0.518371 * self.close + 0.481629 * self.low
            tmp3 = tmp2.diff(1).rank(pct=True)

            self.alpha101["068"] = -1 * (tmp1 < tmp3)

        return self.alpha101["068"]







    @property
    def alpha_074(self):
        """
        :return: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <
                   rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))),
                                         rank(volume), 11.4791))) * -1)
        """

        if self.alpha101["074"] is None:
            adv30 = self.volume.rolling(window=30).mean()
            tmp1 = corr(self.close, adv30.rolling(window=37).sum(), 15).rank(pct=True)
            tmp2 = (0.0261661 * self.high + 0.9738339 * self.vwap).rank(pct=True)
            tmp3 = corr(tmp2, self.volume.rank(pct=True), 11).rank(pct=True)

            self.alpha101["074"] = (tmp1 < tmp3).astype(int)

        return self.alpha101["074"]


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

            self.alpha101["075"] = (tmp1 < tmp2).astype(int)

        return self.alpha101["075"]



    @property
    def alpha_077(self):
        """
        :return: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
                    rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
        """

        if self.alpha101["077"] is None:
            adv40 = self.volume.rolling(window=40).mean()
            tmp1 = decay_linear((3 * self.high + self.low) / 2 - self.vwap - self.high, 20)
            tmp2 = decay_linear(corr((self.high + self.low) / 2, adv40, 3), 6)

            self.alpha101["077"] = np.minimum(tmp1, tmp2)

        return self.alpha101["077"]





    @property
    def alpha_078(self):
        """
        :return: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
                                   sum(adv40, 19.7428), 6.83313))^
                                rank(correlation(rank(vwap), rank(volume), 5.77492)))
        """

        if self.alpha101["078"] is None:
            adv40 = self.volume.rolling(window=40).mean()
            tmp1 = (0.352233 * self.low + 0.647767 * self.vwap).rolling(window=20).sum()
            tmp2 = adv40.rolling(window=20).sum()
            tmp3 = corr(tmp1, tmp2, 7).rank(pct=True)
            tmp4 = corr(self.vwap.rank(pct=True), self.volume.rank(pct=True), 6).rank(pct=True)

            self.alpha101["078"] = np.power(tmp3, tmp4)

        return self.alpha101["078"]








    @property
    def alpha_081(self):
        """
        :return:  ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) <
                    rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
        """

        if self.alpha101["081"] is None:
            adv10 = self.volume.rolling(window=10).mean()
            tmp1 = np.power(corr(self.vwap, adv10.rolling(window=50).sum(), 8).rank(pct=True), 4)
            tmp2 = np.log(tmp1.rank(pct=True).rolling(window=15).prod()).rank(pct=True)
            tmp3 = corr(self.vwap.rank(pct=True), self.volume.rank(pct=True), 5).rank(pct=True)

            self.alpha101["081"] = -1 * (tmp2 < tmp3)

        return self.alpha101["081"]




    @property
    def alpha_083(self):
        """
        :return: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) /
                  (((high - low) /
                  (sum(close, 5) / 5))
                  / (vwap - close)))
        """

        if self.alpha101["083"] is None:
            tmp1 = self.high - self.low
            tmp2 = self.close.rolling(window=5).sum() / 5
            tmp3 = (tmp1 / tmp2).shift(2).rank(pct=True)
            tmp4 = (self.volume.rank(pct=True)).rank(pct=True)
            tmp5 = self.close.rolling(window=5).sum() / 5
            tmp6 = tmp1 / tmp5
            tmp7 = tmp6 / (self.vwap - self.close)

            self.alpha101["083"] = tmp3 * tmp4 / tmp7

        return self.alpha101["083"]





    @property
    def alpha_085(self):
        """
        :return: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^
                rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
        """

        if self.alpha101["085"] is None:
            adv30 = self.volume.rolling(window=30).mean()
            tmp1 = corr(0.876703 * self.high + 0.123297 * self.close, adv30, 10).rank(pct=True)
            tmp2 = ts_rank((self.high + self.low) / 2, 4)
            tmp3 = ts_rank(self.volume, 10)
            tmp4 = corr(tmp2, tmp3, 7).rank(pct=True)

            self.alpha101["085"] = np.power(tmp1, tmp4)

        return self.alpha101["085"]


    @property
    def alpha_086(self):
        """
        :return: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) <
                 rank(((open + close) - (vwap + open)))) * -1)
        """

        if self.alpha101["086"] is None:
            adv20 = self.volume.rolling(window=20).mean()
            tmp1 = ts_rank(corr(self.close, adv20.rolling(15).sum(), 6), 20)
            tmp2 = -1 * ((self.open + self.close) - (self.vwap + self.open)).rank(pct=True)

            self.alpha101["086"] = (tmp1 < tmp2).astype(int)

        return self.alpha101["086"]





    @property
    def alpha_088(self):
        """
        :return: min(rank(decay_linear(((rank(open) + rank(low)) -
                    (rank(high) + rank(close))), 8.06882)),
                    Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728),
                    Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
        """

        if self.alpha101["088"] is None:
            adv60 = self.volume.rolling(window=60).mean()
            tmp1 = self.open.rank(pct=True) + self.low.rank(pct=True)
            tmp2 = self.high.rank(pct=True) + self.close.rank(pct=True)
            tmp3 = decay_linear(tmp1 - tmp2, 8).rank(pct=True)
            tmp4 = corr(ts_rank(self.close, 8), ts_rank(adv60, 21), 8)
            tmp5 = ts_rank(decay_linear(tmp4, 7), 3)

            self.alpha101["088"] = np.minimum(tmp3, tmp5)

        return self.alpha101["088"]




    @property
    def alpha_094(self):
        """
        :return: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,
                 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
        """

        if self.alpha101["094"] is None:
            adv60 = self.volume.rolling(window=60).mean()
            tmp1 = (self.vwap - ts_min(self.vwap, 12)).rank(pct=True)
            tmp2 = ts_rank(corr(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)

            self.alpha101["094"] = -1 * np.power(tmp1, tmp2)

        return self.alpha101["094"]


    @property
    def alpha_095(self):
        """
        :return:  (rank((open - ts_min(open, 12.4105))) <
                   Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351),
                                             sum(adv40, 19.1351),
                                             12.8742))^5), 11.7584))
        """

        if self.alpha101["095"] is None:
            adv40 = self.volume.rolling(window=40).mean()
            tmp1 = (self.open - ts_min(self.open, 12)).rank(pct=True)
            tmp2 = ((self.high + self.low) / 2).rolling(window=19).sum()
            tmp3 = ts_rank(np.power(corr(tmp2, adv40.rolling(window=19).sum(), 13).rank(pct=True), 5), 12)

            self.alpha101["095"] = (tmp1 < tmp3).astype(int)

        return self.alpha101["095"]






    @property
    def alpha_098(self):
        """
        :return:  (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -
                  rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open),
                  rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
        """

        if self.alpha101["098"] is None:
            adv5 = self.volume.rolling(window=5).mean()
            adv15 = self.volume.rolling(window=15).mean()
            tmp1 = decay_linear(corr(self.vwap, adv5.rolling(26).sum(), 5), 7).rank(pct=True)
            tmp2 = ts_argmin(corr(self.open.rank(pct=True), adv15.rank(pct=True), 21), 9)
            tmp3 = decay_linear(ts_rank(tmp2, 7), 8).rank(pct=True)

            self.alpha101["098"] = tmp1 - tmp3

        return self.alpha101["098"]


    @property
    def alpha_099(self):
        """
        :return: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <
                  (rank(correlation(low, volume, 6.28259))) * -1)
        """

        if self.alpha101["099"] is None:
            adv60 = self.volume.rolling(window=60).mean()
            tmp1 = ((self.high + self.low) / 2).rolling(window=20).sum()
            tmp2 = adv60.rolling(window=20).sum()
            tmp3 = corr(tmp1, tmp2, 9).rank(pct=True)
            tmp4 = -1 * corr(self.low, self.volume, 6).rank(pct=True)

            self.alpha101["099"] = (tmp3 < tmp4).astype(int)

        return self.alpha101["099"]






    @property
    def alpha_101(self):
        """
        :return: ((close - open) / ((high - low) + .001))
        """

        if self.alpha101["101"] is None:
            self.alpha101["101"] = (self.close - self.open).div(self.high - self.low + 0.001)

        return self.alpha101["101"]



