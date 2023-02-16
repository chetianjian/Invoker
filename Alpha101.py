from utils import *
from Mongo import Mongo


class Alpha101(Mongo):
    """
    Notations (Ternary operator):
    Ternary operator 'x ? y : z' means that:
        If condition x is TRUE then y;
        If condition x is FALSE then z

    For enhanced usage, when x, y, z are all pd.DataFrames with identical shape, term 'x ? y : z' means:
        Taking values of y where x is statisfied;
        Taking Values of z where (NOT x) is statisfied
    """

    def __init__(self):
        super().__init__()
        self.alpha101 = dict.fromkeys(["0"*(3-len(str(_)))+str(_) for _ in range(1, 102)])


    @property
    def alpha_001(self):
        """
        :return: (rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close),2.),5))-0.5)
        """

        if self.alpha101["001"] is None:
            tmp = self.close
            tmp[self.rate < 0] = stddev(self.rate, 20)

            self.alpha101["001"] = ts_argmax(np.power(tmp, 2), 5).rank(pct=True) - 0.5

        return self.alpha101["001"]


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

        if self.alpha101["007"] is None:
            adv20 = self.volume.rolling(window=20).mean()
            tmp = -1 * ts_rank(np.absolute(self.close.diff(7)), 60) * np.sign(self.close.diff(7))
            tmp[adv20 >= self.volume] = -1

            self.alpha101["007"] = tmp

        return self.alpha101["007"]


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
    def alpha_009(self):
        """
        :return: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : (
                 (ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))
                 ))
        """

        if self.alpha101["009"] is None:
            tmp = self.close.diff(1)
            result = -tmp
            cond1 = ts_min(tmp, 5) > 0
            cond2 = ts_max(tmp, 5) < 0
            result[cond1 | cond2] = tmp

            self.alpha101["009"] = result

        return self.alpha101["009"]


    @property
    def alpha_010(self):
        """
        :return: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
                 ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) :
                 (-1 * delta(close, 1)))))
        """

        if self.alpha101["010"] is None:
            tmp1 = self.close.diff(1)
            tmp2 = tmp1.copy()
            tmp2[ts_max(tmp1, 4) >= 0] = -tmp2
            tmp3 = tmp1.copy()
            tmp3[(0 >= ts_min(tmp1, 4))] = tmp2

            self.alpha101["010"] = tmp3.rank(pct=True)

        return self.alpha101["010"]


    @property
    def alpha_011(self):
        """
        :return: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *
                 rank(delta(volume, 3)))
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
    def alpha_021(self):
        """
        :return: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ?
        (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ?
        1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))
        """

        if self.alpha101["021"] is None:
            adv20 = self.volume.rolling(20).mean()
            tmp1 = self.close.rolling(8).sum() / 8
            tmp2 = stddev(self.close, 8)
            tmp3 = self.close.rolling(2).sum() / 2
            tmp4 = (tmp1 + tmp2 < tmp3)
            tmp5 = (tmp3 < tmp2 - tmp1)
            tmp6 = 2 * ((1 < (self.volume / adv20)) | ((self.volume / adv20) == 1.0)) - 1
            tmp5[~tmp5] = tmp6
            tmp5[tmp4] = -1

            self.alpha101["021"] = tmp5.astype(int)

        return self.alpha101["021"]


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
    def alpha_023(self):
        """
        :return:  (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        """

        if self.alpha101["023"] is None:
            tmp1 = (self.high.rolling(20).sum() / 20 < self.high)
            tmp2 = -1 * self.high.diff(2)
            tmp2[~tmp1] = 0

            self.alpha101["023"] = tmp2.astype(int)

        return self.alpha101["023"]


    @property
    def alpha_024(self):
        """
        :return: ((
        ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||
                 ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ?
                 (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))

        """

        if self.alpha101["024"] is None:
            tmp1 = (self.close.rolling(100).sum() / 100).diff(100)
            tmp2 = self.close.shift(100)
            tmp3 = ((tmp1 / tmp2) < 0.05) | ((tmp1 / tmp2) == 0.05)
            tmp4 = -1 * (self.close - ts_min(self.close, 100))
            tmp5 = -1 * self.close.diff(3)
            tmp5[tmp3] = tmp4

            self.alpha101["024"] = tmp5

        return self.alpha101["024"]


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

            self.alpha101["026"] = -1 * ts_max(corr(tmp1, tmp2, 5), 3)

        return self.alpha101["026"]


    @property
    def alpha_027(self):
        """
        :return: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ?
                 (-1 * 1) : 1)
        """

        if self.alpha101["027"] is None:
            tmp = (0.5 >= (corr(self.volume.rank(pct=True),
                                 self.vwap.rank(pct=True), 6).rolling(2).sum() / 2).rank(pct=True))
            tmp[~tmp] = -1

            self.alpha101["027"] = tmp.astype(int)

        return self.alpha101["027"]


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
        :return: (scale(((sum(close, 7) / 7) - close)) +
                 (20 * scale(correlation(vwap, delay(close, 5), 230))))
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
            tmp3 = corr(self.close.rolling(5).sum(),
                        self.close.rolling(20).sum(), 2).rank(pct=True)

            self.alpha101["045"] = -1 * tmp1 * tmp2 * tmp3

        return self.alpha101["045"]


    @property
    def alpha_046(self):
        """
        :return: ((0.25 < (
        ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
        )) ? (-1 * 1) :
                 (((((delay(close, 20) - delay(close, 10)) / 10) -
                 ((delay(close, 10) - close) / 10)) < 0) ? 1 :
                 ((-1 * 1) * (close - delay(close, 1)))))
        """

        if self.alpha101["046"] is None:
            tmp1 = (self.close.shift(20) + self.close - 2 * self.close.shift(10)) / 10
            tmp2 = (tmp1 < 0)
            tmp2[~tmp2] = self.close.shift(1) - self.close
            tmp2[0.25 < tmp1] = -1

            self.alpha101["046"] = tmp2.astype(float)

        return self.alpha101["046"]


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
    def alpha_048(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *
                 delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) /
                 delay(close, 1))^2), 250))
        """

        if self.alpha101["048"] is None:
            tmp1 = (corr(self.close.diff(1), self.close.shift(1).diff(1), 250) *
                    self.close.diff(1) / self.close).replace([-np.inf, np.inf], np.nan)
            tmp2 = np.power((self.close.diff(1) / self.close.shift(1)).replace(
                [-np.inf, np.inf], np.nan), 2).rolling(window=250).sum()

            self.alpha101["048"] = tmp1.div(tmp2).replace([-np.inf, np.inf], np.nan)

        return self.alpha101["048"]



    @property
    def alpha_049(self):
        """
        :return: (((((delay(close, 20) - delay(close, 10)) / 10) -
                 ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ?
                 1 : ((-1 * 1) * (close - delay(close, 1))))
        """

        if self.alpha101["049"] is None:
            tmp1 = (((self.close.shift(20) + self.close -
                      2 * self.close.shift(10)) / 10) < -0.1)
            tmp2 = self.close.shift(1) - self.close
            tmp1[~tmp1] = tmp2

            self.alpha101["049"] = tmp1.astype(float)

        return self.alpha101["049"]


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
        :return: (((((delay(close, 20) - delay(close, 10)) / 10) -
                 ((delay(close, 10) - close) / 10)) < (-1 * 0.05))
                 ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        """

        if self.alpha101["051"] is None:
            tmp1 = ((self.close.shift(20) + self.close -
                     2 * self.close.shift(10)) / 10 < -0.05)
            tmp2 = self.close.shift(1) - self.close
            tmp1[~tmp1] = tmp2

            self.alpha101["051"] = tmp1.astype(float)

        return self.alpha101["051"]


    @property
    def alpha_052(self):
        """
        :return: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
                 rank(((sum(returns, 240) - sum(returns, 20)) / 220))) *
                 ts_rank(volume, 5))
        """

        if self.alpha101["052"] is None:
            tmp1 = ts_min(self.low, 5)
            tmp2 = self.rate.rolling(240).sum()
            tmp3 = self.rate.rolling(20).sum()

            self.alpha101["052"] = (-1 * tmp1 + tmp1.shift(5)) * \
                                   ((tmp2 - tmp3) / 220).rank(pct=True) * \
                                   ts_rank(self.volume, 5)

        return self.alpha101["052"]


    @property
    def alpha_053(self):
        """
        :return: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        """

        if self.alpha101["053"] is None:
            self.alpha101["053"] = -1 * ((self.close - self.low) - (self.high - self.close).div(
                (self.close - self.low)).diff(9)).replace([np.inf, -np.inf], np.nan)

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
        :return: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) -
                 ts_min(low, 12)))), rank(volume), 6))
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
    def alpha_058(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,
                 3.92795), 7.89291), 5.50322))
        """

        if self.alpha101["058"] is None:

            self.alpha101["058"] = -1 * ts_rank(decay_linear(corr(self.vwap, self.volume, 4), 8), 6)

        return self.alpha101["058"]


    @property
    def alpha_059(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) +
                 (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
        """

        if self.alpha101["059"] is None:

            self.alpha101["059"] = -1 * ts_rank(decay_linear(corr(self.vwap, self.volume, 4), 16), 8)

        return self.alpha101["059"]


    @property
    def alpha_060(self):
        """
        :return: (0 - (1 * ((2 * scale(rank((
        (((close - low) - (high - close)) / (high - low)) * volume)))) -
                 scale(rank(ts_argmax(close, 10)))
                 )))
        """

        if self.alpha101["060"] is None:
            tmp1 = scale(ts_argmax(self.close, 10).rank(pct=True))
            tmp2 = 2 * scale(((2 * self.close - self.low - self.high) *
                              self.volume / (self.high - self.low)).rank(pct=True))

            self.alpha101["060"] = tmp1 - tmp2

        return self.alpha101["060"]


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
    def alpha_063(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) -
                 rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,
                 37.2467), 13.557), 12.2883))) * -1)
        """

        if self.alpha101["063"] is None:
            adv180 = self.volume.rolling(window=180).mean()
            tmp1 = decay_linear(self.close.diff(2), 8).rank(pct=True)
            tmp2 = decay_linear(corr((0.318108 * self.vwap + 0.681892 * self.open),
                                     adv180.rolling(37).sum(), 14), 12).rank(pct=True)

            self.alpha101["063"] = -1 * (tmp1 - tmp2)

        return self.alpha101["063"]


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
    def alpha_066(self):
        """
        :return: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) +
                 Ts_Rank(decay_linear((
                 (((low * 0.96633) + (low * (1 - 0.96633))) - vwap) /
                 (open - ((high + low) / 2))),
                 11.4157), 6.72611)) * -1)
        """

        if self.alpha101["066"] is None:
            tmp1 = decay_linear(self.vwap.diff(4), 7).rank(pct=True)
            tmp2 = (self.low - self.vwap) / (self.open - (self.high + self.low) / 2)
            tmp2 = ts_rank(decay_linear(tmp2, 11), 7)

            self.alpha101["066"] = -1 * (tmp1 + tmp2)

        return self.alpha101["066"]


    @property
    def alpha_067(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,
                 IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
        """

        if self.alpha101["067"] is None:
            adv20 = self.volume.rolling(window=20).mean()
            tmp1 = (self.high - ts_min(self.high, 2)).rank(pct=True)
            tmp2 = corr(self.vwap, adv20, 6).rank(pct=True)

            self.alpha101["067"] = -1 * np.power(tmp1, tmp2)

        return self.alpha101["067"]


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
    def alpha_069(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^
                 Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),
                 9.0615)) * -1)
        """

        if self.alpha101["069"] is None:
            adv20 = self.volume.rolling(window=20).mean()
            tmp1 = ts_max(self.vwap.diff(3), 5).rank(pct=True)
            tmp2 = ts_rank(corr((0.490655 * self.close + 0.509345 * self.vwap), adv20, 5), 9)

            self.alpha101["069"] = -1 * np.power(tmp1, tmp2)

        return self.alpha101["069"]


    @property
    def alpha_070(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,
                 IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
        """

        if self.alpha101["070"] is None:
            adv50 = self.volume.rolling(window=50).mean()
            tmp = ts_rank(corr(self.close, adv50, 18), 18)

            self.alpha101["070"] = -1 * np.power(self.vwap.diff(1).rank(pct=True), tmp)

        return self.alpha101["070"]


    @property
    def alpha_071(self):
        """
        :return: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976),
                 Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948),
                 Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662),
                 4.4388))
        """

        if self.alpha101["071"] is None:
            adv180 = self.volume.rolling(window=180).mean()
            tmp1 = ts_rank(decay_linear(corr(ts_rank(self.close, 3),
                                             ts_rank(adv180, 12), 18), 4), 16)
            tmp2 = ts_rank(decay_linear(np.power(self.low + self.open -
                                                 2 * self.vwap, 2).rank(pct=True), 16), 4)

            self.alpha101["071"] = np.maximum(tmp1, tmp2)

        return self.alpha101["071"]


    @property
    def alpha_072(self):
        """
        :return: (rank(decay_linear(correlation(((high + low) / 2),
                 adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(
                 Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
        """

        if self.alpha101["072"] is None:
            adv40 = self.volume.rolling(window=40).mean()
            tmp1 = decay_linear(corr((self.high + self.low) / 2, adv40, 9), 10).rank(pct=True)
            tmp2 = decay_linear(corr(ts_rank(self.vwap, 4),
                                     ts_rank(self.volume, 19), 7), 3).rank(pct=True)

            self.alpha101["072"] = (tmp1 / tmp2).replace([np.inf, -np.inf], np.nan)

        return self.alpha101["072"]


    @property
    def alpha_073(self):
        """
        :return: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
                 Ts_Rank(decay_linear(((delta(((open * 0.147155) +
                 (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) +
                 (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)

        0.852845
        """

        if self.alpha101["073"] is None:
            tmp1 = decay_linear(self.vwap.diff(5), 3).rank(pct=True)
            tmp2 = 0.147155 * self.open + 0.852845 * self.low
            tmp3 = -1 * ts_rank(decay_linear((-1 * tmp2.diff(2) / tmp2).replace(
                [np.inf, -np.inf], np.nan), 3), 17)

            self.alpha101["073"] = np.maximum(tmp1, tmp3)

        return self.alpha101["073"]


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

            self.alpha101["074"] = -1 * (tmp1 < tmp3).astype(int)

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
    def alpha_076(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(
                 correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543),
                 19.383)) * -1)
        """

        if self.alpha101["076"] is None:
            adv81 = self.volume.rolling(window=81).mean()
            tmp1 = decay_linear(self.vwap.diff(1), 12).rank(pct=True)
            tmp2 = ts_rank(decay_linear(ts_rank(corr(self.low, adv81, 8), 20), 17), 19)

            self.alpha101["076"] = -1 * np.maximum(tmp1, tmp2)

        return self.alpha101["076"]


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
    def alpha_079(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector),
                 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
        """

        if self.alpha101["079"] is None:
            adv150 = self.volume.rolling(window=150).mean()
            tmp1 = (0.60733 * self.close + 0.39267 * self.open).diff(1).rank(pct=True)
            tmp2 = corr(ts_rank(self.vwap, 4), ts_rank(adv150, 9), 15)

            self.alpha101["079"] = (tmp1 < tmp2).astype(int)

        return self.alpha101["079"]


    @property
    def alpha_080(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),
                 IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
        """

        if self.alpha101["080"] is None:
            adv10 = self.volume.rolling(window=10).mean()
            tmp1 = np.sign((0.868128 * self.open + 0.131872 * self.high).diff(4)).rank(pct=True)
            tmp2 = ts_rank(corr(self.high, adv10, 5), 6)

            self.alpha101["080"] = -1 * np.power(tmp1, tmp2)

        return self.alpha101["080"]


    @property
    def alpha_081(self):
        """
        :return:  ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) <
                    rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
        """

        if self.alpha101["081"] is None:
            adv10 = self.volume.rolling(window=10).mean()
            tmp1 = np.power(corr(self.vwap, adv10.rolling(window=50).sum(), 8).rank(pct=True), 4)
            tmp2 = np.log(tmp1.rank(pct=True).rolling(window=15).apply(np.prod, raw=True)).rank(pct=True)
            tmp3 = corr(self.vwap.rank(pct=True), self.volume.rank(pct=True), 5).rank(pct=True)

            self.alpha101["081"] = -1 * (tmp2 < tmp3)

        return self.alpha101["081"]


    @property
    def alpha_082(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(
                 IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))),
                 17.4842), 6.92131), 13.4283)) * -1)
        """

        if self.alpha101["082"] is None:
            tmp1 = decay_linear(self.open.diff(1), 15).rank(pct=True)
            tmp2 = ts_rank(decay_linear(corr(self.volume, self.open, 17), 7), 13)

            self.alpha101["082"] = -1 * np.minimum(tmp1, tmp2)

        return self.alpha101["082"]


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
            tmp3 = tmp1.div(tmp2).shift(2).rank(pct=True)
            tmp4 = (self.volume.rank(pct=True)).rank(pct=True)
            tmp5 = self.close.rolling(window=5).sum() / 5
            tmp6 = tmp1.div(tmp5).div(self.vwap - self.close)

            self.alpha101["083"] = (tmp3 * tmp4 / tmp6).replace([np.inf, -np.inf], np.nan)

        return self.alpha101["083"]


    @property
    def alpha_084(self):
        """
        :return: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
                 delta(close, 4.96796))
        """

        if self.alpha101["084"] is None:
            tmp1 = ts_rank(self.vwap - ts_max(self.vwap, 15), 21)
            tmp2 = self.close.diff(5)

            self.alpha101["084"] = np.power(tmp1, tmp2)

        return self.alpha101["084"]


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
            tmp2 = (self.close - self.vwap).rank(pct=True)

            self.alpha101["086"] = -1 * ((tmp1 < tmp2).astype(int))

        return self.alpha101["086"]


    @property
    def alpha_087(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233),
                 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry),
                 close, 13.4132)), 4.89768), 14.4535)) * -1)
        """

        if self.alpha101["087"] is None:
            adv81 = self.volume.rolling(window=81).mean()
            tmp1 = decay_linear((0.369701 * self.close - 0.630299 * self.vwap).diff(2), 3).rank(pct=True)
            tmp2 = ts_rank(decay_linear(np.absolute(corr(adv81, self.close, 13)), 5), 14)

            self.alpha101["087"] = -1 * np.maximum(tmp1, tmp2)

        return self.alpha101["087"]


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

            self.alpha101["088"] = np.minimum(tmp3, ts_rank(decay_linear(tmp4, 7), 3))

        return self.alpha101["088"]


    @property
    def alpha_089(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))),
                 adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(
                 vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
        """

        if self.alpha101["089"] is None:
            adv10 = self.volume.rolling(window=10).mean()
            tmp1 = ts_rank(decay_linear(corr(self.low, adv10, 7), 6), 4)
            tmp2 = ts_rank(decay_linear(self.vwap.diff(3), 10), 15)

            self.alpha101["089"] = tmp1 - tmp2

        return self.alpha101["089"]


    @property
    def alpha_090(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(
                 adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
        """

        if self.alpha101["090"] is None:
            adv40 = self.volume.rolling(window=40).mean()
            tmp1 = self.close - ts_max(self.close, 5).rank(pct=True)
            tmp2 = ts_rank(corr(adv40, self.low, 5), 3)

            self.alpha101["090"] = -1 * np.power(tmp1, tmp2).replace([-np.inf, np.inf], np.nan)

        return self.alpha101["090"]


    @property
    def alpha_091(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry),
                 volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(
                 vwap, adv30, 4.01303), 2.6809))) * -1)
        """

        if self.alpha101["091"] is None:
            adv30 = self.volume.rolling(window=30).mean()
            tmp1 = ts_rank(decay_linear(decay_linear(corr(self.close, self.volume, 10), 16), 4), 5)
            tmp2 = decay_linear(corr(self.vwap, adv30, 4), 3).rank(pct=True)

            self.alpha101["091"] = -1 * (tmp1 - tmp2)

        return self.alpha101["091"]


    @property
    def alpha_092(self):
        """
        :return: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683),
                 Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
        """

        if self.alpha101["092"] is None:
            adv30 = self.volume.rolling(window=30).mean()
            tmp1 = ((self.high + self.low) / 2 + self.close) < (self.low + self.open)
            tmp2 = ts_rank(decay_linear(tmp1, 15), 19)
            tmp3 = ts_rank(decay_linear(corr(self.low.rank(pct=True), adv30.rank(pct=True), 8), 7), 7)

            self.alpha101["092"] = np.minimum(tmp2, tmp3)

        return self.alpha101["092"]


    @property
    def alpha_093(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193),
                 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))),
                 2.77377), 16.2664)))
        """

        if self.alpha101["093"] is None:
            adv81 = self.volume.rolling(window=81).mean()
            tmp1 = ts_rank(decay_linear(corr(self.vwap, adv81, 17), 20), 8)
            tmp2 = decay_linear((0.524434 * self.close - 0.475566 * self.vwap).diff(3), 16).rank(pct=True)

            self.alpha101["093"] = (tmp1 / tmp2).replace([-np.inf, np.inf], np.nan)

        return self.alpha101["093"]


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
            tmp3 = ts_rank(np.power(corr(tmp2, adv40.rolling(window=19).sum(),
                                         13).rank(pct=True), 5), 12)

            self.alpha101["095"] = (tmp1 < tmp3).astype(int)

        return self.alpha101["095"]


    @property
    def alpha_096(self):
        """
        :return: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151),
                 Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242),
                 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
        """

        if self.alpha101["096"] is None:
            adv60 = self.volume.rolling(window=60).mean()
            tmp1 = ts_rank(decay_linear(corr(self.vwap.rank(pct=True), self.volume.rank(pct=True), 4), 4), 8)
            tmp2 = decay_linear(ts_argmax(corr(ts_rank(self.close, 7), ts_rank(adv60, 4), 4), 13), 14)

            self.alpha101["096"] = -1 * np.maximum(tmp1, ts_rank(tmp2, 13))

        return self.alpha101["096"]


    @property
    def alpha_097(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),
        IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,
        7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
        """

        if self.alpha101["097"] is None:
            adv60 = self.volume.rolling(window=60).mean()
            tmp1 = decay_linear((0.721001 * self.low + 0.278999 * self.vwap).diff(3), 20).rank(pct=True)
            tmp2 = ts_rank(decay_linear(ts_rank(corr(ts_rank(self.low, 8),
                                                     ts_rank(adv60, 17), 5), 19), 16), 7)
            self.alpha101["097"] = -1 * (tmp1 - tmp2)

        return self.alpha101["097"]


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
        :return: ((rank(correlation(sum(((high + low) / 2), 19.8975),
                 sum(adv60, 19.8975), 8.8136)) <
                 (rank(correlation(low, volume, 6.28259))) * -1)
        """

        if self.alpha101["099"] is None:
            adv60 = self.volume.rolling(window=60).mean()
            tmp1 = ((self.high + self.low) / 2).rolling(window=20).sum()
            tmp2 = adv60.rolling(window=20).sum()
            tmp3 = corr(tmp1, tmp2, 9).rank(pct=True)
            tmp4 = corr(self.low, self.volume, 6).rank(pct=True)

            self.alpha101["099"] = -1 * (tmp3 < tmp4).astype(int)

        return self.alpha101["099"]


    @property
    def alpha_100(self):
        """
        Notice that industrial neutralization is omitted here.
        :return: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) -
                 (high - close)) / (high - low)) * volume)), IndClass.subindustry),
                 IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5)
                 - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
        """

        if self.alpha101["100"] is None:
            adv20 = self.volume.rolling(window=20).mean()
            tmp1 = scale(((2 * self.close - self.low - self.high).div(self.high - self.low).replace(
                [-np.inf, np.inf], np.nan) * self.volume).rank(pct=True))
            tmp2 = scale(corr(self.close, adv20.rank(pct=True), 5) - ts_argmin(self.close, 30).rank(pct=True))

            self.alpha101["100"] = -1 * (1.5 * tmp1 - tmp2) * (self.volume / adv20)

        return self.alpha101["100"]


    @property
    def alpha_101(self):
        """
        :return: ((close - open) / ((high - low) + .001))
        """

        if self.alpha101["101"] is None:
            self.alpha101["101"] = (self.close - self.open).div(self.high - self.low + 0.001)

        return self.alpha101["101"]

