from utils import *
from Mongo import Mongo

class Indicator(Mongo):

    def __init__(self):
        super().__init__()

        """
            Noted that, some indicators may also be shown in class "Factor" (but in lowercases).
        The main difference between them is that the lowcases stand for default parameter settings 
        and are expressed in python properties, while the uppercases would adopt user-identified 
        parameters and are expressed in python functions (e.g. window size; whether to use vwap 
        or close as daily prices; etc.).
        
            In fact, this class (Indicator) is the superclass of class Factor.
        """

        self.available_indicator = [
            "ADJ", "CCI", "RSI", "MACD", "OBV", "HT", "ROC", "TR", "ATR", "BP",
            "ULTSOC"
        ]
        self.indicator = dict.fromkeys(self.available_data)


    def OBV(self):
        """
        OBV_today = OBV_prev + volume_today         if close_today > close_prev
                  = OBV_prev + 0                    if close_today == close_prev
                  = OBV_prev - volume_today         if close_today < close_prev
        :return: On-Balance Volume （累积量能）
        """

        if self.indicator["OBV"] is None:
            assert self.volume is not None
            assert self.close is not None

            comparison = self.close.fillna(0).rolling(2).apply(
                lambda arr: 1 * (arr[1] > arr[0]) - 1 * (arr[1] < arr[0]))

            obv = self.volume.copy()

            for _ in range(len(obv) - 1):
                obv.iloc[_ + 1] = obv.iloc[_] + comparison.iloc[_ + 1] * obv.iloc[_ + 1]

            self.indicator["OBV"] = obv

        return self.indicator["OBV"]


    def CCI(self, window=10, closed=None):
        """
        :param window: int, default = 10.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: window日顺势指标。
                CCI := (TYP - MA(TYP, window)) / (0.015 * AVEDEV(TYP, window))
                TYP := (HIGH + LOW + CLOSE) / 3
        """

        if self.indicator["CCI"] is None:
            for dname in ["high", "low", "close"]:
                assert self.data[dname] is not None

            typ = (self.high + self.low + self.close) / 3

            self.indicator["CCI"] = (typ - typ.rolling(window, closed=closed).mean()) / \
               (0.015 * typ.rolling(window).apply(lambda x: arrAvgAbs(x)))

        return self.indicator["CCI"]


    def RSJ(self, window=48, closed=None):
        """
        :param window: int, default = 48.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 高频波动率不对称性
        """

        if self.indicator["RSJ"] is None:

            assert self.rate is not None

            positive = np.power(self.rate[self.rate > 0], 2)
            negative = np.power(self.rate[self.rate < 0], 2)

            self.indicator["RSJ"] = (positive.rolling(window=window, min_periods=1, closed=closed).sum() -
                                     negative.rolling(window=window, min_periods=1, closed=closed).sum()) / \
                                    self.rate.rolling(window=window, min_periods=1,
                                                      closed=closed).sum()

        return self.indicator["RSJ"]

    @property
    def TR(self):
        """
        True Range = MAX(High, Prior Close) - MIN(LOW, Prior Close)
        :return: True Range
        """

        prior = self.close.shift(1)
        return np.maximum(self.high, prior) - np.minimum(self.low, prior)


    def ATR(self, window=14, closed="left"):
        """
        :param window: int, default = 14.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: The Average True Range. Calculated by:
            "window-days-average of: max(high-low, abs(high-previous_close), abs(low-previous_close))"
        """

        if self.indicator["ATR"] is None:

            assert self.close is not None
            assert self.high is not None
            assert self.low is not None

            prior = self.close.shift(1)
            TR = np.maximum(self.high - self.low,
                            np.absolute(self.high - prior),
                            np.absolute(self.low - prior))

            self.indicator["ATR"] = TR.rolling(window=window, min_period=1, closed=closed).mean()

        return self.indicator["ATR"]


    def RSI(self, window=6, closed=None):
        """
        :param window: RSI window length, set default as 6.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: RSI values when RSI length takes 6.
        """

        assert self.rate is not None

        positive = self.rate[self.rate > 0]
        negative = self.rate[self.rate < 0]
        rs = positive.rolling(window=window, min_periods=1, closed=closed).sum() / \
             np.absolute(negative.rolling(window=window, min_periods=1, closed=closed).sum())

        return 100 - 100 / (1 + rs)


    def ROC(self, window=9):
        """
        ROC: Rate of Change indicator.
             100 * (当前收盘价 - window天前收盘价) / window天前收盘价
        :param window: int, default = 9.
        :return: window-days Rate of Change indicator
        """

        assert self.close is not None

        return 100 * self.close / self.close.shift(window) - 100


    @property
    def BP(self):
        """
        Buying Pressure = Close - MIN(Low, Prior Close)
        :return: Buying Pressure
        """
        return self.close - np.minimum(self.low, self.close.shift(1))


    @property
    def ULTSOC(self):
        """
        Ultimate Oscillator
        :return: (4 * AVG7 + 2 * AVG14 + 1 * AVG28) / 7
                 AVGn = n-days rolling sum of BP / n-days rolling sum of TR
                 BP: Buying Pressure
                 TR: True Range
        """

        avg7 = self.BP.rolling(7).sum() / self.TR.rolling(7).sum()
        avg14 = self.BP.rolling(14).sum() / self.TR.rolling(14).sum()
        avg28 = self.BP.rolling(28).sum() / self.TR.rolling(28).sum()

        return (4 * avg7 + 2 * avg14 + avg28) / 7





