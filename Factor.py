from utils import *
from MongoLoader import Mongo

class Factor(Mongo):
    def __init__(self):
        super().__init__()


    def indicator(self):
        """
        :return: Return -1.0 if today's yield < 0 otherwise 1.0
        """

        assert self.rate is not None

        return 2 * ((self.rate >= 0) - 0.5)


    # def adding_data(self, data: list[str]):
    #     """
    #     :param data: If you want to add some data later on.
    #     :return: re-run the __init__ function
    #     """
    #
    #     return self.__init__(data)


    def IC(self, factor, cumulative=False):
        """
        DataFrame IC
        :param factor: Input factor data.
        :param cumulative: Bool, return cumulative IC values or not.
        :return: cross-sectional IC values, or cumulative cross-sectional IC values.
        """

        assert self.rate is not None

        result = factor.shift(1).iloc[1:].apply(
            lambda row: row.corr(self.rate.loc[row.name]), axis=1)

        return pd.DataFrame(result.cumsum(), columns=["IC"]).rename(
            columns={"IC": "Cumulative IC"}) if cumulative else pd.DataFrame(result, columns=["IC"])


    def ic(self, factor, code=None):
        """
        序列 IC
        :param factor: Input factor data.
        :param code: Default as None. Otherwise, return the result of the objective stock code with format:
                "XXXXXX.YZ", such as "000001.SZ".
        :return: IC values for all stocks respectively, or just a single IC value or a specific stock.
        """

        assert self.rate is not None

        if not code:
            return factor.apply(
                lambda col: col.shift(1).corr(self.rate[col.name], min_periods=10), axis=0
            )

        else:
            return factor[code].shift(1).corr(self.rate[code], min_periods=10)


    def TVMA(self, window=6, closed=None):
        """
        :param window: int, default = 6.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: window日成交金额的移动平均值。
        """

        assert self.money is not None

        return self.money.rolling(window, closed=closed).mean()


    def BIAS(self, window=5, closed=None):
        """
        window 日乖离率
        :param window: int, default = 5.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: window日乖离率，100 * (收盘 - 前window日收盘平均值) / 前window日收盘平均值。
        """

        assert self.close is not None

        return 100 * (self.close - self.close.rolling(window, closed=closed).mean()) / \
               self.close.rolling(window, closed=closed).mean()


    def VOL(self, window=10, closed=None):
        """
        :param window: int, default = 10.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: window日平均换手率。
        """

        assert self.turnover is not None

        return self.turnover.rolling(window, closed=closed).mean()


    def CCI(self, window=10, closed=None):
        """
        :param window: int, default = 10.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: window日顺势指标。
                CCI := (TYP - MA(TYP, window)) / (0.015 * AVEDEV(TYP, window))
                TYP := (HIGH + LOW + CLOSE) / 3
        """

        for dname in ["high", "low", "close"]:
            assert self.data[dname] is not None

        typ = (self.high + self.low + self.close) / 3
        return (typ - typ.rolling(window, closed=closed).mean()) / \
               (0.015 * typ.rolling(window).apply(lambda x: arrAvgAbs(x)))


    def CCI_vwap(self, window=10, closed=None):
        """
        :param window: int, default = 10.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: window日顺势指标，使用 vwap 代替 HIGH, LOW, CLOSE 三者的平均值。
                CCI := (vwap - MA(vwap, window)) / (0.015 * AVEDEV(vwap, window))
        """

        assert self.vwap is not None

        return (self.vwap - self.vwap.rolling(window, closed=closed).mean()) / \
               (0.015 * self.vwap.rolling(window, closed=closed).apply(lambda x: arrAvgAbs(x)))


    def BollUp(self, window=20, closed=None):
        """
        :param window: int, default = 20.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 上轨线（布林线）指标。
                (MA(CLOSE, window) + 2 * STD(CLOSE, window)) / CLOSE
        """

        assert self.close is not None

        return (self.close.rolling(window, closed=closed).mean() +
                2 * self.close.rolling(window, closed=closed).std()) / self.close


    def Price_ema(self, which_price="close", window=10, fillna=False):
        """
        价格的指数移动平均线
        :param window: int, default = 20.
        :param which_price: str in ["close", "open", "high", "low", "vwap"], default = "close".
                使用哪种价格计算指数移动均线。
        :param fillna: If fill NaNs, default for False, otherwise input a value.
        :return: window日 价格的指数移动均线。
        """

        if which_price == "close":
            assert self.close is not None
            return EMA(self.close, window=window, fillna=fillna)
        if which_price == "open":
            assert self.open is not None
            return EMA(self.open, window=window, fillna=fillna)
        if which_price == "high":
            assert self.high is not None
            return EMA(self.high, window=window, fillna=fillna)
        if which_price == "low":
            assert self.low is not None
            return EMA(self.low, window=window, fillna=fillna)
        if which_price == "vwap":
            assert self.vwap is not None
            return EMA(self.vwap, window=window, fillna=fillna)


    def ReferencePrice(self, window=6, closed="left"):

        assert self.turnover is not None
        assert self.vwap is not None

        def prod(t):
            return (1 - self.turnover).rolling(t - 1, closed=closed).apply(np.nanprod)

        result = self.turnover.shift(1) * self.vwap.shift(1)
        for n in range(2, window + 1):
            result += self.turnover.shift(n) * prod(t=n) * self.vwap.shift(n)

        return result


    def AR(self, window=7, closed=None):
        """
        每天上涨的动力
        :param window: int, default = 7.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 100 * window日内 (当日high - 当日open)之和 / window日内 (当日open - 当日low)之和
        """

        for dname in ["high", "low", "open"]:
            assert self.data[dname] is not None

        result = 100 * (self.high - self.open).rolling(window, closed=closed).sum() / \
                 (self.open - self.low).rolling(window, closed=closed).sum()
        return result[~np.isinf(result)]


    def BR(self, window=7, closed=None):
        """
        锚定昨日的收盘
        :param window: int, default = 7.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: window日内 (当日high - 昨日close)之和 / window日内 (昨日close - 当日low)之和 × 100
        """

        for dname in ["high", "low", "close"]:
            assert self.data[dname] is not None

        result = 100 * (self.high - self.close.shift(1)).rolling(
                    window, closed=closed).sum() / \
                 (self.close.shift(1) - self.low).rolling(
                     window, closed=closed).sum()
        return result[~np.isinf(result)]


    def CR(self, window=7, closed=None):
        """
        复苏的动力
        :param window: int, default = 7.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return:
        """

        for dname in ["high", "low", "close"]:
            assert self.data[dname] is not None

        result = (self.close - self.low).rolling(window, closed=closed).sum() / \
                 (self.high - self.close).rolling(window, closed=closed).sum()
        return result[~np.isinf(result)]


    def ARBR(self, window=7, closed=None):
        """
        :param window: int, default = 7.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 因子 AR 与因子 BR 的差。
        """
        return self.AR(window=window, closed=closed) - self.BR(window=window, closed=closed)


    def ARCR(self, window=7, closed=None):
        """
        AR、CR结构不仅对称、相反，而且互补。
        :param window: int, default = 7.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 因子 AR 与因子 CR 的和。
        """
        return self.AR(window=window, closed=closed) - self.CR(window=window, closed=closed)


    def VDIFF(self, short=12, long=26):
        """
        DIFF线
        :param short: int, default = 12.
        :param long: int, default = 26.
        :return: fast = EMA(VOLUME，SHORT)
                 slow = EMA(VOLUME，LONG)
                 DIFF = fast - slow
                 DEA  = MA(DIFF, M)
                 MACD = DIFF - DEA
                 return DIFF
        """

        assert self.volume is not None

        return self.volume.ewm(alpha=2 / (short + 1)).mean() - \
               self.volume.ewm(alpha=2 / (long + 1)).mean()


    def VDEA(self, short=12, long=26, window=9, closed=None):
        """
        DEA线
        :param short: int, default = 12.
        :param long: int, default = 26.
        :param window: int, default = 9.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: fast = EMA(VOLUME，SHORT)
                 slow = EMA(VOLUME，LONG)
                 DIFF = fast - slow
                 DEA  = MA(DIFF, M)
                 MACD = DIFF - DEA
                 return DEA
        """

        return self.VDIFF(short=short, long=long).rolling(window, closed=closed).mean()


    def VMACD(self, short=12, long=26, window=9, closed=None):
        """
        MACD 线
        :param short: int, default = 12.
        :param long: int, default = 26.
        :param window: int, default = 9.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: fast = EMA(VOLUME，SHORT)
                 slow = EMA(VOLUME，LONG)
                 DIFF = fast - slow
                 DEA  = MA(DIFF, M)
                 MACD = DIFF - DEA
                 return VMACD
        """

        return self.VDIFF(short=short, long=long) - \
               self.VDEA(short=short, long=long, window=window, closed=closed)


    def VROC(self, window=6):
        """
        window日成交量变化的速率 VROC
        :param window: int, default = 6.
        :return: 成交量减 window 日前的成交量，再除以 window 日前的成交量，放大100倍，得到VROC值
        """

        assert self.volume is not None

        result = 100 * self.volume.shift(1) / self.volume.shift(window) - 100
        return result[~np.isinf(result)]


    def weighted_PV_trend_weekly(self, window=7, closed=None):
        """
        :param window: int, default = 7.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: (收盘价 - window日前收盘价) * window天总成交额 * window天内上涨天数 / window日前收盘价
        """

        assert self.close is not None
        assert self.money is not None

        valid_days = self.indicator().rolling(window, closed=closed).sum()
        return (self.close - self.close.shift(window)) * \
               self.money.rolling(window, closed=closed).sum() * \
               valid_days / self.close.shift(window)


    def Energy(self, window=3, closed=None):
        """
        推广动能定理 E = 1/2 * m * v^2，其中 m 由质量推广至成交量，v 由速度推广至收益
        :param window: int, default = 6.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 模仿动能定理，成交量 乘以 收益的平方
        """

        assert self.volume is not None
        assert self.rate

        result = self.volume * self.rate ** 2
        return result.rolling(window, closed=closed).sum()


    def IMPLUSE(self, window=5, closed=None):
        """
        推广冲量定理 I = F * t，即单位时间内物体所受合外力，所反映的便是 “力” 在时间上的累积。
        :param window: int, default = 5.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 模仿冲量公式，即收益率在时间上的累积
        """

        assert self.rate is not None

        return self.rate.rolling(window, closed=closed).apply(impluse)


    def Geometry(self, window=3, closed=None):
        """
        grad_desc_geometry 是一个梯度下降函数，给定 “1” 作为一个整体，指定一个所需的窗口 window，利用梯度
        下降，将 “1” 按照权重分给前 window 天，其中距离今天越远的日期所分得权重越小，因为各种不稳定短期因素
        对投资者记忆、情绪的影响会随着时间快速消逝，这里定义的速度即几何下降。
        :param window: int, default = 3.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 将前 window 日的收益率以几何级数形式表出。
        """

        assert self.rate is not None

        cr = grad_desc_geometry(w=window)
        weight = np.array([1 / window * cr ** i for i in range(window)])
        return self.rate.rolling(window, closed=closed).apply(
            lambda col: np.dot(col, weight))


    def CGO(self, window=6):

        for dname in ["close", "vwap", "turnover"]:
            assert self.data[dname] is not None

        def prod(x):
            return (1 - self.turnover).rolling(x - 1, closed="left").apply(np.nanprod)

        result = self.turnover.shift(1) * self.vwap.shift(1)
        for n in range(2, window + 1):
            result += self.turnover.shift(n) * prod(x=n) * self.vwap.shift(n)

        return (self.close.shift(1) - result) / self.close.shift(1)


    def TANH_MONEY(self, window=6, shift=(0, 0), squeeze=1, closed=None):
        """
        为什么我要使用 tanh 函数？因为我发现这是tanh函数极其优秀的性质（同样的性质也体现在 tan 身上，只不过要在另
        一个领域）。
        使用 tanh 最理想的地方，应是修正市场上投资者或极度狂热或极度畏惧的心理。首先，tanh函数在 0 处函数值为 0，
        代表了一个场外非投资者、或市场内空仓者比较平静的心态。其次，tanh函数在 0 处一阶导数值为 0，这代表了一旦
        投资者开始介入市场，其情绪即刻开始受到市场盈亏的影响，并且很显然这个值应该是一个正数，否则就代表投资者收益
        越高越胆怯，亏损越多反而越沉着，并且 1 也是一个比较中性的取值。最后，tanh函数在 0 处的二阶导数又变为 0，这
        代表当市场仅仅开始轻微上涨或者下跌时，对投资者还不会造成一种 “加速膨胀” 或 “加速畏惧” 的效应，但是我们从图象
        便可以很显然地看出来，tanh 的二阶导数是单调递减的，因此，这恰好在一定程度上修正了市场情绪，因为尽管亏损与收益
        总是以线性关系体现、最终也落实到线性关系（就是说，今天下跌四个点就是四个点，市场并不会要求投资者多支付损失，
        收益同理）。但是，市场投资者的情绪却总是随着市场的极端变化而加速变化。我们还可以很方便地证明得到，当 微分阶数
        大于 2时，此后就全部为 0 了，这完美地贴合了这个理论。
        更重要的一点，即便我们不假设投资者的情绪位于原点，我们也可以很方便地通过对 tanh 输入变量进行加减，达到平移、挤压、
        拉伸的目的。
        :param window: int, default = 6.
        :param shift: 需要平移的量，输入为 (a, b) 的数组，最终效果为：tanh(x - a) + b
        :param squeeze: 对 tanh 进行挤压或者拉伸的系数，注意这一项并不会与 shift[1] 叠加。
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 将前 window 日的成交额以 squeeze * tanh(x - shift[0]) + shift[1] 形式表出。
        """

        assert self.money is not None

        return (squeeze * np.tanh(self.money - shift[0]) + shift[1]).rolling(
            window, closed=closed).sum()


    def TANH_Balanced_Money(self, window=6, shift=(0, 0), squeeze=1, closed=None):

        """
        :param window: int, default = 6.
        :param shift: 需要平移的量，输入为 (a, b) 的数组，最终效果为：tanh(x - a) + b
        :param squeeze: 对 tanh 进行挤压或者拉伸的系数，注意这一项并不会与 shift[1] 叠加。
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 将前 window 日的成交额以 squeeze * tanh(x - shift[0]) + shift[1] 形式表出。
        """

        assert self.money is not None

        result = (squeeze * np.tanh(self.money - shift[0]) +
                  shift[1]) * (1 + self.rate)
        return result.rolling(window, closed=closed).sum()


    def TANH_Weekly_Cumulation(self, window=5, shift=(0, 0), squeeze=1, closed=None):
        """
        :param window: int, default = 5.
        :param shift: 需要平移的量，输入为 (a, b) 的数组，最终效果为：tanh(x - a) + b
        :param squeeze: 对 tanh 进行挤压或者拉伸的系数，注意这一项并不会与 shift[1] 叠加。
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 将前 window 日的成交额以 squeeze * tanh(x - shift[0]) + shift[1] 形式表出。
        """

        assert self.money is not None
        assert self.rate is not None

        result = squeeze * np.tanh(self.money * self.rate - shift[0]) + shift[1]
        return result.rolling(window=window, closed=closed).sum()


    def RSJ(self, window=7, closed=None):
        """
        :param window: int, default = 7.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: RSJ_7
        """

        assert self.rate is not None

        def serial(arr):
            diff = 0
            for r in arr:
                diff += np.sign(r) * r ** 2
            return diff

        return self.rate.rolling(window, closed=closed).apply(serial) / \
               self.rate.rolling(window, closed=closed).apply(
                   lambda col: np.nansum(col ** 2))


    def TURNOVER_VAR(self, window=12, closed=None):
        """
        换手率的方差（即市场参与度波动率）
        :param window: int, default = 12.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: Variance (volatility) of turnover_ratio
        """

        assert self.turnover is not None

        return self.turnover.rolling(window=window, closed=closed).var()


    def Price_Variation(self, window):
        """
        :param window: int, default = 12.
        :return: A measure to the volatility of stock price.
        """

        assert self.vwap is not None

        return (self.vwap - self.vwap.rolling(window).mean()) ** 2 / \
               self.vwap.rolling(window).var()


    def CP_self(self, window=20, rise=True, closed=None):
        """
        这种定义方法，相当于定义上影线或下影线与实体线长度之比的平方，再乘以今日换手率（进行 window 日标准化）这个乘数。
        :param window: int, default = 20. For window's day average turnover.
        :param rise: bool, default = True. If True then the function will denote rising, otherwise
         denote falling if False.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: CP_self
        """

        for dname in ["turnover", "mv", "high", "low", "open", "close"]:
            assert self.data[dname] is not None

        turnover_coef = 1 + mvNeutralize(self.turnover, self.mv)

        result = turnover_coef * ((self.high - self.open) /
                                  (self.close - self.open)) ** 2 if rise else \
            ((self.open - self.low) - (self.open - self.close)) ** 2

        return result.rolling(window=window, closed=closed).sum()


    def CP_Intraday(self):
        """
        :return: 通过分钟级别 close数据计算的分钟收益率计算当日各股票的 CP 因子值。其中：
                （1）将每天 9:30-11:30 以及 13:00-15:00 之间的 242 分钟，标记为第 1，2，3，...，242 分钟，称其为分钟序号。
                （2）使用每只股票每天 242 个的分钟收盘价，计算出 240 个分钟收益率。
                （3）计算每天 240 个分钟收益率的均值 mean 和标准差 std。
                （4）逐一检视当天 240 个分钟收益率，大于 mean+std 的部分为快速上涨区间，小于 mean-std 的部分为快速下跌区间。
                （5）分别计算快速上涨区间和快速下跌区间的分钟序号的中位数，用下跌中位数减去上涨中位数，得到日频因子 CP_Intraday。
        """

        assert self.data["close_min"] is not None

        return self.data["close_min"].groupby(level=0).agg(CP)


    def CP_Mean(self, window=20, fillna=False, closed=None, intraday=None):
        """
        :param window: int, default = 20.
        :param fillna: If fill NaNs, default for False, otherwise input a value.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :param intraday: If intraday factor already existed.
        :return: CP_Mean
        """

        assert self.mv is not None

        if intraday:
            result = intraday.rolling(window, closed=closed).mean()
            return mvNeutralize(df=result, mv=self.mv, fillna=fillna)

        else:
            result = self.CP_Intraday().rolling(window, closed=closed).mean()
            return mvNeutralize(df=result, mv=self.mv, fillna=fillna)


    def CP_Std(self, window=20, fillna=False, closed=None, intraday=None):
        """
        :param window: int, default = 20.
        :param fillna: If fill NaNs, default for False, otherwise input a value.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :param intraday: If intraday factor already existed.
        :return: CP_Std
        """

        assert self.mv is not None

        if intraday:
            result = intraday.rolling(window, closed=closed).std()
            return mvNeutralize(df=result, mv=self.mv, fillna=fillna)
        else:
            result = self.CP_self().rolling(window, closed=closed).std()
            return mvNeutralize(df=result, mv=self.mv, fillna=fillna)


    def CP_Mean_Rank_Ascending(self, window=20, closed=None):
        """
        :param window: int, default = 20.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: Return CP_Mean in an ascending rank order, starting from 1.
        """

        cpm = self.CP_Mean(window=window, closed=closed)
        for row in range(len(cpm)):
            cpm.iloc[row, :] = np.argsort(a=cpm.iloc[row, :], kind="quicksort")
        return cpm + 2


    def CP_Std_Rank_Descending(self, window=20, closed=None):
        """
        :param window: int, default = 20.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: Return CP_Std in a descending rank order, starting from 1.
        """

        cps = self.CP_Std(window=window, closed=closed)
        for row in range(len(cps)):
            cps.iloc[row, :] = np.argsort(a=-cps.iloc[row, :], kind="quicksort")
        return cps + 2


    def Monthly_CP(self, window=20, closed=None):
        """
        Confidence Persistence: 信心持久度。若内幕消息可信度低，那么很快会被辟谣，因此股价维持天数会很短。
        此处我以 20 日作为一个检验标准，即在 window = 20 的窗口内，观察市场走势特性。
        :param window: int, default = 20.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: Confidence Persistence
        """

        cpmean = self.CP_Mean_Rank_Ascending(window=window, closed=closed)
        cpstd = self.CP_Std_Rank_Descending(window=window, closed=closed)
        assert cpmean.shape == cpstd.shape
        return cpmean + cpstd


    def Long_Power(self, window=13):
        """
        多头的一种度量方式
        :param window: int, default = 13.
        :return: (今日最高价 - EMA(close, 13)) / close
        """

        assert self.high is not None
        assert self.close is not None

        return (self.high - self.Price_ema(
            which_price="close", window=window)) / self.close


    def Short_Power(self, window=13):
        """
        空头的一种度量方式
        :param window: int, default = 13.
        :return: (今日最低价 - EMA(close, 13)) / close
        """

        assert self.low is not None
        assert self.close is not None

        return (self.low - self.Price_ema(
            which_price="close", window=window)) / self.close


    def Popularity(self, window=7, closed=None):
        """
        :param window: int, default = 7.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: Popularity
        """

        for dname in ["high", "open", "close", "volume"]:
            assert self.data[dname] is not None

        result = (self.high - self.open) * self.volume
        result = result.rolling(window=window, closed=closed).sum() / \
                 (self.close - self.open).rolling(window).sum()
        return result[~np.isinf(result)]


    def Price_Accelerity(self, window=6):
        """
        :param window: int, default = 6.
        :return: Velocity of change of price.
        """

        assert self.close is not None

        return 100 * (self.close - self.close.shift(window)) / \
               self.close.shift(window)


    def MFI(self, window=14, closed=None):
        """
        Money Flow Index
        :param window: int, default = 14.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: 资金流量指数
        """

        for dname in ["close", "high", "low", "volume", "money"]:
            assert self.data[dname] is not None

        canonical_price = (self.close + self.high + self.low) / 3
        money_flow = canonical_price * self.volume
        money_flow = 2 * (((self.money - self.money.shift(1)) > 0)
                          - 0.5) * money_flow
        money_ratio = money_flow.rolling(window=window, closed=closed).apply(
            seriesPosNegSumRatio)
        return 100 - 100 / (1 + money_ratio)


    def CAR(self, window=6, market_return="hs300", closed=None):
        """
        Cumulative Abnormal Return
        :param window: int, default = 6.
        :param market_return: Default "hs300", or choose "zz500" as market return.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: window 窗口内累计反常收益。
        """

        assert self.rate is not None
        assert self.data[market_return] is not None

        if market_return == "hs300":
            abnormal_return = self.rate().apply(
                lambda col: col - self.data["hs300"]["hs300_return"], axis=0)
        else:
            abnormal_return = self.rate().apply(
                lambda col: col - self.data["zz500"]["zz500_return"], axis=0)

        return abnormal_return.rolling(window=window, closed=closed).sum()


    def DR(self, window=20, closed=None):
        """
        :param window: int, default = 20.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: mid: high for yesterday
                 increase: today's high - yesterday's mid (remove all negative values)
                 decrease: yesterday's mid - today's high (remove all negative values)
                 DR: 100 * the sum of increase values in the past window days /
                           the sum of decrease values in the past window days
        """

        assert self.high is not None
        assert self.low is not None

        mid = (self.high.shift(1) + self.low.shift(1)) / 2
        increase = dfReLU(self.high - mid.shift(1))
        decrease = dfReLU(mid.shift(1) - self.low)
        result = 100 * dfRemoveInf(increase.rolling(window=window, closed=closed).sum() /
                                   decrease.rolling(window=window, closed=closed).sum())
        return result[~np.isinf(result)]


    def VR(self, window=24, closed=None):
        """
        :param window: int, default = 24.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: AVS: The volume of the day with positive return.
                 BVS: The volume of the day with negative return.
                 CVS: The volume of the day with zero return.
                 Volume Ratio: (AVS + 1/2 * CVS) / (BVS + 1/2 * CVS)
        """

        assert self.volume is not None
        assert self.rate is not None

        AVS = self.volume[self.rate > 0].fillna(0).rolling(
            window=window, closed=closed).sum()
        BVS = self.volume[self.rate < 0].fillna(0).rolling(
            window=window, closed=closed).sum()
        CVS = self.volume[self.rate == 0].fillna(0).rolling(
            window=window, closed=closed).sum()
        return ((AVS + 1 / 2 * CVS) / (BVS + 1 / 2 * CVS)).fillna(0)


    def JumpTest(self, window=16, closed="left"):
        """
        :param window: int, default = 16.
        :param closed: str in ["left", "right", "both", "neither"], default = "left".
        :return: Non-parametric Jump Test put forwarded by Lee and Mykland.
        """

        assert self.close is not None

        logr = np.log(self.close / self.close.shift(1))
        prod_consec_abslogr = np.abs(logr) * np.abs(logr.shift(1))
        return np.sqrt(window - 2) * logr / \
               np.sqrt(prod_consec_abslogr.rolling(window, closed=closed).sum())


    def Jackknife_Weighted_Profit(self, window=10, closed=None, method="variation"):
        """
        :param window: int, default = 10.
        :param closed: str in ["left", "right", "both", "neither"], default = "None".
        :param method: default for "variation" (coefficient of variation), or "variance".
        :return: Jackknife Non-Parametric method for daily weighted profit.
        """

        for dname in ["money", "rate", "mv"]:
            assert self.data[dname] is not None

        profit = mvNeutralize(self.money * self.rate, self.mv)
        return profit.rolling(window, closed=closed).apply(
            lambda col: jackknife(col, method=method)[0])


    def Volume_std(self, window=20, closed=None):
        """
        :param window: int, default = 20.
        :param closed: str in ["left", "right", "both", "neither"], default = "None".
        :return: window-day's standard deviation of volume.
        """

        assert self.volume is not None

        return self.volume.rolling(window=window, closed=closed).std()


    def Yield_var(self, window=20, closed=None):
        """
        :param window: int, default = 20.
        :param closed: str in ["left", "right", "both", "neither"], default = "None".
        :return: window-day's standard deviation of return.
        """

        assert self.rate is not None

        return self.rate.rolling(window=window, closed=closed).var()


    def Volume_ema(self, window=10, fillna=None):
        """
        :param window: int, default = 20.
        :param fillna: If fill NaNs, default for False, otherwise input a value.
        :return: window-day's exponential moving average of volume.
        """

        assert self.volume is not None

        return EMA(self.volume, window=window, fillna=fillna)


    def EMAC(self, window=20):
        """
        window日指数移动均线
        :param window: int, default = 20.
        :return: window-day's exponential moving average of volume over today's close price.
        """

        assert self.volume is not None
        assert self.close is not None

        return self.volume.ewm(alpha=2 / (window + 1)).mean() / self.close


    def Combined_volstd_volema_turnovermean_10(self, window=10, closed=None):
        """
        :param window: int, default = 10.
        :param closed: str in ["left", "right", "both", "neither"], default = "None".
        :return: Factor that combined volume EMA, volume std and turnover mean with window length of 10.
        """

        assert self.volume is not None
        assert self.turnover

        volema = EMA(self.volume, window=window)
        volstd = self.volume.rolling(window=window, closed=closed).std()
        turnovermean = self.turnover.rolling(window=window, closed=closed).mean()

        return volema * volstd * turnovermean


    def Upper_Envelop(self, weight=0.1, window=20, closed=None):
        """
        :param weight: float in [0, 1], default = 0.1
        :param window: int, default = 20.
        :param closed: str in ["left", "right", "both", "neither"], default = "None".
        :return: 20 Period MA + (20 Period MA * 0.1)
        """

        assert self.high is not None

        return (1 + weight) * self.high.rolling(window, closed=closed).mean()


    def Lower_Envelop(self, weight=0.1, window=20, closed=None):
        """
        :param weight: float in [0, 1], default = 0.1
        :param window: int, default = 20.
        :param closed: str in ["left", "right", "both", "neither"], default = "None".
        :return: 20 Period MA - (20 Period MA * 0.1)
        """

        assert self.low is not None

        return (1 - weight) * self.low.rolling(window, closed=closed).mean()


    def RSI_quick(self, window=25, closed=None):
        """
        :param window: RSI window length, set default as 25.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: RSI values when RSI length takes 25.
        """

        assert self.rate is not None

        RS_25 = self.rate.rolling(window=window, closed=closed).apply(
                    lambda series: np.nansum(series[series > 0])) / \
                self.rate.rolling(window=window, closed=closed).apply(
                    lambda series: abs(np.nansum(series[series < 0])))

        return 100 - 100 / (1 + RS_25)


    def RSI_slow(self, window=100, closed=None):
        """
        :param window: RSI window length, set default as 100.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: RSI values when RSI length takes 100.
        """

        assert self.rate is not None

        RS_100 = self.rate.rolling(window=window, closed=closed).apply(
                    lambda series: np.nansum(series[series > 0])) / \
                 self.rate.rolling(window=window, closed=closed).apply(
                     lambda series: abs(np.nansum(series[series < 0])))

        return 100 - 100 / (1 + RS_100)


    def ATR(self, usedPrice="close", window=1, closed="left"):
        """
        :param usedPrice: Denote the type of prious-day-price used for calculation.
                Default as "close", but can also use "vwap".
        :param window: int, default = 1.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: The Average True Range. Calculated by
            "window-days-average of: max(high-low, abs(high-previous_close), abs(low-previous_close))"
        """

        if usedPrice == "close":
            assert self.close is not None
            price = self.close.shift(1)
        elif usedPrice == "vwap":
            assert self.vwap is not None
            price = self.vwap.shift(1)

        assert self.high is not None
        assert self.low is not None


        TR = np.maximum(self.high - self.low,
                        np.absolute(self.high - price),
                        np.absolute(self.low - price))

        return TR.rolling(window=window, closed=closed).mean()


    def ChandelierExit(self, usedPrice="close", multiplier=1.85, window=1, closed="left"):
        """
        :param usedPrice: Denote which type of prices is used for calculation.
                Default as "close", but can also use "vwap".
        :param multiplier: float, default = 1.85. The multiplier used to
                control ATR. Theoretically, techonological companies shoud
                have larger multipliers compared to others.
        :param window: int, default = 1.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return: Signals which denote position of a stock for the next day.
            Short if PreviousClose < window-days-highest of high - multiplier * ATR(window)
            Long if PreviousClose > window-days-highest of high + multiplier * ATR(window)
        """

        assert self.high is not None
        assert self.low is not None

        atr = multiplier * self.ATR(usedPrice=usedPrice, window=window, closed=closed)
        ExitLong = self.high.rolling(window=window, closed=closed).max() - atr
        ExitShort = self.low.rolling(window=window, closed=closed).min() + atr

        return 1 * (self.close > ExitShort) - (self.close < ExitLong)


    def OBV(self):
        """
        OBV_today = OBV_prev + volume_today         if close_today > close_prev
                  = OBV_prev + 0                    if close_today == close_prev
                  = OBV_prev - volume_today         if close_today < close_prev
        :return: On-Balance Volume （累积量能）
        """

        assert self.volume is not None
        assert self.close is not None

        comparison = self.close.fillna(0).rolling(2).apply(
            lambda arr: 1 * (arr[1] > arr[0]) - 1 * (arr[1] < arr[0]))

        obv = self.volume.copy()

        for _ in range(len(obv)-1):
            obv.iloc[_+1] = obv.iloc[_] + comparison.iloc[_+1] * obv.iloc[_+1]

        return obv


    def MAOBV(self, window=30, closed=None):
        """
        window日 移动平均 OBV
        :param window: int, default = 12.
        :param closed: str in ["left", "right", "both", "neither"], default = None.
        :return:
        """

        obv = self.OBV()

        return obv.rolling(window=window, closed=closed).mean()