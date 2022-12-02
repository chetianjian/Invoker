from MongoLoader import Mongo
import numpy as np


class TradingStrategy151(Mongo):
    def __init__(self):
        super().__init__()

    def Strategy_1_Price_Momemtum(self, H, Q=0.1, S=1, T=12):
        """
        :param H: Holding Period.
        :param Q: Choose top quantile = Q stocks., default to 0.1.
        :param S: Skip Period, default to 1.
        :param T: Formation Perid, default to 12.
        :return: Strategy Yields.
        """

        numStock = int(len(self.list_stock_code) * Q)
        Rcum = self.close.shift(S) / self.close.shift(S+T) - 1
        Rmean = self.rate.shift(S).rolling(T).sum() / T
        df = np.square((self.rate.shift(S)-Rmean))
        sigma = np.sqrt(df.rolling(T).sum() / (T-1))
        R_risk_adjusted = Rmean / sigma

        backtest_yield = 0


        for r in range(len(Rcum)-H-1):
            high_Rcum = Rcum.iloc[r, :].sort_values(ascending=False)[:20].index
            high_Rmean = Rmean.iloc[r, :].sort_values(ascending=False)[:20].index
            high_R_risk_adjusted = R_risk_adjusted.iloc[r, :].sort_values(ascending=False)[:20].index

            finalist = list(set(high_Rcum) & set(high_Rmean) & set(high_R_risk_adjusted))

            if finalist is None:
                continue
            else:
                finalist = self.close[finalist]
                backtest_yield += ((finalist.iloc[r+H+1] - finalist.iloc[r+1]) / finalist.iloc[r+1]).sum()

        return backtest_yield


    def Strategy_4_Low_Volatility_anomaly(self, H, T=22):
        """
        :param H: Holding Period.
        :param T: Formation Perid, default to 22.
        :return:
        """


    def Strategy_9_Mean_Reversion_Single_Cluster(self, n):
        pass







