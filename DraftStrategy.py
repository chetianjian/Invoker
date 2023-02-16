from Factor import Factor
from utils import *


class DraftStrategy(Factor):

    def __init__(self):
        super().__init__()


    def ATR_band(self, code, n):
        """
        :param code: str. The code that is queried. Format: XXXXXX
        :param n: ATR 乘数
        :return: The band graph and calculated data correspondingly.
        """

        upper = self.open[code] + self.atr[code]
        lower = self.open[code] - self.atr[code]
        draw_lines(series_list=[upper, lower], legend_list=["upper", "lower"])

        return pd.concat([upper, lower], axis=1, keys=["upper", "lower"])











