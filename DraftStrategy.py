from Factor import Factor

class DraftStrategy(Factor):

    def __init__(self):
        super().__init__()


    def ATR_band(self, n):
        """
        :param n: ATR 乘数
        :return:
        """

        upper = self.open + self.atr
        lower = self.open - self.atr







