from TradingStrategy151 import TradingStrategy151

class Strategy(TradingStrategy151):

    def __init__(self):
        super().__init__()


    def Markowitz(self, code_list, n=252):
        """
        :param code_list: python list object.
        :param n: int. Default to 252. The number of day of historical data that we track.
        :return: The optimal portfolio consists of the stocks that are listed in the
                 'code_list', under Markowitz's portfolio theory.
        """

        std_list = [self.rate[code][-n:].std() for code in code_list]
        mean_list = [self.rate[code][-n:].mean() for code in code_list]





# def follow_dy_similar(self, code):

