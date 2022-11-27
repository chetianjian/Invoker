from Factor import Factor
from Strategy import Strategy
from utils import *


# noinspection PyBroadException
class Invoker(Factor, Strategy):
    def __init__(self):
        # print("I am a beacon of knowledge blazing out across a black sea of ignorance.")
        super().__init__()


    @classmethod
    def code2df(cls, code: str):
        return pd.DataFrame(cls().db["stock_day"].find({"code": code})).drop(
            columns=["_id", "date_stamp"]
        ).rename(
            columns={"amount": "money", "vol": "volume"}
        ).set_index("date")


    def yield_candle(self, code_list, n=0):
        """
        Usage:
                iteration = Invoker.yield_candle(code_list)
                next(iteration)
        :param code_list: An iterable that stores the target stock codes.
        :param n: Default to 0. Draw the candle plot for the last-n-days.
        :return: yield one candle graph each time you call the function.
        """
        for code in code_list:
            try:
                print(f"{code}: {self.data['stock_list'].loc[code, 'name']}")
                yield draw_candle(self.code2df(code))
            except:
                print("All candle graphs have been displayed.")