from typing import List

from Factor import Factor
from Strategy import Strategy
from Alpha101 import Alpha101
from utils import *


class Invoker(Factor, Strategy, Alpha101):

    def __init__(self):
        # print("I am a beacon of knowledge blazing out across a black sea of ignorance.")
        super().__init__()


    @classmethod
    def code2df(cls, code: str) -> pd.DataFrame:
        """
        :param code: A single stock code.
        :return: A dataframe with all daily data of the given stock.
        """
        return pd.DataFrame(cls().db["stock_day"].find({"code": code})).drop(
            columns=["_id", "date_stamp"]
        ).rename(
            columns={"amount": "money", "vol": "volume"}
        ).set_index("date")


    @classmethod
    def code2name(cls, code: str) -> str:
        """
        :param code: A single stock code.
        :return: The corresponding name of the given stock.
        """
        result = pd.DataFrame(cls().db["stock_list"].find({"code": code}))
        return result["name"][0] if 0 not in result.shape else None


    @classmethod
    def name2code(cls, name: str) -> str:
        """
        :param name: A single stock name.
        :return: The corresponding code of the given stock.
        """
        result = pd.DataFrame(cls().db["stock_list"].find({"name": name}))
        return result["code"][0] if 0 not in result.shape else None


    @classmethod
    def code2block(cls, code: str) -> List[str]:
        """
        :param code: A single stock code.
        :return: The block names which the stock belongs to.
        """
        result = pd.DataFrame(cls().db["stock_block"].find({"code": code}))
        return list(set(result["blockname"])) if 0 not in result.shape else None


    @classmethod
    def block2code(cls, name: str) -> List[str]:
        """
        :param name: The name of the queried block, such as "地下管网".
        :return: A list of matching stock codes.
        """
        result = pd.DataFrame(cls().db["stock_block"].find({"blockname": name}))
        return list(set(result["code"])) if 0 not in result.shape else None


    def yield_candle(self, code_list, n=0):
        """
        Usage:
                iteration = Invoker.yield_candle(code_list)
                next(iteration)
        :param code_list: An iterable that stores the target stock codes.
        :param n: Default to 0, which means all data are required to be included.
                  Draw the candle plot for the last-n-days.
        :return: yield one candle graph each time you call the function.
        """
        for code in code_list:
            # noinspection PyBroadException
            try:
                print(f"{code}: {self.data['stock_list'].loc[code, 'name']}")
                yield draw_candle(self.code2df(code).iloc[-n:, :])
            except:
                print("All candle graphs have been displayed.")



    def nrate(self, n, method="simple"):
        """
        :param n: int, the period size.
        :param method: Default to "simple", alternative to "compound".
        :return: pd.DataFrame, rolled n-period rate of returns.
        """

        if method == "simple":
            assert self.rate is not None
            return self.rate.rolling(n).sum()

        elif method == "compound":
            assert self.close is not None
            return self.close.diff(n) / self.close.shift(n)

        else:
            raise NameError("Method should take value as 'simple' or 'compound'.")


