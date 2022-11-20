from Factor import Factor
from Strategy import Strategy
from utils import *


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