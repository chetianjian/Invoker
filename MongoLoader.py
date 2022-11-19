import pymongo
import numpy as np
import pandas as pd
from datetime import datetime


class Mongo(object):
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client.get_database("quantaxis")

        self.available_dname = ["open", "close", "high", "low", "volume", "money",
                                "rate", "vwap", "mv", "turnover", "adj", "xdxr",
                                "stock_day", "index_day", "hs300", "zz500", "reports",
                                "stock_block", "stock_list", "etf_list", "index_list",
                                "close_min"]

        self.data = dict.fromkeys(self.available_dname)


    @property
    def list_database(self) -> list:
        # List all databases' names under the given client.
        return self.client.list_database_names()


    @property
    def list_schema(self) -> list:
        # List all schemas' names inside the given database.
        return self.db.collection_names()


    @property
    def list_trading_day(self) -> list:
        # List the historical record of trading days in the database.
        return self.data["close"].index


    @property
    def list_stock_code(self) -> list:
        # Return the code list of all stocks in the database.
        return self.data["close"].columns


    @property
    def open(self) -> pd.DataFrame:
        return self.data["open"]


    @property
    def close(self) -> pd.DataFrame:
        return self.data["close"]


    @property
    def high(self) -> pd.DataFrame:
        return self.data["high"]


    @property
    def low(self) -> pd.DataFrame:
        return self.data["low"]


    @property
    def vwap(self) -> pd.DataFrame:
        return self.data["vwap"]


    @property
    def rate(self) -> pd.DataFrame:
        return self.data["rate"]


    @property
    def volume(self) -> pd.DataFrame:
        return self.data["volume"]


    @property
    def turnover(self) -> pd.DataFrame:

        return self.data["turnover"]


    @property
    def money(self) -> pd.DataFrame:
        return self.data["money"]


    @property
    def adj(self) -> pd.DataFrame:
        return self.data["adj"]


    @property
    def mv(self) -> pd.DataFrame:
        return self.data["mv"]


    @property
    def xdxr(self) -> pd.DataFrame:
        return self.data["xdxr"]


    @classmethod
    def single_stock_day(cls, code: str) -> pd.DataFrame:
        return pd.DataFrame(cls().db["stock_day"].find({"code": code})).drop(
            columns=["_id", "date_stamp"]
        ).rename(
            columns={"amount": "money", "vol": "volume"}
        ).set_index("date")


    def release_memory(self, dname=None, clear_all=False):
        """
        :param dname: Default as None. Should be a str or list[str] or other iterable object
                        which contains the undesired data-names.
        :param clear_all: Default as False. Set clear_all=True if you want to clear all loaded data.
        """

        if clear_all:
            self.data = dict.fromkeys(self.available_dname)
            print("Successfully cleared all the data.")

        elif type(dname) == str:
            self.data[dname] = None
            print(f"Successfully cleared data in {dname}.")

        else:
            for _ in dname:
                self.data[_] = None
                print(f"Successfully cleared data in {_}.")


    def truncate(self, start_date="2017-01-01", end_date=pd.to_datetime(datetime.now())):
        """
        :param start_date: Default as None.Expected for a date in str format such as "1974-06-23".
        :param end_date: Default as None. Expected for a date in str format such as "1974-06-23".
        """

        for dname in self.available_dname:
            if self.data[dname] is None:
                continue
            elif self.data[dname].index.names[0] == "date":
                if len(self.data[dname].index.names) == 1:

                    self.data[dname] = self.data[dname][
                        (pd.to_datetime(start_date) <= pd.to_datetime(self.data[dname].index))
                        &
                        (pd.to_datetime(self.data[dname].index) <= pd.to_datetime(end_date))
                        ]

                    print(f"Successfully truncated for {dname} data.")

                else:
                    self.data[dname] = self.data[dname].reset_index("code")[
                        (pd.to_datetime(start_date) <= pd.to_datetime(self.data[dname].reset_index("code").index))
                        &
                        (pd.to_datetime(self.data[dname].reset_index("code").index) <= pd.to_datetime(end_date))
                        ].reset_index().set_index(self.data[dname].index.names)

                    print(f"Successfully truncated for {dname} data.")

            else:
                continue


    def slim(self, maxNanProp=0.1):
        """
        Remove all the columns of stock data which have more NaNs than we expected.
        :param maxNanProp: The maximum proportion of NaN values we can endure.
        """

        pass
        threshold = len(self.list_trading_day) * maxNanProp

        for dname in self.available_dname:
            count = 0
            if self.data[dname] is None:
                continue
            elif self.data[dname].shape == (len(self.list_trading_day), len(self.list_stock_code)):
                for col in self.list_stock_code[::-1]:
                    if np.nansum(np.isnan(self.data[dname][col])) > threshold:
                        count += 1
                        del self.data[dname][col]
                print(f"Succesfully removed {count} columns from stock {dname} data.")


    def load_stock_day(self):
        """
        :return: Load daily stock data into self.data dictionary.
        """

        df = pd.DataFrame(self.db["stock_day"].find()).drop(
            columns=["_id", "date_stamp"]
        ).rename(
            columns={"amount": "money", "vol": "volume"}
        ).set_index(["date", "code"])

        for table_name in ["open", "close", "low", "high", "volume", "money"]:
            self.data[table_name] = df[table_name].unstack()
            print(f"Daily {table_name} loaded successfully.")

        del df

        self.data["vwap"] = ((self.data["high"] + self.data["low"] + self.data["close"]) / 3)
        print("Daily vwap loaded successfully.")

        self.data["rate"] = self.data["close"].diff(1) / self.data["close"].shift(1)
        print("Daily rate loaded successfully.")


    def load_index_day(self):
        """
        :return: Load daily index data into self.data dictionary.
        """
        df = pd.DataFrame(self.db["index_day"].find()).drop(
            columns=["_id", "date_stamp"]
        ).rename(
            columns={"amount": "money", "vol": "volume"}
        ).set_index(["date", "code"])

        for table_name in ["open", "close", "low", "high", "volume", "money"]:
            self.data["index_" + table_name] = df[table_name].unstack()
            print(f"Daily index {table_name} data loaded successfully.")

        del df

        self.data["index_vwap"] = ((self.data["high"] + self.data["low"] + self.data["close"]) / 3)
        print("Daily index vwap data loaded successfully.")

        self.data["index_rate"] = self.data["close"].diff(1) / self.data["close"].shift(1)
        print("Daily index return data loaded successfully.")


    def load_open_day(self):
        """
        :return: Load daily open prices into self.data dictionary.
        """

        self.data["open"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["open"].unstack()

        print("Daily open prices loaded successfully.")
        return self.data["open"]


    def load_close_day(self):
        """
        :return: Load daily close prices into self.data dictionary.
        """

        self.data["close"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["close"].unstack()

        print("Daily close prices loaded successfully.")
        return self.data["close"]


    def load_low_day(self):
        """
        :return: Load daily low prices into self.data dictionary.
        """

        self.data["low"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["low"].unstack()

        print("Daily low prices loaded successfully.")
        return self.data["low"]


    def load_high_day(self):
        """
        :return: Load daily high prices into self.data dictionary.
        """

        self.data["high"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["high"].unstack()

        print("Daily high prices loaded successfully.")
        return self.data["high"]


    def load_volume_day(self):
        """
        :return: Load daily volume into self.data dictionary.
        """

        self.data["volume"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["vol"].unstack()

        print("Daily volume size loaded successfully.")
        return self.data["volume"]


    def load_money_day(self):
        """
        :return: Load daily trade amount into self.data dictionary.
        """

        self.data["money"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["amount"].unstack()

        print("Daily money amount loaded successfully.")
        return self.data["money"]


    def load_rate_day(self):
        """
        :return: Load daily stock returns into self.data dictionary.
        """

        self.load_close_day()
        self.data["rate"] = self.data["close"].diff(1) / self.data["close"].shift(1)
        self.release_memory(dname="close")

        print("Daily stock returns loaded successfully.")
        return self.data["rate"]


    def load_adj_day(self):
        """
        :return: Load daily stock adj prices into self.data dictionary (Large Dataset!).
        """

        self.data["adj"] = pd.DataFrame(self.db["stock_adj"].find()).drop(
            columns=["_id"]).set_index(["date", "code"])["adj"].unstack()

        print("Daily stock adj prices loaded successfully.")
        return self.data["adj"]


    def load_stock_list(self):
        """
        :return: Load stock list into self.data dictionary.
        """

        self.data["stock_list"] = pd.DataFrame(self.db["stock_list"].find()).drop(
            columns=["_id", "volunit", "decimal_point", "sec"]).set_index("code")

        print("Stock list data loaded successfully.")
        return self.data["stock_list"]


    def load_etf_list(self):
        """
        :return: Load ETF list into self.data dictionary.
        """

        self.data["etf_list"] = pd.DataFrame(self.db["etf_list"].find()).drop(
            columns=["_id", "volunit", "decimal_point", "sec"]).set_index("code")

        print("ETF list data loaded successfully.")
        return self.data["etf_list"]


    def load_index_list(self):
        """
        :return: Load list of indices into self.data dictionary.
        """

        self.data["index_list"] = pd.DataFrame(self.db["index_list"].find()).drop(
            columns=["_id", "volunit", "decimal_point", "sec"]).set_index("code")

        print("Index list data loaded successfully.")
        return self.data["index_list"]


    def load_stock_block(self):
        """
        :return: Load stock block data (blocknames, types & code) to self.data dictionary.
        """

        self.data["stock_block"] = pd.DataFrame(self.db["stock_block"].find()).drop(
            columns=["_id", "source"]).set_index("code")

        print("Stock block data loaded successfully.")
        return self.data["stock_block"]


    def load_xdxr(self):
        """
        :return: Load data of stocks' XDXR (EX-Dividend and EX-Right) into self.data dictionary.
        """

        self.data["xdxr"] = pd.DataFrame(self.db["stock_xdxr"].find()).drop(
            columns=["_id", "name", "category", "suogu", "fenshu"]).rename(
            columns={"fenhong": "分红", "peigu": "配股", "peigujia": "配股价",
                     "songzhuangu": "送转股", "xingquanjia": "行权价",
                     "category_meaning": "category"}).set_index(["date", "code"])

        print("Stock XDXR data loaded successfully.")
        return self.data["xdxr"]


