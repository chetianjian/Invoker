import pymongo
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
    def list_database(self):
        return self.client.list_database_names()


    @property
    def list_schema(self):
        return self.db.collection_names()


    @property
    def open(self):
        assert self.data["open"] is not None
        return self.data["open"]


    @property
    def close(self):
        assert self.data["close"] is not None
        return self.data["close"]


    @property
    def high(self):
        assert self.data["high"] is not None
        return self.data["high"]


    @property
    def low(self):
        assert self.data["low"] is not None
        return self.data["low"]


    @property
    def vwap(self):
        assert self.data["vwap"] is not None
        return self.data["vwap"]


    @property
    def rate(self):
        assert self.data["rate"] is not None
        return self.data["rate"]


    @property
    def volume(self):
        assert self.data["volume"] is not None
        return self.data["volume"]


    @property
    def turnover(self):
        assert self.data["turnover"] is not None
        return self.data["turnover"]


    @property
    def money(self):
        assert self.data["money"] is not None
        return self.data["money"]


    @property
    def adj(self):
        assert self.data["adj"] is not None
        return self.data["adj"]


    @property
    def mv(self):
        assert self.data["mv"] is not None
        return self.data["mv"]


    @property
    def xdxr(self):
        assert self.data["xdxr"] is not None
        return self.data["xdxr"]


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


    def load_stock_daily(self):
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


    def load_index_daily(self):
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


    def load_open_daily(self):
        """
        :return: Load daily open prices into self.data dictionary.
        """

        self.data["open"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["open"].unstack()

        print("Daily open prices loaded successfully.")


    def load_close_daily(self):
        """
        :return: Load daily close prices into self.data dictionary.
        """

        self.data["close"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["close"].unstack()

        print("Daily close prices loaded successfully.")


    def load_low_daily(self):
        """
        :return: Load daily low prices into self.data dictionary.
        """

        self.data["low"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["low"].unstack()

        print("Daily low prices loaded successfully.")


    def load_high_daily(self):
        """
        :return: Load daily high prices into self.data dictionary.
        """

        self.data["high"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["high"].unstack()

        print("Daily high prices loaded successfully.")


    def load_volume_daily(self):
        """
        :return: Load daily volume into self.data dictionary.
        """

        self.data["volume"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["vol"].unstack()

        print("Daily volume size loaded successfully.")


    def load_money_daily(self):
        """
        :return: Load daily trade amount into self.data dictionary.
        """

        self.data["moeny"] = pd.DataFrame(self.db["stock_day"].find()).set_index(
            ["date", "code"])["amount"].unstack()

        print("Daily money amount loaded successfully.")


    def load_rate_daily(self):
        """
        :return: Load daily stock returns into self.data dictionary.
        """

        self.load_close_daily()
        self.data["rate"] = self.data["close"].diff(1) / self.data["close"].shift(1)
        self.release_memory(dname="close")

        print("Daily stock returns loaded successfully.")


    def load_adj_daily(self):
        """
        :return: Load daily stock adj prices into self.data dictionary (Large Dataset!).
        """

        self.data["adj"] = pd.DataFrame(self.db["stock_adj"].find()).drop(
            columns=["_id"]).set_index(["date", "code"])["adj"].unstack()

        print("Daily stock adj prices loaded successfully.")


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


    def load_index_list(self):
        """
        :return: Load list of indices into self.data dictionary.
        """

        self.data["index_list"] = pd.DataFrame(self.db["index_list"].find()).drop(
            columns=["_id", "volunit", "decimal_point", "sec"]).set_index("code")

        print("Index list data loaded successfully.")


    def load_stock_block(self):
        """
        :return: Load stock block data (blocknames, types & code) to self.data dictionary.
        """

        self.data["stock_block"] = pd.DataFrame(self.db["stock_block"].find()).drop(
            columns=["_id", "source"]).set_index("code")

        print("Stock block data loaded successfully.")


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