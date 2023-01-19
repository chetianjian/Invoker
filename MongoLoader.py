import pymongo
import numpy as np
import pandas as pd
from datetime import datetime


class Mongo(object):
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client.get_database("quantaxis")

        self.available_data = [
            "stock_list", "index_list", "etf_list",
            "open", "close", "high", "low", "volume", "money", "rate", "lograte",
            "vwap", "adj", "mv", "turnover", "circ_mv", "circ_turnvoer", "xdxr",
            "index_open", "index_close", "index_high", "index_low", "index_volume",
            "index_money", "index_rate", "index_vwap",
            "reports", "stock_block", "stock_basic", "daily_basic", "stock_info",
            "namechange",
            "hs300", "zz500", "sh50",
            "close_min",
        ]

        self.data = dict.fromkeys(self.available_data)


    @property
    def list_database(self) -> list:
        # List all databases' names under the given client.
        return self.client.list_database_names()


    @property
    def list_schema(self) -> list:
        # List all schemas' names inside the given database.
        return self.db.list_collection_names()


    @property
    def calendar(self) -> list:
        # List the historical record of trading days in the database.
        return self.data["close"].index


    @property
    def list_stock_code(self) -> list:
        # Return the code list of all stocks in the database.
        # * Note:
        #       self.stock_list including the codes of stocks that are not listed yet.
        #       self.list_stock_code only contains the codes of currently listed stocks.
        return self.data["close"].columns


    # -------------------------------------------------------------------------------------------
    # Quick calls for stock list, index list and etf list respectively.


    @property
    def stock_list(self):
        # * Note:
        #       self.stock_list including the codes of stocks that are not listed yet.
        #       self.list_stock_code only contains the codes of currently listed stocks.
        if self.data["stock_list"] is None:
            self.data["stock_list"] = self.load_stock_list()
        return self.data["stock_list"]


    @property
    def index_list(self):
        if self.data["index_list"] is None:
            self.data["index_list"] = self.load_index_list()
        return self.data["index_list"]


    @property
    def etf_list(self):
        if self.data["etf_list"] is None:
            self.data["etf_list"] = self.load_etf_list()
        return self.data["etf_list"]


    # -------------------------------------------------------------------------------------------
    # Properties for quick calling of daily stock data.


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
    def money(self) -> pd.DataFrame:
        return self.data["money"]


    @property
    def adj(self) -> pd.DataFrame:
        if self.data["adj"] is None:
            self.load_adj_day()
        return self.data["adj"]


    @property
    def dummy(self):
        """
        :return: Return -1.0 if today's yield < 0 otherwise 1.0
        """

        return 2 * ((self.rate >= 0) - 0.5)


    @property
    def lograte(self) -> pd.DataFrame:
        if self.data["lograte"] is None:
            self.data["lograte"] = np.log(self.rate + 1)
        return self.data["lograte"]


    @property
    def mv(self) -> pd.DataFrame:
        if self.data["mv"] is None:
            self.load_total_mv()
        return self.data["mv"]


    def circ_mv(self) -> pd.DataFrame:
        if self.data["circ_mv"] is None:
            self.load_circ_mv()
        return self.data["circ_mv"]


    @property
    def turnover(self) -> pd.DataFrame:
        if self.data["turnover"] is None:
            self.load_turnover()
        return self.data["turnover"]


    @property
    def xdxr(self) -> pd.DataFrame:
        return self.data["xdxr"]


    @property
    def block(self) -> pd.DataFrame:
        if self.data["stock_block"] is None:
            self.load_stock_block()
        return self.data["stock_block"]


    @property
    def stock_basic(self) -> pd.DataFrame:
        if self.data["stock_basic"] is None:
            self.load_stock_basic()
        return self.data["stock_basic"]


    @property
    def info(self) -> pd.DataFrame:
        if self.data["stock_info"] is None:
            self.load_stock_info()
        return self.data["stock_info"]


    @property
    def namechange(self) -> pd.DataFrame:
        return self.data["namechange"]


    # -------------------------------------------------------------------------------------------
    # Properties for quick calling of index daily data.
    # Format: initiate with "i" for representing they are index data.


    @property
    def iopen(self) -> pd.DataFrame:
        return self.data["index_open"]

    @property
    def iclose(self) -> pd.DataFrame:
        return self.data["index_close"]


    @property
    def ihigh(self) -> pd.DataFrame:
        return self.data["index_high"]


    @property
    def ilow(self) -> pd.DataFrame:
        return self.data["index_low"]


    @property
    def ivolume(self) -> pd.DataFrame:
        return self.data["index_volume"]


    @property
    def imoney(self) -> pd.DataFrame:
        return self.data["index_money"]


    @property
    def irate(self) -> pd.DataFrame:
        return self.data["index_rate"]


    @property
    def ivwap(self) -> pd.DataFrame:
        return self.data["index_vwap"]


    # -------------------------------------------------------------------------------------------
    # Properties for quick calling of several special
    # Format: initiate with "i" for representing they are index data.


    @property
    def hs300(self):
        # 沪深 300
        if self.data["hs300"] is None:
            self.data["hs300"] = self.single_index_day(code="000300")
        return self.data["hs300"]


    @property
    def zz500(self):
        # 中证 500
        if self.data["zz500"] is None:
            self.data["zz500"] = self.single_index_day(code="000905")
        return self.data["zz500"]


    @property
    def sh50(self):
        # 上证 50
        if self.data["sh50"] is None:
            self.data["sh50"] = self.single_index_day(code="000016")
        return self.data["sh50"]


    # -------------------------------------------------------------------------------------------
    # For single query.


    @classmethod
    def single_stock_day(cls, code: str) -> pd.DataFrame:
        """
        :param code: Target stock code.
        :return: A dataframe that containing daily data for the given stock code.
                    * Note that the data of a single underlying would not be stored
                        in the Invoker.data dictionary.
        """

        return pd.DataFrame(cls().db["stock_day"].find({"code": code})).drop(
            columns=["_id", "date_stamp"]
        ).rename(
            columns={"amount": "money", "vol": "volume"}
        ).set_index("date")


    @classmethod
    def single_index_day(cls, code: str) -> pd.DataFrame:
        """
        :param code: Target index code (including ETFs).
        :return: A dataframe that containing daily data for the given index or etf code.
                    * Note that the data of a single underlying would not be stored
                        in the Invoker.data dictionary.
        """

        return pd.DataFrame(cls().db["index_day"].find({"code": code})).drop(
            columns=["_id", "date_stamp"]
        ).rename(
            columns={"amount": "money", "vol": "volume"}
        ).set_index("date")


    # -------------------------------------------------------------------------------------------
    # Some basic functions.


    @property
    def cache(self) -> list:
        return [dname for dname in self.available_data if self.data[dname] is not None]


    def release_memory(self, dname=None, clear_all=False):
        """
        :param dname: Default as None. Should be a str or list[str] or other iterable object
                        which contains the undesired data-names.
        :param clear_all: Default as False. Set clear_all=True if you want to clear all loaded data.
        """

        if clear_all:
            self.data = dict.fromkeys(self.available_data)
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

        for dname in self.available_data:
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
                        (pd.to_datetime(start_date) <=
                         pd.to_datetime(self.data[dname].reset_index("code").index))
                        &
                        (pd.to_datetime(self.data[dname].reset_index("code").index) <=
                         pd.to_datetime(end_date))].reset_index().set_index(self.data[dname].index.names)

                    print(f"Successfully truncated for {dname} data.")
            else:
                continue


    def slim(self, maxNanProp=0.1):
        """
        Remove all the columns of stock data which have more NaNs than we expected.
        :param maxNanProp: The maximum proportion of NaN values we can endure.
        """

        pass
        threshold = len(self.calendar) * maxNanProp

        for dname in self.available_data:
            count = 0
            if self.data[dname] is None:
                continue
            elif self.data[dname].shape == (len(self.calendar), len(self.list_stock_code)):
                for col in self.list_stock_code[::-1]:
                    if np.nansum(np.isnan(self.data[dname][col])) > threshold:
                        count += 1
                        del self.data[dname][col]
                print(f"Succesfully removed {count} columns from stock {dname} data.")


    # -------------------------------------------------------------------------------------------
    # Data loading functions.
    # Usage:
    #       1. inv = Invoker()
    #          Invoker.load_func(inv)
    # Or:
    #       2. inv.load_func()


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


    def load_total_mv(self):
        """
        :return: Load total market capitalization into self.data dictionary.
        """

        df = pd.DataFrame(self.db["daily_basic"].find(
            {}, {"_id": 0, "code": 1, "trade_date": 1, "total_mv": 1}))
        df = df.set_index(["trade_date", "code"]).unstack().droplevel(0, axis=1)
        df.index = map(lambda d: d[0:4] + "-" + d[4:6] + "-" + d[6:], df.index)
        df.index.name = "date"

        self.data["mv"] = df

        print("Daily market total capitalization loaded successfully.")
        return df


    def load_circ_mv(self):
        """
        :return: Load information of circulating market capitalization into self.data dictionary.
        """

        df = pd.DataFrame(self.db["daily_basic"].find(
            {}, {"_id": 0, "code": 1, "trade_date": 1, "circ_mv": 1}))
        df = df.set_index(["trade_date", "code"]).unstack().droplevel(0, axis=1)
        df.index = map(lambda d: d[0:4] + "-" + d[4:6] + "-" + d[6:], df.index)
        df.index.name = "date"

        self.data["circ_mv"] = df

        print("Stock circulating market capitalization loaded successfully.")
        return df


    def load_turnover(self):
        """
        :return: Load turnover ratio into self.data dictionary.
        """

        df = pd.DataFrame(self.db["daily_basic"].find(
            {}, {"_id": 0, "code": 1, "trade_date": 1, "turnover_rate": 1}))
        df = df.set_index(["trade_date", "code"]).unstack().droplevel(0, axis=1)
        df.index = map(lambda d: d[0:4] + "-" + d[4:6] + "-" + d[6:], df.index)
        df.index.name = "date"

        self.data["turnover"] = df

        print("Stock turnover ratio loaded successfully.")
        return df


    def load_circ_turnover(self):
        """
        :return: Load turnover ratio of circulating capitalization into self.data dictionary.
        """

        df = pd.DataFrame(self.db["daily_basic"].find(
            {}, {"_id": 0, "code": 1, "trade_date": 1, "turnover_rate_f": 1}))
        df = df.set_index(["trade_date", "code"]).unstack().droplevel(0, axis=1)
        df.index = map(lambda d: d[0:4] + "-" + d[4:6] + "-" + d[6:], df.index)
        df.index.name = "date"

        self.data["circ_turnover"] = df

        print("Stock circulating turnover ratio loaded successfully.")
        return df


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
            print(f"Daily index {table_name} loaded successfully.")

        self.data["index_rate"] = self.data["close"].diff(1) / self.data["close"].shift(1)

        del df

        self.data["index_vwap"] = ((self.data["high"] + self.data["low"] + self.data["close"]) / 3)
        print("Daily index vwap loaded successfully.")

        self.data["index_rate"] = self.data["close"].diff(1) / self.data["close"].shift(1)
        print("Daily index return loaded successfully.")


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

        print("Index list loaded successfully.")
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


    def load_stock_basic(self):
        """
        :return: Load stock basic data, including information of listed date, name,
                    area and industry.
        """

        df = pd.DataFrame(self.db["stock_basic"].find()).drop(columns=["_id"]).rename(
            columns={"list_date_stamp": "time_stamp"}).set_index("code")

        df["list_date"] = list(map(lambda x: x[0:4] + "-" + x[4:6] + "-" + x[6:], df["list_date"]))
        self.data["stock_basic"] = df
        print("Stock basic data loaded successfully.")
        return df


    def load_stock_info(self):
        """
        :return: Load stock information, including the following items:
                流通股, 总股本, 国家股, 发起人法人股, 法人股, B股, H股, 职工股, 总资产, 流动资产, 固定资产,
                固定资产, 无形资产, 股东人数, 流动负债, 长期负债, 资本公积金, 净资产, 主营收入, 主营利润,
                应收账款, 营业利润, 投资收入, 经营现金流, 总现金流, 存货, 利润总和, 税后利润, 净利润,
                未分配利润, 每股净资产
        """

        df = pd.DataFrame(self.db["stock_info"].find()).drop(columns=["_id"])

        df = df.rename(columns={
            "liutongguben": "流通股", "zongguben": "总股本", "guojiagu": "国家股",
            "faqirenfarengu": "发起人法人股", "farengu": "法人股", "bgu": "B股",
            "hgu": "H股", "zhigonggu": "职工股", "zongzichan": "总资产",  "liudongzichan": "流动资产",
            "gudingzichan": "固定资产", "wuxingzichan": "无形资产", "gudongrenshu": "股东人数",
            "liudongfuzhai": "流动负债", "changqifuzhai": "长期负债", "zibengongjijin": "资本公积金",
            "jingzichan": "净资产", "zhuyingshouru": "主营收入", "zhuyinglirun": "主营利润",
            "yingshouzhangkuan": "应收账款", "yingyelirun": "营业利润", "touzishouyu": "投资收入",
            "jingyingxianjinliu": "经营现金流", "zongxianjinliu": "总现金流", "cunhuo": "存货",
            "lirunzonghe": "利润总和", "shuihoulirun": "税后利润", "jinglirun": "净利润",
            "weifenpeilirun": "未分配利润", "meigujingzichan": "每股净资产"
        })

        df["market"] = df["market"].apply(lambda x: "深圳" if x == 0 else "上海")

        self.data["stock_info"] = df
        print("Stock information loaded successfully.")
        return df


    def load_namechange(self):
        """
        :return: Load the historical name of stocks and the reason why the name was changed.
        """

        self.data["namechange"] = pd.DataFrame(self.db["namechange"].find()).drop(
            columns=["_id"]).set_index("code")
        print("Name change information loaded successfully.")
        return self.data["namechange"]

