import pandas as pd
import tushare as ts
import time
from datetime import datetime
import func_timeout
from func_timeout import func_set_timeout
import os


class ShareHolderFetcher(object):
    def __init__(self, start_date=None, end_date=None):
        """
        :param start_date: str. Format: 'YYYYMMDD'
        :param end_date: str. Format: 'YYYYMMDD'
        """

        if start_date is None:
            self.start = "".join(str(pd.to_datetime(datetime.now()) -
                                     pd.Timedelta(24, "m")).split(" ")[0].split("-"))
        else:
            self.start = start_date

        if end_date is None:
            self.end = "".join(str(datetime.now()).split(" ")[0].split("-"))
        else:
            self.end = end_date


        self.engine = ts.pro_api()
        self.root = os.getcwd().replace("\\", "/")
        self.info = self.engine.query(
            "stock_basic", exchange="", list_status="L",
            fields="ts_code, symbol, name, area, industry, list_date")

        self.codes = self.info["ts_code"]
        self.tmp = self.fetch_code(code=self.codes[0])
        self.tmp.to_csv(self.root + "/shareholder.csv", encoding="utf_8_sig")
        self.count = 1
        self.total = len(self.codes)
        print(f"Initializing crawling progress: {self.count}/{self.total}")



    @func_set_timeout(70)
    def fetch_code(self, code: str):
        """
        code: string, should be the format as "dddddd.XX", such as "000001.SZ".
        """
        try:
            return self.engine.query("top10_holders", ts_code=code,
                                     start_date=self.start, end_date=self.end)
        except func_timeout.exceptions.FunctionTimedOut:
            print(f"抓取数据超时，股票代码：{code}")





    def fetcher(self):

        for code in self.codes[self.count:]:
            if self.count % 10 == 0:
                self.tmp.to_csv(self.root + "/shareholder.csv", encoding="utf_8_sig")
                msg = f"""
                The last stock we get: {code}, \n
                Wait for 65 seconds...
                """
                print(msg)
                time.sleep(65)

            self.count += 1

            print(f"Now saving {self.count}/{self.total}...")

            try:
                tmp = self.fetch_code(code)
            except:
                self.count -= 1
                self.tmp.to_csv(self.root + "/shareholder.csv", encoding="utf_8_sig")
                break
            self.tmp = pd.concat([self.tmp, tmp])

        else:
            return "All data collected."

        print("Retrying...")




if __name__ == "__main__":
    holder = ShareHolderFetcher()
    while holder.count != holder.total:
        holder.tmp = pd.read_csv(holder.root + "/shareholder.csv")
        del holder.tmp["Unnamed: 0"]
        holder.fetcher()
        assert holder.count == holder.total
