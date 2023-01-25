import numpy as np
import pandas as pd
from Mongo import Mongo
import os


class ShareHolderRank(Mongo):
    def __init__(self):
        super().__init__()
        self.root = os.getcwd().replace("\\", "/")


    def ShareHolderCumSum(self):
        df = pd.read_csv("./shareholders.csv", encoding="utf-8-sig")
        del df["Unnamed: 0"]
        df["ts_code"] = list(map(lambda x: x[: 6], df["ts_code"]))
        df = df.rename(columns={"ts_code": "code"}).set_index("holder_name")

        available_codes = list(set(self.close.columns))
        calendar = list(self.close.index)
        holder_list = list(set(df.index))

        result = {}

        for holder in holder_list:

            holder_df = df.loc[[holder]].sort_values("end_date")
            codes = list(set(holder_df["code"]))
            result[holder] = 0

            for code in codes:
                if code not in available_codes:
                    continue

                sep_df = holder_df[holder_df["code"] == code]
                start_date = str(sep_df["end_date"][0])
                start_date = start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:]

                if start_date not in calendar:
                    continue
                if len(sep_df) == 1:
                    end_date = self.close.index[-1]
                else:
                    end_date = str(sep_df.iloc[-1, :]["end_date"])
                    end_date = end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:]
                    if end_date not in calendar:
                        continue

                sep_return = (self.close[code][end_date] - self.close[code][start_date]) / \
                             self.close[code][start_date]

                if np.isnan(sep_return):
                    continue
                result[holder] += sep_return

        result = pd.DataFrame(pd.Series(result))
        result = result.rename(columns={0: "simple return"})
        result = result.sort_values("simple return", ascending=False).dropna()
        result.to_csv(self.root + "/holder_cumsum.csv", encoding="utf_8_sig")

        return result
