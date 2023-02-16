from QUANTAXIS.QAFactor.localize import QA_ts_update_daily_basic_single, QA_fetch_get_daily_basic


if __name__ == "__main__":
    missing_date_list = ['2017-02-23', '2017-05-09', '2017-07-20', '2017-09-28', '2018-05-17']
    df = QA_fetch_get_daily_basic(trade_date='2017-02-23')

    print(df)