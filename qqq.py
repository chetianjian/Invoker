from QUANTAXIS.QAFactor.localize import QA_ts_update_daily_basic_single

date_list = ['2017-08-14', '2017-09-19', '2021-06-29']

for date in date_list:
    QA_ts_update_daily_basic_single(date)