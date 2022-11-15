from Invoker import Invoker
from utils import *


invoker = Invoker()
invoker.load_stock_day()
invoker.truncate()
print(invoker.rate)



RS_100 = invoker.data["rate"].rolling(100).apply(lambda series: np.nansum(series[series > 0])) /\
         invoker.data["rate"].rolling(100).apply(lambda series: abs(np.nansum(series[series < 0])))
RSI_100 = 100 - 100 / (1 + RS_100)


print(RSI_100)
print(RSI_100)
