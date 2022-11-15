from Invoker import Invoker
from utils import *

print("0000000000000000000000000000000000000000000000000000000000000000000000000000000000")
print("0000000000000000000000000000000000000000000000000000000000000000000000000000000000")

invoker = Invoker()
invoker.load_open_day()
invoker.load_close_day()
invoker.load_money_day()

print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111")
print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111")

for i in invoker.available_dname:
    if invoker.data[i] is not None:
        print(invoker.data[i])
    else:
        continue

print("2222222222222222222222222222222222222222222222222222222222222222222222222222222222")
print("2222222222222222222222222222222222222222222222222222222222222222222222222222222222")

invoker.release_memory(dname=["open", "money"])

print("3333333333333333333333333333333333333333333333333333333333333333333333333333333333")
print("3333333333333333333333333333333333333333333333333333333333333333333333333333333333")

for i in invoker.available_dname:
    if invoker.data[i] is not None:
        print(invoker.data[i])
    else:
        continue

print("4444444444444444444444444444444444444444444444444444444444444444444444444444444444")
print("4444444444444444444444444444444444444444444444444444444444444444444444444444444444")

invoker.release_memory(clear_all=True)
invoker.load_stock_day()

invoker.truncate()

for i in invoker.available_dname:
    if invoker.data[i] is not None:
        print(i)
    else:
        continue

print("5555555555555555555555555555555555555555555555555555555555555555555555555555555555")
print("5555555555555555555555555555555555555555555555555555555555555555555555555555555555")

popularity = invoker.Popularity()
print(popularity)

print("6666666666666666666666666666666666666666666666666666666666666666666666666666666666")
print("6666666666666666666666666666666666666666666666666666666666666666666666666666666666")

ic = invoker.IC(popularity, cumulative=True)
print(ic)

print("7777777777777777777777777777777777777777777777777777777777777777777777777777777777")
print("7777777777777777777777777777777777777777777777777777777777777777777777777777777777")

draw_line(ic, jupyter=False, description="Cumulative IC of factor Popularity")