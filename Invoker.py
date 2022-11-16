from Factor import Factor
from Strategy import Strategy


class Invoker(Factor, Strategy):
    def __init__(self):
        print("I am a beacon of knowledge blazing out across a black sea of ignorance.")
        super().__init__()