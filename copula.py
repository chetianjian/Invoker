from density import *


class BivariateCopula(object):
    def __init__(self):
        self.marginals_1 = self.univariate_func_convertor(func=uniform_pdf)
        self.marginals_2 = self.univariate_func_convertor(func=uniform_pdf)

    @staticmethod
    def univariate_func_convertor(func):
        lambda_func = lambda x: func(x)
        return lambda_func



bc = BivariateCopula()
print(bc.marginals_1(uniform_pdf(0.5)))



