from Density import Density


class BivariateCopula(object):
    def __init__(self):
        self.marginals_1 = self.univariate_func_convertor(func=Density.uniform_pdf)
        self.marginals_2 = self.univariate_func_convertor(func=Density.uniform_pdf)


    @staticmethod
    def univariate_func_convertor(func):
        lambda_func = lambda x: func(x)
        return lambda_func


if __name__ == "__main__":
    bc = BivariateCopula()
    print(bc.marginals_1(Density.uniform_pdf(0.5)))



