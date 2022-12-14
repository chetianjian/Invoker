import numpy as np
import pandas as pd

def impluse(arr):
    if sum(np.isnan(arr)) == len(arr):
        return np.nan
    result = 0
    for i in range(len(arr)):
        if np.isnan(i):
            continue
        direction = arr[i] > 0
        j = i
        while j < len(arr) and (arr[j] > 0) == direction:
            j += 1
        if direction:
            result += (j - i) ** 2
        else:
            result -= (j - i) ** 2
    return result



    def row_normalize(row):
        return row / np.nansum(row) if np.nansum(row) != 0 else row / np.nan

    bias = (close - close.shift(window)) / (100 * close.shift(window))
    return bias.apply(lambda row: row_normalize(row), axis=1)





def impluse_10(return_adj, window=10):
    import numpy as np

    def impluse(arr):
        result = 0
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                continue
            direction = arr[i] > 0
            j = i
            while j + 1 < len(arr) and (arr[j] > 0) == direction and not np.isnan(arr[j]):
                j += 1
            if direction:
                result += (j - i) ** 2
            else:
                result -= (j - i) ** 2
        return result

    return return_adj.rolling(window).apply(impluse)


def long_impluse_5_max_linear_version(return_adj, window=5):
    import numpy as np

    def impluse(arr):
        positive = 0
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                continue
            if arr[i] > 0:
                j = i
                while j + 1 < len(arr) and arr[j] > 0:
                    j += 1
                if j - i > positive:
                    positive = j - i
        return positive

    return return_adj.rolling(window).apply(impluse)


def short_impluse_5_min_version(return_adj, window=5):
    def impluse(arr):
        if sum(np.isnan(arr)) == len(arr):
            return np.nan
        negative = 0
        for i in range(len(arr)):
            if arr[i] < 0:
                j = i + 1
                while j < len(arr) and arr[j] < 0:
                    j += 1
                    if j - i > negative:
                        negative = j - i
        return negative**2

    return return_adj.rolling(window).apply(impluse)


def short_impluse_5_max_linear_version(return_adj, window=5):
    import numpy as np

    def impluse(arr):
        negative = 0
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                continue
            if arr[i] < 0:
                j = i + 1
                while j < len(arr) and arr[j] < 0:
                    j += 1
                if j - i > negative:
                    negative = j - i
        return negative

    return return_adj.rolling(window).apply(impluse)



