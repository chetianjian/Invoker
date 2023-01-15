import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata


def rolling_rank(arr):
    """
    :param arr: Numpy array.
    :return: The rank of the last value in the array.
    """

    return rankdata(arr)[-1]


def stddev(df, window=1):
    """
    Rolling standard deviation.
    :param df: Target DataFrame.
    :param window: Rolling window size.
    :return: DataFrame consists of standard deviations over the
             past-window-days for each line of the original input.
    """

    return df.rolling(window).std()


def ts_rank(df, window=1):
    """
    :param df: Target DataFrame.
    :param window: Default to 1. Rolling window size.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """

    return df.rolling(window).apply(rolling_rank)


def ts_min(df, window) -> pd.DataFrame:
    """
    :param df: Input dataframe.
    :param window: The rolling window size.
    :return: Time-series minimum over the past 'window' days.
    """

    return df.rolling(window=window).min()


def ts_max(df, window) -> pd.DataFrame:
    """
    :param df: Input dataframe.
    :param window: The rolling window size.
    :return: Time-series maximum over the past 'window' days.
    """

    return df.rolling(window=window).max()


def ts_argmax(df, window):
    """
    :param df: Input dataframe.
    :param window: The rolling window size.
    :return:
    """

    return df.rolling(window=window).apply(np.argmax) + 1


def decay_linear(df, window):
    """
    Weighted moving average over the past 'window' days with linearly decaying weights: window, window – 1, …, 1
    (rescaled to sum up to 1)
    :param df: Target pd.DataFrame.
    :param window: The linear weighted moving average period.
    :return: The rescaled pd.DataFrame of linear weighted moving average.
    """

    result = df * window
    for _ in range(1, window):
        result = result + df.shift(_) * (window - _)

    return 2 * result / (window + window**2)


def corr(df1, df2, window) -> pd.DataFrame:
    """
    :param df1: First dataframe.
    :param df2: Second dataframe.
    :param window: The rolling window size.
    :return: Rolling correlations between dataframes df1 and df, meaning that
             df1 and df2 should be identical in their shapes.
    """

    try:
        assert df1.shape == df2.shape
    except:
        msg = f"""
        Input dataframes should be identical in their shapes. \n
        Received shapes: {df1.shape} and {df2.shape}
        """
        raise AssertionError(msg)

    return df1.rolling(window=window).corr(df2)


def covar(df1: pd.DataFrame, df2: pd.DataFrame, window) -> pd.DataFrame:
    """
    :param df1: First dataframe.
    :param df2: Second dataframe.
    :param window: The rolling window size.
    :return: Rolling covariances between dataframes df1 and df, meaning that
             df1 and df2 should be identical in their shapes.
    """

    try:
        assert df1.shape == df2.shape
    except:
        msg = f"""
        Input dataframes should be identical in their shapes. \n
        Received shapes: {df1.shape} and {df2.shape}
        """
        raise AssertionError(msg)

    return df1.rolling(window=window).cov(df2)


def scale(df, alpha=1):
    """
    :param df: Input dataframe.
    :param alpha: Scaling factor, default to 1.
    :return: Rescaled dataframe such that sum(abs(df)) = alpha.
    """

    return alpha / np.abs(df).sum() * df





def seriesPosNegSumRatio(series):
    """
    :param series: 一列数据
    :return: 将该列数据所有正数与负数分别求和，再求比值: 正数之和 / 绝对值(负数之和)
    """

    positive_sum, negative_sum = np.nansum(series[series >= 0]), np.nansum(series[series < 0])
    return np.nan if negative_sum == 0 else positive_sum / abs(negative_sum)


def arrAvgAbs(arr, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param arr: 一列数据
    :return: 计算数组的平均绝对偏差。偏差表示每个数值与平均值之间的差，平均偏差表示每个偏差绝对值的平均值。
    """

    result = np.nanmean(abs(arr - np.nanmean(arr)))
    return result if not fillna else fillna


def rowWeighted(arr, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param arr: 一列数据
    :return: 各自按自身占总体的比例加权
    """

    result = arr / np.nansum(arr)
    return result if not fillna else result.fillna(fillna)


def arrNormalize(arr, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param arr: 一列数据
    :return: 令行向量模长 = 1，即 L^2 范数为 1。
    """

    result = arr / np.nansum(arr ** 2)
    return result if not fillna else result.fillna(fillna)


def arrStandardize(arr, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param arr: 一列数据
    :return: (arr - mean(arr)) / std(arr)
    """

    result = (arr - arr.mean()) / arr.std()
    return result if not fillna else result.fillna(fillna)


def seriesStandardize(series, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param series: 一列数据
    :return: (series - mean(series)) / std(series)
    """

    if type(series) == pd.DataFrame:
        series = series.iloc[:, 0]
    result = (series - np.nanmean(series)) / np.nanstd(series)
    return result if not fillna else result.fillna(fillna)


def mvNeutralize(df: pd.DataFrame, mv: pd.DataFrame, fillna=False) -> pd.DataFrame:
    """
    :param df: Objective DataFrame which is going to be Neutralized.
    :param mv: DataFrame which records MV data.
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :return: Factor DataFrame neutralized cross-sectionally by Market Value.
    """

    assert df.shape == mv.shape
    data, mv = df.fillna(0), mv.fillna(0)
    for row in range(len(data)):
        features = data.iloc[row, :].values.reshape(-1, 1)
        pred = LinearRegression().fit(X=features, y=mv.iloc[row, :].values).predict(X=features)
        data.iloc[row, :] = data.iloc[row, :] - pred

    return data if not fillna else data.fillna(fillna)


def EMA(df: pd.DataFrame, window, fillna=False) -> pd.DataFrame:
    """
    :param df: Objective DataFrame.
    :param window: Use to compute alpha: alpha=2 / (window+1)
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :return: Exponential Moving Average.
    """

    return df.ewm(alpha=2 / (window + 1)).mean() if not fillna else df.ewm(alpha=2 / (window + 1)).mean().fillna(fillna)


def impluse(arr):
    result, i = 0, 0
    while i < len(arr):
        if np.isnan(arr[i]):
            i += 1
            continue
        direction = arr[i] > 0
        j = i
        while j + 1 < len(arr) and (arr[j + 1] > 0) == direction and not np.isnan(arr[j + 1]):
            j += 1
        if direction:
            result += (j - i + 1) ** 2
        else:
            result -= (j - i + 1) ** 2
        i = j + 1
    return result


def grad_desc_geometry(w, accuracy=1e-7, max_iter=10000000, step=5e-6):
    common_ratio = 1 / w ** 2
    result = common_ratio ** (w - 1) - w * common_ratio + w - 1
    for i in range(max_iter):
        gradient = (w - 1) * common_ratio ** (w - 2) - w + 1
        common_ratio -= step * gradient
        new_result = common_ratio ** (w - 1) - w * common_ratio + w - 1
        diff = abs(new_result - result)

        if i % 1000000 == 0:
            print(f"Completed {i} iterations, minimum approximates: {common_ratio}, accuracy: {diff}")
            if diff < accuracy:
                print(f"Accuracy 1e-7 triggered, minimum approximates: {common_ratio}, accuracy: {diff}")
                print("Algorithm Terminated.")
                break
        result = new_result
    return common_ratio


def dfReLU(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only non-positive values in a DataFrame, and replace all other values by 0.
    return df[df >= 0].fillna(0)


def dfRemoveInf(df: pd.DataFrame, fillna=False) -> pd.DataFrame:
    """
    :param df: Objective DataFrame.
    :param fillna: Whether to keep NaNs, default is False, otherwise input a value.
    :return: Remove all infinite values within a DataFrame.
    """

    return df[~np.isinf(df)] if not fillna else df[~np.isinf(df)].fillna(fillna)


def jackknife(series: np.array, method) -> tuple:
    """
    :param series: An array.
    :param method: Type of estimation: str in ["variation", "variance", "mean"]
    :return: (Jackknife Estimation, Jackknife Bias, Jackknife Variance)
    """

    if not series.sum():
        return np.nan, np.nan, np.nan
    theta_j_lst, stack = [], 0

    if method == "variation":
        estimator = lambda arr: np.nanstd(arr) / np.nanmean(arr)
    elif method == "variance":
        estimator = lambda arr: np.nanvar(arr)
    elif method == "mean":
        estimator = lambda arr: np.nanmean(arr)
    else:
        estimator = None

    assert estimator is not None

    array = series.values

    for j in range(len(array)):
        leaved = np.delete(array, j)
        theta_j_lst.append(estimator(leaved))

    estimate = estimator(array)
    Jack_Bias = (len(array) - 1) * (np.nansum(theta_j_lst) / len(array) - estimate)

    for j in range(len(array)):
        stack += (theta_j_lst[j] - np.nansum(theta_j_lst) / len(array)) ** 2
    Jack_Var = (1 - 1 / len(array)) * stack

    return estimate - Jack_Bias, Jack_Bias, Jack_Var


def CP(series: pd.Series):
    series = (series.diff(1) / series.shift(1)).reset_index(drop=True)
    if not series.any():
        return np.nan
    avg, std = series.mean(), series.std()
    upper = series[series > avg + std]
    lower = series[series < avg - std]
    if len(upper) == 0:
        upper = 0
    else:
        upper = np.nanmedian(upper.index)
    if len(lower) == 0:
        lower = 0
    else:
        lower = np.nanmedian(lower.index)
    return lower - upper


def tFilter(signal: pd.Series) -> pd.Series:
    """
    :param signal: pd.Series that records the trading decision for each day.
            -1 denotes "short", 1 denotes "long", and 0 denotes "hold current position".
    :return: A filtered decision flow (series) where long decisions and short decisions
                are alternating. Therefore, we can simply focus on whether we are holding
                a position currently, and temporarily ignore the weight of position
                that we hold.
    """

    if signal is None:
        return np.nan
    direction, flst = signal[0], [True]
    for _ in signal[1:]:
        if _ == direction:
            flst.append(False)
        else:
            flst.append(True)
            direction = _
    return signal[flst]


def tFilteredReturn(code: str, signal: pd.Series, rdf: pd.DataFrame) -> float:
    """
    :param code: Code of a specific stock, format as "000001"
    :param signal: pd.Series that records the trading decision for each day.
            -1 denotes "short", 1 denotes "long", and 0 denotes "hold current position".
    :param rdf: pd.DataFrame, basically ought to be Invoker.rate or Invoker.data["rate"].
    :return: A float which stands for the cumulative simple rate of return.
    """

    cr, idx, rate = 0, signal.index, rdf[code]
    for _ in range(len(signal) - 1):
        start, end = idx[_], idx[_ + 1]
        cr += signal[_] * (rate.loc[start: end].sum() - rate[idx[_]] + rate[idx[_ + 1]])
    return cr


########################################################################################################################
########################################################################################################################
########################################################################################################################


def draw_line(series, legend=None, jupyter=True, color="blue", description=None):
    if type(series) == pd.DataFrame:
        series = series.iloc[:, 0]

    traces = []
    trace = plotly.graph_objs.Scattergl(
        name=legend,
        x=series.index,
        y=series.values,
        line=dict(color=color)
    )

    traces.append(trace)

    if description:
        layout = plotly.graph_objs.Layout(
            title=description
        )
    else:
        layout = plotly.graph_objs.Layout(
            title="Plot series data of: " + series.name
        )

    fig = plotly.graph_objs.Figure(data=traces, layout=layout)
    if jupyter:
        plotly.offline.init_notebook_mode(connected=True)
    return plotly.offline.iplot(fig, filename="dataplot")


def draw_lines(series_list, color_list, legend_list, jupyter=True, description=None):
    assert len(series_list) == len(color_list) == len(legend_list)

    traces = []

    for _ in range(len(series_list)):
        if type(series_list[_]) == pd.DataFrame:
            series_list[_] = series_list[_].iloc[:, 0]

        trace = plotly.graph_objs.Scattergl(
            name=legend_list[_],
            x=series_list[_].index,
            y=series_list[_].values,
            line=dict(color=color_list[_])
        )

        traces.append(trace)

    if description:
        layout = plotly.graph_objs.Layout(
            title=description
        )
    else:
        layout = plotly.graph_objs.Layout(
            title="Plot series data of: " + series_list[0].name
        )

    fig = plotly.graph_objs.Figure(data=traces, layout=layout)
    if jupyter:
        plotly.offline.init_notebook_mode(connected=True)
    return plotly.offline.iplot(fig, filename="dataplot")


def draw_df_lines(df: pd.DataFrame, color_list, jupyter=True, title=""):
    assert df.shape[1] == len(color_list)

    traces = []

    dfcol = df.columns
    for _ in range(df.shape[1]):
        trace = plotly.graph_objs.Scattergl(
            name=dfcol[_],
            x=df.index,
            y=df[dfcol[_]].values,
            line=dict(color=color_list[_])
        )

        traces.append(trace)

    layout = plotly.graph_objs.Layout(
        title="Plot series data of: " + title
    )

    fig = plotly.graph_objs.Figure(data=traces, layout=layout)
    if jupyter:
        plotly.offline.init_notebook_mode(connected=True)
    return plotly.offline.iplot(fig, filename="dataplot")


def draw_candle(df: pd.DataFrame, n=0):
    """
    :param df: pd.DataFrame which stores the required data.
    :param n: Default to 0, which means all data are required to be included.
              Draw the candle plot for the last-n-days.
    :return: None.
    """
    df = df.iloc[-n:, :]
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"])])
    fig.show()
