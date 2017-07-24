import pandas as pd
import numpy as np
from functools import reduce


def append_feature(data, price):
    next_index = data.index[-1] + pd.DateOffset(1)
    data.loc[next_index] = [price, data['Adj. Volume'][-1]]
    return data


def features(data):
    X = data[['Adj. Close', 'Adj. Volume']]
    X['Month'] = data.index.month
    X['Year'] = data.index.year
    X['Last 14 days diff'] = [avg_diff(data, i) for i in range(0, len(data))]
    return X


def avg_diff(data, i):
    initial_index = max(0, i - 14)
    initial = data['Adj. Close'][initial_index]
    sum = reduce((lambda a, c: a + c), data['Adj. Close'][initial_index:i], 0)
    return (sum / 14) - initial


def label(data):
    y = data['Adj. Close'].shift(-1)
    y.dropna(inplace=True)
    return np.array(y)
