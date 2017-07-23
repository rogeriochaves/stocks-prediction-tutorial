# coding: utf-8

# # Predicting GOOG Stocks
#
# ## Setup

# In[0]:

import pandas as pd
import quandl as Quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot
from matplotlib import style
import os
from lib.stocks import load_stock

style.use('ggplot')

# This is the data we have available from Quandl

# In[1]:

# 'WIKI/GOOG'

# Use 21 last days of data to predict the stock value
forecast_range = 21

Quandl.ApiConfig.api_key = os.environ['QUANDL_KEY']


def train_with(classifier, stock):
    data = load_stock(stock)

    # ## Label and Features

    # In[2]:

    data['High To Low Percentage'] = (
        data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100
    data['Change Percentage'] = (
        data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100

    data = data[['Adj. Close', 'Adj. Volume']]
    data['Month'] = data.index.month
    data['Year'] = data.index.year

    forecast_col = 'Adj. Close'

    data['label'] = data[forecast_col].shift(-forecast_range)

    # X is features, everything but label
    X = np.array(data.drop(['label'], 1))
    X = preprocessing.scale(X)

    # Features for the last 21 days
    X_lately = X[-forecast_range:]
    # All features but the latest 21 days
    X = X[:-forecast_range]
    # drop NaNs or else we get an error while trying to run it through the classification algorithm, they will exist because of the previous shift
    data.dropna(inplace=True)
    y = np.array(data['label'])

    print("X (features, what we use for prediction)", X)
    print("y (label, what we want to predict)", forecast_col, y)

    # ## Train, Test and Prediction

    # In[3]:

    # split data for be used later for training and testing
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.2)

    # train
    classifier.fit(X_train, y_train)
    # test
    accuracy = classifier.score(X_test, y_test)
    print("Accuracy", accuracy)
    return X_lately, classifier

    # end setup, now to the prediction!


classifier = LinearRegression()
_, classifier = train_with(classifier, 'GOOG')
_, classifier = train_with(classifier, 'FB')
_, classifier = train_with(classifier, 'GOOGL')
X_lately, classifier = train_with(classifier, 'AAPL')

print("X_lately", X_lately)
# predict
forecast_result = classifier.predict(X_lately)

print("AAPL prices for the next", forecast_range, "days")
print(forecast_result)

# ## Graphics

# ### Predicted Prices

# In[4]:
#
# data['Forecast'] = np.nan
#
# last_date = data.iloc[-1].name
# last_unix = last_date.timestamp()
# one_day = 86400
# next_unix = last_unix + one_day
#
# for i in forecast_result:
#     next_date = datetime.datetime.fromtimestamp(next_unix)
#     next_unix += one_day
#     data.loc[next_date] = [np.nan for _ in range(len(data.columns) - 1)] + [i]
#
# prediction_data = data[-forecast_range:]
# prediction_data['Forecast'].plot()
# plot.legend(loc=4)
# plot.xlabel('Date')
# plot.ylabel('Price')
# plot.show()
#
# # ### Full Prices History
#
# # In[5]:
#
# data['Adj. Close'].plot()
# data['Forecast'].plot()
# plot.legend(loc=4)
# plot.xlabel('Date')
# plot.ylabel('Price')
# plot.show()
