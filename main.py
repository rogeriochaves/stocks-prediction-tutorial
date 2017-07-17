import pandas as pd
import quandl as Quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

print("Downloading data...")
df = Quandl.get('WIKI/GOOGL')
print("Data ready")

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['High To Low Percentage'] = (
    df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['Change Percentage'] = (
    df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[[
    'Adj. Close', 'High To Low Percentage', 'Change Percentage', 'Adj. Volume'
]]

forecast_col = 'Adj. Close'

# defines that the label is based of 10% of previous Adj. Close prices
forecast_range = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_range)

# X is features, everything but label
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
# Features for the last n days (like 33 days)
X_lately = X[-forecast_range:]
# All features but the latest n days (like 33 days)
X = X[:-forecast_range]
# drop NaNs or else we get an error while trying to run it through the classification algorithm, they will exist because of the previous shift
df.dropna(inplace=True)
y = np.array(df['label'])

# split data for be used later for training and testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2)

# end setup, now to the prediction!

classifier = LinearRegression()
# train
classifier.fit(X_train, y_train)
# test
accuracy = classifier.score(X_test, y_test)
# predict
forecast_result = classifier.predict(X_lately)

print("GOOG prices for the next", forecast_range, "days")
print(forecast_result)
print("Accuracy", accuracy)

# plotting graphic

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_result:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
