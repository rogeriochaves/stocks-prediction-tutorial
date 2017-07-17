import pandas as pd
import quandl as Quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot
from matplotlib import style

style.use('ggplot')

print("Downloading data...")
data = Quandl.get('WIKI/GOOGL')
print("Data ready")

data['High To Low Percentage'] = (
    data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100
data['Change Percentage'] = (
    data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100

data = data[[
    'Adj. Close', 'High To Low Percentage', 'Change Percentage', 'Adj. Volume'
]]

forecast_col = 'Adj. Close'

# defines that the label is based of 10% of previous Adj. Close prices
forecast_range = int(math.ceil(0.01 * len(data)))
data['label'] = data[forecast_col].shift(-forecast_range)

# X is features, everything but label
X = np.array(data.drop(['label'], 1))
X = preprocessing.scale(X)
# Features for the last n days (like 33 days)
X_lately = X[-forecast_range:]
# All features but the latest n days (like 33 days)
X = X[:-forecast_range]
# drop NaNs or else we get an error while trying to run it through the classification algorithm, they will exist because of the previous shift
data.dropna(inplace=True)
y = np.array(data['label'])

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

data['Forecast'] = np.nan

last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_result:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data.loc[next_date] = [np.nan for _ in range(len(data.columns) - 1)] + [i]

# prediction prices

prediction_data = data[-forecast_range:]
prediction_data['Forecast'].plot()
plot.legend(loc=4)
plot.xlabel('Date')
plot.ylabel('Price')
plot.show()

# full prices history

data['Adj. Close'].plot()
data['Forecast'].plot()
plot.legend(loc=4)
plot.xlabel('Date')
plot.ylabel('Price')
plot.show()
