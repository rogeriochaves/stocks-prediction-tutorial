import matplotlib.pyplot as plot
from matplotlib import style
from lib.trainer import features, trained_classifier, forecast_range
from lib.stocks import load_stock
from lib.features import append_feature
import numpy as np
import time
import datetime
import pandas as pd

style.use('ggplot')

classifier = trained_classifier(load_stock)


def predict(stock):
    data_used_to_predict = load_stock(stock)[-forecast_range():][[
        'Adj. Close', 'Adj. Volume'
    ]]

    predicted_prices = []
    for i in range(0, forecast_range()):
        next_price = price_for_next_day(data_used_to_predict)
        predicted_prices = predicted_prices + [next_price]
        append_feature(data_used_to_predict, next_price)

    return np.array(predicted_prices)


def price_for_next_day(data):
    return classifier.predict(features(data))[-1]


def plot_prediction(stock, prediction):
    print("Predicted", stock, "prices next", forecast_range(), "days")
    print(prediction)

    pd.Series(prediction).plot()
    plot.xlabel('Days')
    plot.ylabel('Price')
    plot.show()


def predict_and_plot(stock):
    prediction = predict(stock)
    plot_prediction(stock, prediction)
