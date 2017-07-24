import matplotlib.pyplot as plot
from matplotlib import style
from lib.trainer import features, trained_classifier, forecast_range, capping_range
from lib.stocks import load_stock
from lib.features import append_feature
import numpy as np
import time
import datetime
import pandas as pd

style.use('ggplot')

classifier = trained_classifier()


def simulate_prediction(stock):
    real_stock_data = load_stock(stock)

    data_used_to_predict = real_stock_data[
        -(capping_range() + forecast_range()):-capping_range()][[
            'Adj. Close', 'Adj. Volume'
        ]]

    predicted_prices = []
    for i in range(0, forecast_range()):
        next_price = price_for_next_day(data_used_to_predict)
        predicted_prices = predicted_prices + [next_price]
        append_feature(data_used_to_predict, next_price)

    data_that_should_have_been_predicted = real_stock_data[
        -capping_range():-(capping_range() - forecast_range())]
    real_prices = np.array(
        data_that_should_have_been_predicted['Adj. Close'].tolist())

    return np.array(predicted_prices), real_prices


def price_for_next_day(data):
    return classifier.predict(features(data))[-1]


def plot_prediction(stock, prediction, real_prices):
    print("Starting at", capping_range(), "days ago:")

    print("Predicted", stock, "prices next", forecast_range(), "days")
    print(prediction)

    print("Real", stock, "prices next", forecast_range(), "days")
    print(real_prices)

    pd.Series(prediction).plot()
    pd.Series(real_prices).plot()
    plot.xlabel('Days')
    plot.ylabel('Price')
    plot.show()


def simulate_and_plot_prediction(stock):
    prediction, real_prices = simulate_prediction(stock)
    plot_prediction(stock, prediction, real_prices)
