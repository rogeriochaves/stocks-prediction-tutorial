import matplotlib.pyplot as plot
from matplotlib import style
from lib.trainer import features, trained_classifier, forecast_range
from lib.stocks import load_stock
import numpy as np
import time
import datetime
import pandas as pd

style.use('ggplot')

classifier = trained_classifier()


def simulate_prediction(stock):
    real_stock_data = load_stock(stock)

    data_used_to_predict = real_stock_data[-forecast_range() * 2:
                                           -forecast_range()]
    predicted_prices = classifier.predict(features(data_used_to_predict))

    data_to_be_predicted = real_stock_data[-forecast_range():]
    real_prices = np.array(data_to_be_predicted['Adj. Close'].tolist())

    return predicted_prices, real_prices


def plot_prediction(stock, prediction, real_prices):
    print("Predicted", stock, "prices for the last", forecast_range(), "days")
    print(prediction)

    print("Real", stock, "prices for the last", forecast_range(), "days")
    print(real_prices)

    pd.Series(prediction).plot()
    pd.Series(real_prices).plot()
    plot.xlabel('Days')
    plot.ylabel('Price')
    plot.show()


def simulate_and_plot_prediction(stock):
    prediction, real_prices = simulate_prediction(stock)
    plot_prediction(stock, prediction, real_prices)
