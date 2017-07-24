import pandas as pd
import quandl as Quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot
import os
from lib.stocks import load_stock
from lib.features import label, features

# Use 21 last days of data to predict the stock value


def forecast_range():
    return 21


def capping_range():
    return 42


def load_capped_stock(stock):
    # Load the stocks without the latest days to check prediction
    return load_stock(stock)[:-capping_range()]


def extract_features_and_label(data):
    X = features(data)
    y = label(data)

    # Make X the same size as y
    X = X[:len(y)]

    return X, y


def train_with(classifier, stock):
    data = load_capped_stock(stock)
    X, y = extract_features_and_label(data)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.2)

    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)
    print(stock, "accuracy", accuracy)

    return classifier


def trained_classifier():
    classifier = LinearRegression()
    classifier = train_with(classifier, 'TWTR')
    classifier = train_with(classifier, 'GOOG')
    classifier = train_with(classifier, 'FB')
    classifier = train_with(classifier, 'GOOGL')
    classifier = train_with(classifier, 'AAPL')
    return classifier
