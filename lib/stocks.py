import quandl
import pickle
import os

quandl.ApiConfig.api_key = os.environ['QUANDL_KEY']
dump_path = os.path.join(os.path.dirname(__file__), "..", "stocks")


def fetch_stock(stock):
    print("Downloading " + stock + " stocks")
    data = quandl.get("WIKI/" + stock)
    pickle.dump(data, open(dump_path + "/" + stock, "wb"))


def load_stock(stock):
    return pickle.load(open(dump_path + "/" + stock, "rb"))
