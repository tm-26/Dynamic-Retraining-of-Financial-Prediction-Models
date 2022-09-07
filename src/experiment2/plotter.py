import pandas
import sys
from matplotlib import pyplot


def plot(path, stockCode, type):

    # Get data
    data = pandas.read_csv(path, usecols=["Date", type])
    data["Date"] = pandas.to_datetime(data["Date"])

    # Plot it
    figure, axes = pyplot.subplots()
    axes.plot_date(data["Date"], data[type], 'ro')

    pyplot.title("Historical data of " + stockCode)
    axes.set_xlabel("Time")
    axes.set_ylabel("Adj Close Price ($)")
    axes.tick_params(axis='x', labelsize=9)
    pyplot.xticks(fontsize=6)
    pyplot.show()


def validateParameter(path, stockCode, type):
    try:
        plot(path, stockCode, type)
    except FileNotFoundError:
        print("Parameter Error: Stock=\"" + stockCode + "\" could not be found")
        exit(-1)
