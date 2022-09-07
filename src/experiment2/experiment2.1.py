# This script plots the Adj Stock price of two entered stocks
import sys
from plotter import *

if __name__ == "__main__":
    """""
        Parameters:
            args[0] --> Stock in the KDD17 dataset to be plotted
            args[1] --> Stock in the StockNet dataset to be plotted
    """""

    # Handling Parameters
    if len(sys.argv) > 2:
        validateParameter("../../data/kdd17/numerical/price_long_50/" + sys.argv[1] + ".csv", sys.argv[1], "Adj Close")
        validateParameter("../../data/stocknet/numerical/price_long_50/" + sys.argv[2] + ".csv", sys.argv[2], "Adj Close")
    elif len(sys.argv) > 1:
        print("Parameter Error: Two parameters need to be entered")
        exit(-1)
    else:
        plot("../../data/kdd17/numerical/price_long_50/MSFT.csv", "MSFT", "Adj Close")
        plot("../../data/stocknet/numerical/price_long_50/V.csv", "V", "Adj Close")
