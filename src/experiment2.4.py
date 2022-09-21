# This script plots sentiment scores of two entered stocks.

import sys
from functions import plot, validatePlot


if __name__ == "__main__":
    """""
        Parameters:
            args[0] --> Stock in the KDD17 dataset to be plotted
            args[1] --> Stock in the StockNet dataset to be plotted
    """""

    # Handling Parameters
    if len(sys.argv) > 2:
        validatePlot("../data/kdd17/sentiment-scores/" + sys.argv[1] + ".csv", sys.argv[1], "sentimentScore", "Sentiment Score")
        validatePlot("../data/stocknet/sentiment-scores/" + sys.argv[2] + ".csv", sys.argv[2], "sentimentScore","Sentiment Score")
    elif len(sys.argv) > 1:
        print("Parameter Error: Two parameters need to be entered")
        exit(-1)
    else:
        plot("../data/kdd17/sentiment-scores/MSFT.csv", "MSFT", "sentimentScore", "Sentiment Score")
        plot("../data/kdd17/sentiment-scores/VALE.csv", "VALE", "sentimentScore", "Sentiment Score")
