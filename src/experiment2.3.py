# This script counts the number of positive, neutral and negative labels.

import csv
import os
from matplotlib import pyplot


def labelCounter(dataset):
    labels = {"positive": 0, "neutral": 0, "negative": 0}

    for stock in os.listdir("../data/" + dataset + "/sentiment-scores/"):
        with open("../data/" + dataset + "/sentiment-scores/" + stock, encoding="utf-8") as file:
            csvReader = csv.reader(file)
            next(csvReader)  # Skip headers
            csvReader = list(csvReader)
            for row in csvReader:
                labels[row[2]] += 1

    figure, axes = pyplot.subplots()
    bars = axes.bar(*zip(*labels.items()))
    axes.bar_label(bars)

    axes.set_xlabel("Year")
    axes.set_ylabel("Numer of articles")
    pyplot.show()


if __name__ == "__main__":
    labelCounter("kdd17")
    labelCounter("stocknet")
