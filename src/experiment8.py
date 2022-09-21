# This script checks how many sentiment changes each detector detects in the KDD17 and the Stocknet dataset.

import os
import pandas
import sys
from detecta import detect_cusum
from skmultiflow.drift_detection import adwin
from statistics import mean


def detect(path, detector):

    numberOfDrifts = 0

    for stockCode in os.listdir(path):
        sentimentScores = []
        data = pandas.read_csv(path + '/' + stockCode, index_col="Date", parse_dates=["Date"])

        if len(data) < 1:
            break
        currentDate = data["Date"][0]
        currentSentimentScore = []
        for i in range(len(data)):
            if data["Date"][i] == currentDate:
                currentSentimentScore.append(data["sentimentScore"][i])
            else:
                sentimentScores.append(mean(currentSentimentScore))
                currentSentimentScore = [data["sentimentScore"][i]]
                currentDate = data["Date"][i]

        if detector == "adwin":
            SCD = adwin.ADWIN(10)

            for i in range(len(sentimentScores)):
                SCD.add_element(sentimentScores[i])
                if SCD.detected_change():
                    numberOfDrifts += 1

        elif detector == "cusum":
            numberOfDrifts = numberOfDrifts + len(detect_cusum(sentimentScores, show=False)[0].tolist())

        else:
            print("Parameter error: detector=" + str(detector) + " is not a valid sentiment change detector")
            exit(-1)

    return numberOfDrifts


if __name__ == "__main__":
    """""
        Parameters:
            args[0] --> Controls which sentiment change detector is going to be selected.
            Can be set to equal to {"adwin", "cusum"}
    """""

    # Handling Parameters
    if len(sys.argv) > 1:
        detector = sys.argv[1].lower().strip()
    else:
        detector = "cusum"

    # Variable Declaration
    KDD17NumberOfDrifts = detect("../data/kdd17/sentiment-scores", detector)
    StockNetNumberOfDrifts = detect("../data/stocknet/sentiment-scores", detector)

    print("Total number of drifts detected by the " + detector + " concept drift detector:")
    print("KDD17 = " + str(KDD17NumberOfDrifts))
    print("StockNet = " + str(StockNetNumberOfDrifts))
    print("Total = " + str(KDD17NumberOfDrifts + StockNetNumberOfDrifts))
