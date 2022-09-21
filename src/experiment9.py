"""""
This script test for type II errors by testing the sentiment change detectors on two different periods in time when
sentiment change occurred beyond any unreasonable doubt.
"""""

import os
import pandas
import sys
from detecta import detect_cusum
from skmultiflow.drift_detection import adwin
from statistics import mean


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

    for counter, stockName in enumerate(os.listdir("../data/changeEvents")):
        # Get data on each stock
        data = pandas.read_csv("../data/changeEvents/" + stockName, index_col="Date", parse_dates=["Date"])
        data.sort_index(inplace=True)

        sentimentScores = {}
        currentDate = data.index[0]
        currentSentimentScore = []
        duringEvent = 0
        afterEvent = 0

        for i in range(len(data)):
            if data.index[i] == currentDate:
                currentSentimentScore.append(data["sentimentScore"][i])
            else:
                currentDate = data.index[i]
                sentimentScores[currentDate] = (mean(currentSentimentScore))
                currentSentimentScore = [data["sentimentScore"][i]]

        previousMonth = data.index[0].month
        status = 0

        if detector == "adwin":
            SCD = adwin.ADWIN(10)
            for date in sentimentScores.keys():

                if previousMonth != (data.index[0] + pandas.DateOffset(months=status)).month:
                    status += 1

                previousMonth = date.month

                SCD.add_element(sentimentScores[date])

                if SCD.detected_change():

                    if status > 1:
                        afterEvent += 1
                    else:
                        duringEvent += 1
        elif detector == "cusum":
            dates = list(sentimentScores.keys())
            sentimentScores = list(sentimentScores.values())
            changePoints = detect_cusum(sentimentScores, show=False)[0].tolist()

            for changePoint in changePoints:
                if dates[changePoint].month == (dates[0] + pandas.DateOffset(months=2)).month:
                    afterEvent += 1
                else:
                    duringEvent += 1
        else:
            print("Parameter error: detector=" + str(detector) + " is not a valid sentiment change detector")
            exit(-1)

        print("-------------------------------Event #" + str(counter + 1) + "---------------------------------")
        print("Number of sentiment changes detected during event " + str(duringEvent) + " times")
        print("Number of sentiment changes detected after event " + str(afterEvent) + " times")
        print("------------------------------------------------------------------------")

