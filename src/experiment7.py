"""""
This script test for type II errors by testing the concept drift detectors on ten different periods in time when concept
drift occurred beyond any unreasonable doubt.
"""""

import os
import pandas
import skmultiflow.drift_detection
import sys
from financialCDD import minps, mySD, myTanDD


if __name__ == "__main__":
    """""
        Parameters:
            args[0] --> Controls which concept drift detector is going to be selected.
            Can be set to equal to {"eddm", "hddma", "hddmw", "minps", "mysd", "mytandd", "ph"}
    """""

    # Handle arguments

    # Handling Parameters
    if len(sys.argv) > 1:
        detectorType = sys.argv[1].lower().strip()
    else:
        detectorType = "hddma"

    for counter, stockName in enumerate(os.listdir("../data/driftEvents")):
        # Get data on each stock
        data = pandas.read_csv("../data/driftEvents/" + stockName, index_col="Date", parse_dates=["Date"])["Close"]

        # Detect concepts for each stock
        if detectorType == "eddm":
            detector = skmultiflow.drift_detection.eddm.EDDM()
        elif detectorType == "hddma":
            detector = skmultiflow.drift_detection.hddm_a.HDDM_A()
        elif detectorType == "hddmw":
            detector = skmultiflow.drift_detection.hddm_w.HDDM_W()
        elif detectorType == "minps":
            detector = minps.MINPS(20)
        elif detectorType == "mysd":
            detector = mySD.mySDDD(20)
        elif detectorType == "mytandd":
            detector = myTanDD.myTanDD(20)
        elif detectorType == "ph":
            detector = skmultiflow.drift_detection.page_hinkley.PageHinkley()
        else:
            print("Parameter error: detector=" + str(detectorType) + " is not a valid concept drift detector")
            exit(-1)

        status = 0
        previous = 0
        previousMonth = data.index[0].month
        duringEvent = 0
        afterEvent = 0
        for i in range(len(data)):

            if previousMonth != (data.index[0] + pandas.DateOffset(months=status)).month:
                status += 1

            previousMonth = data.index[i].month

            if type(detector) == skmultiflow.drift_detection.EDDM:
                if previous < data[i]:
                    detector.add_element(1)
                else:
                    detector.add_element(0)
                previous = data[i]
            else:
                detector.add_element(data[i])
            if detector.detected_change():
                if type(detector) in [minps.MINPS, mySD.mySDDD, myTanDD.myTanDD]:
                    detector.reset()

                if status > 1:
                    afterEvent += 1
                else:
                    duringEvent += 1

        print("-------------------------------Event #" + str(counter + 1) + "---------------------------------")
        print("Concept Drifts detected during event " + str(duringEvent) + " times")
        print("Concept Drifts detected after event " + str(afterEvent) + " times")
        print("------------------------------------------------------------------------")
