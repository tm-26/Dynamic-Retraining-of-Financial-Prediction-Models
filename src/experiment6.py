# This script checks how many concept drifts each concept drift detector detects in the KDD17 and the Stocknet dataset.

import os
import pandas
import skmultiflow.drift_detection
import sys
from copy import deepcopy
from financialCDD import minps, mySD, myTanDD


def detect(path, detector):

    numberOfDrifts = 0

    for stockCode in os.listdir(path):
        data = pandas.read_csv(path + '/' + stockCode, index_col="Date", parse_dates=["Date"])["Close"]

        currentDetector = deepcopy(detector)

        previous = 0
        for i in range(len(data)):
            if type(currentDetector) == skmultiflow.drift_detection.EDDM:
                if previous < data[i]:
                    currentDetector.add_element(1)
                else:
                    currentDetector.add_element(0)
                previous = data[i]
            else:
                currentDetector.add_element(data[i])

            if currentDetector.detected_change():
                if type(currentDetector) in [minps.MINPS, mySD.mySDDD, myTanDD.myTanDD]:
                    currentDetector.reset()
                numberOfDrifts += 1

    return numberOfDrifts


if __name__ == "__main__":
    """""
        Parameters:
            args[0] --> Controls which concept drift detector is going to be selected.
            Can be set to equal to {"eddm", "hddma", "hddmw", "minps", "mysd", "mytandd", "ph"}
    """""

    # Handling Parameters
    if len(sys.argv) > 1:
        detector = sys.argv[1].lower().strip()
    else:
        detector = "hddma"

    if detector == "eddm":
        KDD17Detector = skmultiflow.drift_detection.eddm.EDDM()
        StockNetDetector = skmultiflow.drift_detection.eddm.EDDM()
    elif detector == "hddma":
        KDD17Detector = skmultiflow.drift_detection.hddm_a.HDDM_A()
        StockNetDetector = skmultiflow.drift_detection.hddm_a.HDDM_A()
    elif detector == "hddmw":
        KDD17Detector = skmultiflow.drift_detection.hddm_w.HDDM_W()
        StockNetDetector = skmultiflow.drift_detection.hddm_w.HDDM_W()
    elif detector == "minps":
        KDD17Detector = minps.MINPS(20)
        StockNetDetector = minps.MINPS(20)
    elif detector == "mysd":
        KDD17Detector = mySD.mySDDD(20)
        StockNetDetector = mySD.mySDDD(20)
    elif detector == "mytandd":
        KDD17Detector = myTanDD.myTanDD(20)
        StockNetDetector = myTanDD.myTanDD(20)
    elif detector == "ph":
        KDD17Detector = skmultiflow.drift_detection.page_hinkley.PageHinkley()
        StockNetDetector = skmultiflow.drift_detection.page_hinkley.PageHinkley()
    else:
        print("Parameter error: detector=" + str(detector) + " is not a valid concept drift detector")
        exit(-1)

    KDD17NumberOfDrifts = detect("../data/kdd17/numerical/price_long_50", KDD17Detector)
    StockNetNumberOfDrifts = detect("../data/stocknet/numerical/price_long_50", StockNetDetector)

    print("Total number of drifts detected by the " + detector + " concept drift detector:")
    print("KDD17 = " + str(KDD17NumberOfDrifts))
    print("StockNet = " + str(StockNetNumberOfDrifts))
    print("Total = " + str(KDD17NumberOfDrifts + StockNetNumberOfDrifts))