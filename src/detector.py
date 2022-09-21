import os
import pandas
import skmultiflow.drift_detection
from datetime import timedelta
from detecta import detect_cusum
from statistics import mean
from financialCDD import minps, mySD, myTanDD


def sentimentChangeDetector(stream, SCD="cusum", changeConfidence=1):
    if SCD == "cusum":
        return detect_cusum(stream, changeConfidence, show=False)[0].tolist()
    elif SCD == "adwin":
        driftPoints = []
        SCD = skmultiflow.drift_detection.adwin.ADWIN()

        for i in range(len(stream)):
            SCD.add_element(stream[i])
            if SCD.detected_change():
                driftPoints.append(i)
        return driftPoints


def stockChangeDetector(data, CDD="hddma", driftConfidence=0.001, SCD="cusum",
                        changeConfidence=1, typeOfReturn="all"):
    priceSeries = pandas.Series(data["Adj-Close Price"].values, index=data["Date"]).dropna()
    driftPoints = conceptDriftDetector(priceSeries.tolist(), CDD, driftConfidence)

    sentimentSeries = pandas.Series(data["Sentiment"].values, index=data["Date"]).dropna()
    changePoints = sentimentChangeDetector(sentimentSeries.tolist(), SCD, changeConfidence)

    conceptDriftDays = []

    if typeOfReturn == "drift" or typeOfReturn == "all":
        for point in driftPoints:
            conceptDriftDays.append(priceSeries.index[point])

    if typeOfReturn == "change" or typeOfReturn == "all":
        for point in changePoints:
            # Needs to be done to account for days when stock market is not open
            currentDate = sentimentSeries.index[point]
            while True:
                if currentDate in priceSeries.index:
                    conceptDriftDays.append(currentDate)
                    break
                currentDate = currentDate + pandas.DateOffset(days=1)

    if typeOfReturn == "conjunction":

        driftDays = []

        for point in driftPoints:
            driftDays.append(priceSeries.index[point])

        for point in changePoints:
            currentDate = sentimentSeries.index[point]

            # Needs to be done to account for days when stock market is not open
            while True:
                if currentDate in priceSeries.index:

                    # Sort by dates closest to current
                    driftDays.sort(key=lambda i: abs(i - currentDate))
                    if abs((driftDays[0] - currentDate).days) < 7:
                        conceptDriftDays.append(currentDate)
                    break
                else:
                    currentDate = currentDate + pandas.DateOffset(days=1)

    return list(dict.fromkeys(conceptDriftDays))


def conceptDriftDetector(stream, CDD="hddma", driftConfidence=0.001):
    driftPoints = []

    if CDD == "eddm":
        EDDM = skmultiflow.drift_detection.eddm.EDDM()

        previous = 0
        for i in range(len(stream)):
            if previous < stream[i]:
                EDDM.add_element(1)
            else:
                EDDM.add_element(0)

            previous = stream[i]

            if EDDM.detected_change():
                driftPoints.append(i)
        return driftPoints

    elif CDD == "hddma":
        CDD = skmultiflow.drift_detection.hddm_a.HDDM_A(driftConfidence)
    elif CDD == "hddmw":
        CDD = skmultiflow.drift_detection.hddm_w.HDDM_W()
    elif CDD == "minps":
        CDD = minps.MINPS(20)
    elif CDD == "mysd":
        CDD = mySD.mySDDD(20)
    elif CDD == "mytandd":
        CDD = myTanDD.myTanDD(20)
    elif CDD == "ph":
        CDD = skmultiflow.drift_detection.page_hinkley.PageHinkley()

    for i in range(len(stream)):
        CDD.add_element(stream[i])
        if CDD.detected_change():
            driftPoints.append(i)
            if type(CDD) in [minps.MINPS, mySD.mySDDD, myTanDD.myTanDD]:
                CDD.reset()
    return driftPoints


def countNumberOfDrifts(stock, dataset, typeOfReturn):
    numerical = \
    pandas.read_csv("../data/" + dataset + "/Numerical/price_long_50/" + stock, index_col="Date", parse_dates=["Date"])[
        "Close"].iloc[::-1]

    sentiment = pandas.read_csv("../data/" + dataset + "/SentimentScores/NYT-Business/" + stock, header=0)
    sentiment["Date"] = pandas.to_datetime(sentiment["Date"])

    sentimentScores = {}
    for index, row in sentiment.iterrows():
        if row["Date"] in sentimentScores:
            sentimentScores[row["Date"]].append(row["sentimentScore"])
        else:
            sentimentScores[row["Date"]] = [row["sentimentScore"]]

    if dataset == "kdd17":
        startDate = pandas.Timestamp("2016-01-04")
        endDate = pandas.Timestamp("2016-12-30")
    else:
        startDate = pandas.Timestamp("2015-10-01")
        endDate = pandas.Timestamp("2015-12-31")

    allDates = []
    allPrices = []
    allSentiment = []
    delta = endDate - startDate

    for counter in range(delta.days + 1):
        date = startDate + timedelta(days=counter)

        allDates.append(date)

        if date in numerical.index:
            allPrices.append(numerical[date])
        else:
            allPrices.append(None)

        if date in sentimentScores:
            allSentiment.append(mean(sentimentScores[date]))
        else:
            allSentiment.append(None)

    dataFrame = pandas.DataFrame({"Date": allDates, "Adj-Close Price": allPrices, "Sentiment": allSentiment})
    drifts = [startDate] + stockChangeDetector(dataFrame, typeOfReturn=typeOfReturn)
    stockDays = list(pandas.Series(dataFrame["Adj-Close Price"].values, index=dataFrame["Date"]).dropna().index)

    counter = 1
    dayCounter = 0

    for day in stockDays[1:]:

        if counter >= len(drifts):
            break

        if day == drifts[counter]:
            if dayCounter < 63:
                del drifts[counter]
            else:
                counter += 1
                dayCounter = 0
        else:
            dayCounter += 1



    return len(drifts) - 1


if __name__ == "__main__":
    # Parameter Declaration
    dataset = "stocknet"  # Can be either "kdd17" or "stocknet"
    stockCode = "all"
    typeOfReturn = "conjunction"

    if stockCode == "all":
        numberOfDrifts = 0

        for stock in os.listdir("../data/" + dataset + "/Numerical/ourpped"):
            numberOfDrifts += countNumberOfDrifts(stock, dataset, typeOfReturn)
        print(numberOfDrifts)

    else:
        print(countNumberOfDrifts(stockCode + ".csv", dataset, typeOfReturn))
