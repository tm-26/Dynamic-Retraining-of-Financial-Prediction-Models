# This script estimates the performance that FinBERT produced on some target dataset.

import csv
import os
import numpy
import pandas
import random
import sklearn
import sys
from finbert import testFinBERT
from finbert_embedding.embedding import FinbertEmbedding
from functions import calculateDistance, getData
from torch.nn.functional import kl_div
from tqdm import tqdm


def getClosestDataset(targetDataset):
    distances = {}

    for dataset in ["economynews", "slsamazon", "slsimbd", "slsyelp", "stsgold"]:
        if targetDataset == dataset:
            continue

        distances[dataset] = calculateDistance(dataset, targetDataset, "hellinger")

    return min(distances, key=distances.get)


def getDistances(data):
    values = {}

    for point in tqdm(data):
        prediction = True
        distance = round(
            kl_div(chosenEmbedding, finbertEmbedder.sentence_vector(point[0]), reduction="batchmean").item(), 4)

        if point[1] != point[2]:
            prediction = False

        if distance in values:
            values[distance].append(prediction)
        else:
            values[distance] = [prediction]

    dataFrame = pandas.DataFrame(columns=["Domain-shift Detection Metric", "Classification Drop"])

    for i, (x, y) in enumerate(values.items()):
        dataFrame.loc[i] = [x, len([prediction for prediction in y if not prediction]) / len(y)]

    return dataFrame.iloc[:, :-1].values, dataFrame.iloc[:, 1].values


if __name__ == "__main__":
    """""
        Parameters:
            args[0] --> Dataset to be used
    """""

    if len(sys.argv) > 1:
        targetDataset = sys.argv[1].lower().strip()
    else:
        targetDataset = "kdd17"

    # Variable Declaration
    formattedTrainData = []
    finbertEmbedder = FinbertEmbedding(model_path="../models/fin_model")
    iterationCounter = 1
    realAccuracies = []
    estimatedAccuracies = []

    if targetDataset == "kdd17" or targetDataset == "stocknet":

        # Find closest dataset
        dataset = getClosestDataset(targetDataset)

        # Get data of closest dataset
        allTrainData = testFinBERT(dataset, True)

        i = 0
        while i < len(allTrainData[0]):
            formattedTrainData.append((allTrainData[0][i], allTrainData[1][i], allTrainData[2][i]))
            i += 1

        chosenEmbedding = 0

        for point in formattedTrainData:
            if point[1] == point[2]:
                chosenEmbedding = finbertEmbedder.sentence_vector(point[0])
                break
        xTrain, yTrain = getDistances(formattedTrainData)
        print(xTrain)
        print(yTrain)

        model = sklearn.linear_model.LinearRegression()
        model.fit(xTrain, yTrain)

        # Get unlabelled test data
        testData = []
        for stockCode in tqdm(os.listdir("../data/" + targetDataset + "/sentiment-scores")):
            with open("../data/" + targetDataset + "/sentiment-scores/" + stockCode, 'r',
                      encoding="UTF-8") as file:
                csvReader = csv.reader(file)
                next(csvReader)  # skip header

                for row in tqdm(csvReader):
                    testData.append(round(kl_div(chosenEmbedding, finbertEmbedder.sentence_vector(row[1]), reduction="batchmean").item(), 4))

        while iterationCounter <= 10:
            pseudoTruths = []

            predictions = model.predict(numpy.array(testData).reshape(-1, 1))

            for prediction in predictions:
                if prediction == 0:
                    pseudoTruths.append(True)
                elif prediction == 1:
                    pseudoTruths.append(False)
                else:
                    pseudoTruths.append(random.random() > prediction)

            estimatedAccuracy = len([pseudoTruth for pseudoTruth in pseudoTruths if pseudoTruth]) / len(
                pseudoTruths)

            estimatedAccuracies.append(estimatedAccuracy)

            print("Estimated Accuracy = " + str(estimatedAccuracy))
            iterationCounter += 1

    else:

        # Generate real accuracy score of FinBERT on targetDataset
        allTestData = testFinBERT(targetDataset, True)
        realACC = sklearn.metrics.accuracy_score(allTestData[1], allTestData[2])

        dataset = getClosestDataset(targetDataset)

        formattedTestData = []
        chosenEmbedding = 0
        allTrainData = testFinBERT(dataset, True)

        i = 0
        while i < len(allTrainData[0]):
            formattedTrainData.append((allTrainData[0][i], allTrainData[1][i], allTrainData[2][i]))
            i += 1

        i = 0
        while i < len(allTestData[0]):
            formattedTestData.append((allTestData[0][i], allTestData[1][i], allTestData[2][i]))
            i += 1

        for point in formattedTrainData:
            if point[1] == point[2]:
                chosenEmbedding = finbertEmbedder.sentence_vector(point[0])
                break

        xTrain, yTrain = getDistances(formattedTrainData)

        model = sklearn.linear_model.LinearRegression()
        model.fit(xTrain, yTrain)

        while iterationCounter <= 10:

            random.shuffle(formattedTestData)

            xTest, yTest = getDistances(formattedTestData)

            yPredicted = model.predict(xTest)

            pseudoTruths = []

            for prediction in yPredicted:
                if prediction == 0:
                    pseudoTruths.append(True)
                elif prediction == 1:
                    pseudoTruths.append(False)
                else:
                    pseudoTruths.append(random.random() > prediction)

            estimatedAccuracy = len([pseudoTruth for pseudoTruth in pseudoTruths if pseudoTruth]) / len(
                pseudoTruths)

            print("Estimated Accuracy = " + str(estimatedAccuracy))

            estimatedAccuracies.append(estimatedAccuracy)

            realPredictions = []
            for point in formattedTestData:
                if point[1] == point[2]:
                    realPredictions.append(True)
                else:
                    realPredictions.append(False)

            realAccuracy = len([realPrediction for realPrediction in realPredictions if realPrediction]) / len(realPredictions)

            print("Real Accuracy = " + str(realAccuracy))

            realAccuracies.append(realAccuracy)

            iterationCounter += 1

        print("All estimated accuracies:")
        print("-------------------------")
        for i, estimatedAccuracy in enumerate(estimatedAccuracies):
            print("Estimated accuracy #" + str(i + 1) + " = " + str(estimatedAccuracy))
        print("-------------------------")
        print("MAE = " + str(sklearn.metrics.mean_absolute_error(realAccuracies, estimatedAccuracies)))

        i = 0
        difference = []
        while i < len(realAccuracies):
            difference.append(abs(realAccuracies[i] - estimatedAccuracies[i]))
            i += 1

        print("MAX = " + str(max(difference)))


