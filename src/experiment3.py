"""""
This script uses a regression based estimation technique to estimate the performance that FinBERT produced on some
target dataset.
"""""

import numpy
import warnings
from finbert import testFinBERT
from functions import calculateDistance
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


if __name__ == "__main__":

    # Variable Declaration
    distances = []
    ACCDrop = []

    # When the PAD Distance is used, the linear model is expected not to converge
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    groundTruths, predicted = testFinBERT("phrasebank")
    mainACC = accuracy_score(groundTruths, predicted)

    for targetDataset in ["economynews", "slsamazon", "slsimbd", "slsyelp", "stsgold"]:

        # Calculate distance from source dataset to target dataset
        distances.append(calculateDistance("phrasebank", targetDataset, "pad"))
        print("PAD distance from phrasebank to " + targetDataset + " = " + str(distances[-1]))

        groundTruths, predicted = testFinBERT(targetDataset)
        accuracy = accuracy_score(groundTruths, predicted)
        print(accuracy)
        ACCDrop.append(mainACC - accuracy)
        print(mainACC - accuracy)
        print("ACC drop of " + str(ACCDrop[-1]) + " on the " + targetDataset + " dataset")
        print("-----------------------------------------------------------------------------------------------")

    distanceKDD17 = calculateDistance("phrasebank", "kdd17", "pad")
    print("PAD distance from phrasebank to kdd17 = " + str(distanceKDD17))

    finalRegressionModel = LinearRegression()
    finalRegressionModel.fit(numpy.array(distances).reshape(-1, 1), numpy.array(ACCDrop).reshape(-1, 1))
    resultKDD17 = finalRegressionModel.predict(numpy.array(distanceKDD17).reshape(1, -1))[0][0]
    print("Estimated ACC score on kdd17 = " + str(mainACC - resultKDD17))

    distanceStockNet = calculateDistance("phrasebank", "stocknet", "pad")
    print("PAD distance from phrasebank to stocknet = " + str(distanceStockNet))

    resultStockNet = finalRegressionModel.predict(numpy.array(distanceStockNet).reshape(1, -1))[0][0]
    print("Estimated ACC score on stocknet = " + str(mainACC - resultStockNet))
