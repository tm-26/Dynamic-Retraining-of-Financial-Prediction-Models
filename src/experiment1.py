# This script tests FinBERT on the EconomyNews dataset.

import sys
import sklearn.metrics
from finbert import testFinBERT


def write(text, filePath):
    print(text)
    if filePath != "":
        with open(filePath, 'a') as file:
            file.write(text + '\n')


if __name__ == "__main__":
    """""
    Parameters:
        args[0] --> Path of results save file
    """""

    # Handling Parameters
    if len(sys.argv) > 1:
        try:
            file = open(sys.argv[1], "w")
            file.close()
        except FileNotFoundError:
            print("Parameter Error: The file with path=\"" + sys.argv[1] + "\" could not be created")
            exit(-1)

        filePath = sys.argv[1]
    else:
        filePath = ""

    groundTruths, predicted = testFinBERT("economynews")

    write("----FinBERT Performance on The EconomyNews Dataset----", filePath)
    write("Accuracy Score = " + str(sklearn.metrics.accuracy_score(groundTruths, predicted)), filePath)
    write("Precision Score = " + str(sklearn.metrics.precision_score(groundTruths, predicted)), filePath)
    write("Recall Score = " + str(sklearn.metrics.recall_score(groundTruths, predicted)), filePath)
    write("F1 Score = " + str(sklearn.metrics.f1_score(groundTruths, predicted)), filePath)
    write("------------------------------------------------------", filePath)
