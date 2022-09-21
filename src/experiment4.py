# This script evaluates an entered distance metric.

import sys
import warnings
from functions import calculateDistance
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import normalize
from tqdm import tqdm

if __name__ == "__main__":
    """""
        Parameters:
            args[0] --> Distance measure to be evaluated. Can be set to "pad" or "kl" or "hellinger"
    """""

    # Handling Parameters
    if len(sys.argv) > 1:
        distanceMetric = sys.argv[1].lower().strip()
    else:
        distanceMetric = "pad"

    # Variable Declaration
    allDistances = []

    # When the PAD Distance is used, the linear model is expected not to converge
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for dataset in tqdm(["kdd17", "stocknet", "phrasebank", "economynews"]):
        current = []
        for targetDataset in ["slsamazon", "slsimbd", "slsyelp", "stsgold"]:
            current.append(calculateDistance(dataset, targetDataset, distanceMetric))

        allDistances.append(current)

    allDistances = list(map(abs, normalize(allDistances)))

    print("Single Link --> " + str(min([min(minDistance) for minDistance in allDistances])))
    print("Complete Link --> " + str(max([max(maxDistance) for maxDistance in allDistances])))
    print("Average Link --> " + str(sum([sum(sumDistance) for sumDistance in allDistances]) / 16))
