# This script counts the number of articles in a given dataset

import csv
import json
import os
import sys
from matplotlib import pyplot


if __name__ == "__main__":
    """""
        Parameters:
            args[0] --> Controls which dataset is going to be selected. Can be set to "kdd17" or "stocknet"
    """""

    # Handling Parameters
    if len(sys.argv) > 1:
        sys.argv[1] = sys.argv[1].lower().strip()
        if sys.argv[1] != "kdd17" and sys.argv[1] != "stocknet":
            print("Parameter Error: The entered dataset could not be found" + sys.argv[1])
            exit(-1)
        dataset = sys.argv[1]
    else:
        dataset = "kdd17"

    # Variable Declaration
    counter = 0
    yearlyCounter = {"2007": 0, "2008": 0, "2009": 0, "2010": 0, "2011": 0, "2012": 0, "2013": 0, "2014": 0, "2015": 0, "2016": 0}

    for stock in os.listdir("../../data/" + dataset + "/NYT-Business/ourpped"):
        with open("../../data/" + dataset + "/NYT-Business/ourpped/" + stock, encoding="utf-8") as file:
            csvReader = csv.reader(file)
            next(csvReader)  # Skip headers
            csvReader = list(csvReader)
            for row in csvReader:

                data = json.loads(row[1])
                if type(data) is list:
                    yearlyCounter[row[0][:4]] = yearlyCounter[row[0][:4]] + len(data)
                    counter = counter + len(data)
                else:
                    counter += 1
                    yearlyCounter[row[0][:4]] += 1
    print("The " + dataset + " dataset contains " + str(counter) + " articles")

    yearlyCounter = {x: y for x, y in yearlyCounter.items() if y != 0}

    figure, axes = pyplot.subplots()
    axes.bar(*zip(*yearlyCounter.items()))

    axes.set_xlabel("Label")
    axes.set_ylabel("Numer of articles")
    pyplot.show()

