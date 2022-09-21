import csv
import json
import numpy
import os
import pandas
import pickle
import torch
from finbert_embedding.embedding import FinbertEmbedding
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm


def createEmbeddings(dataset):

    print("Embeddings file for the " + dataset + " dataset is missing.")
    print("Creating it now...")

    finbertEmbedder = FinbertEmbedding(model_path="../models/fin_model")
    saveMe = []

    if dataset == "kdd17" or dataset == "stocknet":
        for stock in tqdm(os.listdir("../data/" + dataset + "/NYT-Business/ourpped")):
            with open("../data/" + dataset + "/NYT-Business/ourpped/" + stock, encoding="utf-8") as file:
                csvReader = csv.reader(file)
                next(csvReader)  # Skip headers
                for row in csvReader:
                    data = json.loads(row[1])
                    if type(data) is list:
                        for article in data:
                            saveMe.append([article["abstract"], finbertEmbedder.sentence_vector(article["abstract"])])
                    else:
                        saveMe.append([data["abstract"], finbertEmbedder.sentence_vector(data["abstract"])])
        pickle.dump(saveMe, open("../data/" + dataset + "/NYT-Business/embeddings.pkl", "wb"))
    elif dataset == "economynews":
        with open("../data/economyNews/economyNews.json", encoding="UTF-8") as file:
            economyNews = json.load(file)
            for point in tqdm(economyNews):
                saveMe.append([point["headlineText"], finbertEmbedder.sentence_vector(point["headlineText"])])

        pickle.dump(saveMe, open("../data/economyNews/embeddings.pkl", "wb"))
    elif dataset == "phrasebank":
        with open("../data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt", encoding="ISO-8859-1") as file:
            lines = file.readlines()
            for line in tqdm(lines):
                line = line.rsplit(' ', 1)[0]
                saveMe.append([line, finbertEmbedder.sentence_vector(line)])

        pickle.dump(saveMe, open("../data/FinancialPhraseBank-v1.0/embeddings.pkl", "wb"))
    elif dataset == "slsamazon":
        with open("../data/SLS/amazon_cells_labelled.txt") as file:
            lines = file.readlines()
            for line in tqdm(lines):
                line = line[:-3]
                saveMe.append([line, finbertEmbedder.sentence_vector(line)])
        pickle.dump(saveMe, open("../data/SLS/amazonEmbeddings.pkl", "wb"))
    elif dataset == "slsimbd":
        with open("../data/SLS/imdb_labelled.txt") as file:
            lines = file.readlines()
            for line in tqdm(lines):
                line = line[:-3]
                saveMe.append([line, finbertEmbedder.sentence_vector(line)])
        pickle.dump(saveMe, open("../data/SLS/imbdEmbeddings.pkl", "wb"))
    elif dataset == "slsyelp":
        with open("../data/SLS/yelp_labelled.txt") as file:
            lines = file.readlines()
            for line in tqdm(lines):
                line = line[:-3]
                saveMe.append([line, finbertEmbedder.sentence_vector(line)])
        pickle.dump(saveMe, open("../data/SLS/yelpEmbeddings.pkl", "wb"))
    elif dataset == "stsgold":
        with open("../data/STS-Gold/sts_gold_tweet.csv") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip headers
            for row in tqdm(reader):
                saveMe.append([row[2], finbertEmbedder.sentence_vector(row[2])])
        pickle.dump(saveMe, open("../data/STS-Gold/embeddings.pkl", "wb"))

    else:
        print("Parameter error: dataset=" + str(dataset) + " is not an accepted input")
        exit(-2)


def getEmbeddings(filePath, dataset):

    if not os.path.exists(filePath):
        createEmbeddings(dataset)

    dataList = []

    embeddings = pickle.load(open(filePath, "rb"))
    for embedding in embeddings:
        dataList.append(embedding[1])
    return dataList


def equalizeDate(p, q):
    if p.shape[0] > q.shape[0]:
        q = torch.cat((q, torch.zeros([p.shape[0] - q.shape[0], 768])))
    elif p.shape[0] < q.shape[0]:
        p = torch.cat((p, torch.zeros([q.shape[0] - p.shape[0], 768])))

    return p, q


def getData(dataset, label):

    if label == "embedding":
        if dataset == "kdd17" or dataset == "stocknet":
            return getEmbeddings("../data/" + dataset + "/NYT-Business/embeddings.pkl", dataset)
        elif dataset == "economynews":
            return getEmbeddings("../data/economyNews/embeddings.pkl", dataset)
        elif dataset == "phrasebank":
            return getEmbeddings("../data/FinancialPhraseBank-v1.0/embeddings.pkl", dataset)
        elif dataset == "slsamazon":
            return getEmbeddings("../data/SLS/amazonEmbeddings.pkl", dataset)
        elif dataset == "slsimbd":
            return getEmbeddings("../data/SLS/imbdEmbeddings.pkl", dataset)
        elif dataset == "slsyelp":
            return getEmbeddings("../data/SLS/yelpEmbeddings.pkl", dataset)
        elif dataset == "stsgold":
            return getEmbeddings("../data/STS-Gold/embeddings.pkl", dataset)

    dataList = []
    if dataset == "kdd17" or dataset == "stocknet":
        for stock in os.listdir("../data/" + dataset + "/NYT-Business/ourpped"):
            with open("../data/" + dataset + "/NYT-Business/ourpped/" + stock, encoding="utf-8") as file:
                csvReader = csv.reader(file)
                next(csvReader)  # Skip headers
                for row in csvReader:
                    data = json.loads(row[1])
                    if type(data) is list:
                        for article in data:
                            dataList.append([article["abstract"], label])
                    else:
                        dataList.append([data["abstract"], label])
    elif dataset == "economynews":
        with open("../data/economyNews/economyNews.json", encoding="UTF-8") as file:
            economyNews = json.load(file)
            for point in economyNews:
                dataList.append([point["headlineText"], label])
    elif dataset == "phrasebank":
        with open("../data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt", encoding="ISO-8859-1") as file:
            lines = file.readlines()
            for line in lines:
                dataList.append([line.rsplit(' ', 1)[0], label])
    elif dataset == "slsamazon":
        with open("../data/SLS/amazon_cells_labelled.txt") as file:
            lines = file.readlines()
            for line in lines:
                dataList.append([line[:-3], label])
    elif dataset == "slsimbd":
        with open("../data/SLS/imdb_labelled.txt") as file:
            lines = file.readlines()
            for line in lines:
                dataList.append([line[:-3], label])
    elif dataset == "slsyelp":
        with open("../data/SLS/yelp_labelled.txt") as file:
            lines = file.readlines()
            for line in lines:
                dataList.append([line[:-3], label])
    elif dataset == "stsgold":
        with open("../data/STS-Gold/sts_gold_tweet.csv") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip headers
            for row in reader:
                dataList.append([row[2], label])

    else:
        print("Parameter error: dataset=" + str(dataset) + " is not an accepted input")
        exit(-2)

    return dataList


def calculateDistance(sourceDataset, targetDataset, distanceMetric):
    if distanceMetric == "pad":
        iterationCounter = 0
        dataList = getData(sourceDataset, 0) + getData(targetDataset, 1)
        padList = []

        while iterationCounter < 10:
            dataFrame = pandas.DataFrame(dataList, columns=["Text", "Origin"])
            xTrain, xTest, yTrain, yTest = train_test_split(
                TfidfVectorizer(strip_accents="unicode").fit_transform(dataFrame["Text"]), dataFrame["Origin"],
                test_size=0.2)
            svcModel = make_pipeline(StandardScaler(with_mean=False), LinearSVC(class_weight="balanced"))
            svcModel.fit(xTrain, yTrain)
            predicted = svcModel.predict(xTest)
            mae = mean_absolute_error(yTest, predicted)
            padList.append(2 * (1 - 2 * mae))

            iterationCounter += 1

        return sum(padList) / len(padList)

    elif distanceMetric == "kl":
        """
        N.B, KL Distance has the limitation that the two lists should be equal. If they are not the longer one 
        will be shortened to the length of the shorter one. 
        """

        sourceData, targetData = equalizeDate(torch.stack(getData(sourceDataset, "embedding")),
                                              torch.stack(getData(targetDataset, "embedding")))

        return torch.nn.functional.kl_div(sourceData, targetData, reduction="batchmean").item()

    elif distanceMetric == "hellinger":

        sourceData, targetData = equalizeDate(torch.stack(getData(sourceDataset, "embedding")), torch.stack(getData(targetDataset, "embedding")))

        return numpy.sqrt(numpy.nansum((numpy.sqrt(sourceData) - numpy.sqrt(targetData)) ** 2)) / numpy.sqrt(2)

    elif distanceMetric == "tv":

        sourceData, targetData = equalizeDate(torch.stack(getData(sourceDataset, "embedding")), torch.stack(getData(targetDataset, "embedding")))

        return numpy.sqrt(numpy.nansum((numpy.sqrt(sourceData) - numpy.sqrt(targetData)) ** 2)) / numpy.sqrt(2)

    else:
        print("Parameter error: distanceMetric=" + str(distanceMetric) + " is not an accepted input")
        exit(-1)


def plot(path, stockCode, myType, yLabel="Adj Close Price ($)"):
    # Get data
    data = pandas.read_csv(path, usecols=["Date", myType])
    data["Date"] = pandas.to_datetime(data["Date"])

    # Plot it
    figure, axes = pyplot.subplots()
    axes.plot_date(data["Date"], data[myType], '-')

    pyplot.title("Historical data of " + stockCode)
    axes.set_xlabel("Time")
    axes.set_ylabel(yLabel)
    axes.tick_params(axis='x', labelsize=9)
    pyplot.xticks(fontsize=6)
    pyplot.show()


def validateDataset(myInput):
    myInput = myInput.lower().strip()
    if myInput != "kdd17" and myInput != "stocknet":
        print("Parameter Error: The entered dataset could not be found")
        exit(-1)

    return myInput


def validatePlot(path, stockCode, myType, yLabel="Adj Close Price ($)"):
    try:
        plot(path, stockCode, myType, yLabel)
    except FileNotFoundError:
        print("Parameter Error: Stock=\"" + stockCode + "\" could not be found")
        exit(-1)
