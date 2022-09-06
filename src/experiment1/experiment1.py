import json
import sys
import sklearn.metrics
from finbert import predict2
from statistics import mean
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm


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

    # Variable Declaration
    finBert = AutoModelForSequenceClassification.from_pretrained("../../models/FinBert", cache_dir=None, num_labels=3)
    groundTruths = []
    predicted = []

    with open("../../data/economyNews/economyNews.json", encoding="utf-8") as file:
        data = json.load(file)
        for point in tqdm(data):
            scores = []

            answer = predict2(point["headlineText"], finBert)
            for j, score in enumerate(answer["sentiment_score"]):
                scores.append(score)

            answer = predict2(point["headlineTitle"], finBert)
            for j, score in enumerate(answer["sentiment_score"]):
                scores.append(score)

            groundTruths.append(point["classification"])

            if mean(scores) > 0:
                predicted.append(1)
            else:
                predicted.append(-1)

    write("----FinBERT Performance on The EconomyNews Dataset----", filePath)
    write("Accuracy Score = " + str(sklearn.metrics.accuracy_score(groundTruths, predicted)), filePath)
    write("Precision Score = " + str(sklearn.metrics.precision_score(groundTruths, predicted)), filePath)
    write("Recall Score = " + str(sklearn.metrics.recall_score(groundTruths, predicted)), filePath)
    write("F1 Score = " + str(sklearn.metrics.f1_score(groundTruths, predicted)), filePath)
    write("------------------------------------------------------", filePath)
