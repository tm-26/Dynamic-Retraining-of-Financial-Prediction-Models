"""
Data extracted from New York Times:
-abstract
-snippet
-lead_paragraph
-headline
--main
--kicker
--content_kicker
--print_headline
--name
--seo
--sub
-keywords
--value

Ensured that the above contains all the important data by checking different data values from various timestamps
"""

import csv
import json
import os
import requests
import shutil
import time
import yfinance
from tqdm import tqdm

# Global variable declaration
unwantedCharacters = "!#$%\'()*+,-./:;<=>?@[\\]^_`{|}~012456789"
stopwords = []
words = []
datasetName = ""


def saveArticle(stockCode, article):

    if os.path.exists(datasetName + "/NYT-Business/Individual Articles/" + stockCode.upper() + '/' + article["pub_date"][0:10] + ".json"):
        with open(datasetName + "/NYT-Business/Individual Articles/" + stockCode.upper() + '/' + article["pub_date"][0:10] + ".json", "r+",
                  encoding="utf-8") as saveFile:
            currentArticles = json.load(saveFile)
            if type(currentArticles) is list:
                currentArticles.append(article)
            else:
                currentArticles = [currentArticles, article]
            saveFile.seek(0)
            saveFile.write(json.dumps(currentArticles))
    else:
        with open(datasetName + "/NYT-Business/" + stockCode.upper() + '/' + article["pub_date"][0:10] + ".json", "w+",
                  encoding="utf-8") as saveFile:
            json.dump(article, saveFile)


def compare(stockName, word):
    # If stockName has multiple words
    if " " in stockName:
        save = True
        for j, current in enumerate(stockName.split()):
            if i + j >= len(words):
                break
            if current != words[i + j]:
                save = False
                break
        if save:
            return True
        return False

    if word == stockName:
        return True

    return False


def validate(words):
    if words is not None:
        return [word for word in words.translate(str.maketrans('', '', unwantedCharacters)).lower().split() if
                word not in stopwords]
    return []


if __name__ == "__main__":

    # Parameter Declaration
    remakeFiles = False
    remakeStockNames = False
    datasetName = "stocknet"  # Can be either "kdd17" or "stocknet"
    createOurpped = True

    os.chdir("../data")

    # Variable Declaration
    stockCodes = os.listdir(datasetName + "/numerical/ourpped")
    stockNames = []

    # Get all stock codes
    for i in range(len(stockCodes)):
        stockCodes[i] = stockCodes[i][:-4]

    # Make directories and delete previously extracted articles
    if remakeFiles:
        if os.path.exists(datasetName + "/NYT-Business"):
            shutil.rmtree(datasetName + "/NYT-Business")
            os.mkdir(datasetName + "/NYT-Business")
        for code in stockCodes:
            os.mkdir(datasetName + "/NYT-Business/" + code)

    if not os.path.isdir("NYT-Business"):

        if not os.path.isdir("NYT"):
            os.mkdir("NYT")

            # First get all NYT data
            year = "2007"
            month = '1'
            key = ""  # Insert NYT API Key here
            filterQuery = "\"source:(\"The New York Times\")"
            done = os.listdir("NYT")
            i = 0

            print("Getting all NYT data using API")

            for year in tqdm(range(2007, 2017)):
                for month in range(1, 13):
                    if str(month) + '-' + str(year) + ".json" not in done:
                        query_url = "https://api.nytimes.com/svc/archive/v1/" + str(year) + '/' + str(
                            month) + ".json?api-key=" + key + "&fq=" + filterQuery
                        response = requests.get(query_url).json()
                        with open("../dataset/NYT/" + str(month) + '-' + str(year) + ".json", "w+") as file:
                            json.dump(response["response"], file)
                        time.sleep(6.1)
                    i += 1

        data = os.listdir("NYT")

        os.mkdir("NYT-Business")

        print("Getting all NYT-Business data")

        for i in data:
            with open("NYT/" + i) as file:
                currentBusiness = []
                currentAll = json.load(file)["docs"]
                for j in currentAll:
                    if j["section_name"].lower() in ["business", "business day"] or j["news_desk"].lower() in \
                            ["business", "business day"]:
                        currentBusiness.append(j)
            with open("NYT-Business/" + i, "w+", encoding="utf-8") as file:
                json.dump(currentBusiness, file, ensure_ascii=False)

    # Create stockNames.txt
    if remakeStockNames or not os.path.exists(datasetName + "/stockNames.txt"):
        if os.path.exists(datasetName + "/stockNames.txt"):
            os.remove(datasetName + "/stockNames.txt")
        with open(datasetName + "/stockNames.txt", "w+") as file:
            for code in tqdm(stockCodes):
                try:
                    file.write(yfinance.Ticker(code).info["longName"] + "\n")
                except (KeyError, TypeError):
                    file.write(code + "\n")
        print("Manual checking of stock names is now required.")
        print("Rerun script with remakeStockNames=False when stock names are checked.")
        exit()

    if not createOurpped:

        # Get stock names
        with open(datasetName + "/stockNames.txt") as file:
            stockNames = file.read().split('\n')

        # Generate dictionary with stock codes and names
        stocks = {}
        for i in range(len(stockNames)):

            if '/' in stockNames[i]:
                stocks[stockCodes[i].lower()] = [c.lower() for c in stockNames[i].split('/')]
            else:
                stocks[stockCodes[i].lower()] = stockNames[i].lower()

        # Get stopwords
        with open("stopwords.txt") as stopwordsFile:
            stopwords = stopwordsFile.read().split("\n")
            # Get articles
            for fileName in tqdm(os.listdir("NYT-Business")):

                if datasetName == "stocknet" and int(fileName[:-5][-4:]) < 2014:
                    continue

                with open("NYT-Business/" + fileName, encoding="utf-8") as file:
                    data = json.load(file)
                    for article in data:
                        # Get all needed data from article
                        words = validate(article["abstract"]) + validate(article["snippet"]) + \
                                validate(article["lead_paragraph"]) + validate(article["headline"]["main"]) + \
                                validate(article["headline"]["kicker"]) + \
                                validate(article["headline"]["content_kicker"]) + \
                                validate(article["headline"]["print_headline"]) + \
                                validate(article["headline"]["name"]) + validate(article["headline"]["seo"]) + \
                                validate(article["headline"]["sub"])

                        if article["keywords"] is not None:
                            for keyword in article["keywords"]:
                                words.extend(validate(keyword["value"]))

                        savedAlready = []

                        for i, word in enumerate(words):
                            for stock in stocks.items():

                                if stock[0] in savedAlready:
                                    continue

                                # Check if stock code in article
                                if stock[0] == word:
                                    saveArticle(stock[0], article)
                                    savedAlready.append(stock[0])
                                    continue

                                generated = False

                                # Check if stock name in article

                                if isinstance(stock[1], list):
                                    for current in stock[1]:
                                        if compare(current, word):
                                            saveArticle(stock[0], article)
                                            savedAlready.append(stock[0])
                                            generated = True
                                            break
                                else:
                                    if compare(stock[1], word):
                                        saveArticle(stock[0], article)
                                        savedAlready.append(stock[0])
                                        continue
                                if generated:
                                    continue

    for stock in tqdm(os.listdir(datasetName + "/NYT-Business/Individual Articles")):
        saveMe = [["Date", "Articles"]]
        for article in os.listdir(datasetName + "/NYT-Business/Individual Articles/" + stock):
            with open(datasetName + "/NYT-Business/Individual Articles/" + stock + '/' + article, "r", encoding="utf-8") as file:
                saveMe.append([article[:-5], file.read()])
        with open(datasetName + "/NYT-Business/ourpped/" + stock + ".csv", "w+", encoding="utf-8", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(saveMe)
