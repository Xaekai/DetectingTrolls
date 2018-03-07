import keras
import emoji
import re
import csv
import spacy
import datetime
import time
import numpy as np


emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
def filterEmojis(s):
    s = re.sub(r, ' ', s)
    return s


def filterEmojisFromFile(path):
    output = []
    filtered = []

    with open(path, 'r', encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            output.append((row['text'], row['userid']))
    print(datetime.datetime.now())
    return output


def preprocessInput(path):
    lines = []

    with open(path, 'r') as inputFile:
        for line in inputFile:
            lines.append(line)
    with open(path + "_no_emojis", 'w', encoding="utf8") as outputFile:
        for line in lines:
            outputFile.write(filterEmojis(line))
















if __name__ == "__main__":
    now = time.time()
    print("Preprocessing begins at " + str(now))
    nlp = spacy.load('en_vectors_web_lg')
    bot_tweets = filterEmojisFromFile('../data/bot_tweets.csv')
    bot_labels = np.ones(len(bot_tweets), dtype=int)
    regular_tweets = []

    # a bit brutish but I don't want to deal with a lot of file IO
    for chunkNumber in range(0, 7):
        #regular_tweets += filterEmojisFromFile('../data/non_bot_chunk_' + str(chunkNumber) + '.csv')
        preprocessInput('../data/non_bot_chunk_' + str(chunkNumber) + '.csv')
    regular_labels = np.zeros(len(regular_tweets), dtype=int)
    print("Preprocessing complete at " + str(now))
